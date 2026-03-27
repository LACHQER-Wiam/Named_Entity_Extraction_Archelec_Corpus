"""
Script 02 — Chargement et nettoyage des textes OCRisés
=======================================================
Explore la structure du repo arkindex_archelec, relie chaque
document CSV à son fichier texte, applique un preprocessing
adapté à CamemBERT, et sauvegarde le résultat en JSON.
"""

import os
import re
import json
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
PROC_DIR    = BASE_DIR / "data" / "processed"
CSV_PATH    = PROC_DIR / "archelec_1973_1978.csv"
OUT_JSON    = PROC_DIR / "texts_clean.json"

# Dossier racine du repo cloné (modifiable si cloné ailleurs)
ARKINDEX_DIR = BASE_DIR.parent / "arkindex_archelec"

# Limite de tokens CamemBERT (512 − 2 tokens spéciaux = 510 tokens utiles)
MAX_TOKENS = 510
CAMEMBERT_MODEL = "camembert-base"

# ─── Preprocessing ────────────────────────────────────────────────────────────

def nettoyer_texte(texte: str) -> str:
    """
    Nettoie un texte OCRisé en vue de l'annotation NER.

    - Supprime les caractères de contrôle (sauf \\n)
    - Normalise les espaces multiples
    - Normalise les apostrophes typographiques → apostrophe droite
    - Supprime les lignes vides multiples
    - Conserve les accents et la casse originale
    """
    if not isinstance(texte, str):
        return ""

    # 1. Supprimer les caractères de contrôle (\\x00–\\x1f sauf \\n et \\t)
    texte = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texte)
    texte = texte.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normaliser les apostrophes typographiques
    texte = texte.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    texte = texte.replace("\u00ab", '"').replace("\u00bb", '"')
    texte = texte.replace("\u201c", '"').replace("\u201d", '"')

    # 3. Normaliser les tirets longs → tiret court
    texte = texte.replace("\u2014", "-").replace("\u2013", "-").replace("\u2012", "-")

    # 4. Supprimer les espaces multiples (en gardant les sauts de ligne)
    lignes = texte.split("\n")
    lignes = [re.sub(r"[ \t]+", " ", ligne).strip() for ligne in lignes]

    # 5. Supprimer les lignes vides consécutives (max 1 ligne vide)
    lignes_clean = []
    ligne_vide_precedente = False
    for ligne in lignes:
        if ligne == "":
            if not ligne_vide_precedente:
                lignes_clean.append(ligne)
            ligne_vide_precedente = True
        else:
            lignes_clean.append(ligne)
            ligne_vide_precedente = False

    return "\n".join(lignes_clean).strip()


def tronquer_a_512_tokens(texte: str, tokenizer) -> tuple[str, int]:
    """
    Tronque le texte pour qu'il ne dépasse pas MAX_TOKENS tokens CamemBERT.

    Retourne (texte_tronqué, nombre_de_tokens).
    La troncature se fait au niveau des tokens (pas des caractères),
    puis le texte est reconstruit depuis les tokens pour garder la cohérence.
    """
    tokens = tokenizer.encode(texte, add_special_tokens=False)
    nb_tokens = len(tokens)

    if nb_tokens <= MAX_TOKENS:
        return texte, nb_tokens

    # Tronquer et décoder pour obtenir le texte correspondant
    tokens_tronques = tokens[:MAX_TOKENS]
    texte_tronque = tokenizer.decode(tokens_tronques, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    return texte_tronque, MAX_TOKENS


# ─── Exploration de la structure du repo ─────────────────────────────────────

def afficher_arborescence(dossier: Path, profondeur_max: int = 3,
                           prefixe: str = "", profondeur: int = 0) -> None:
    """
    Affiche l'arborescence d'un dossier de façon récursive.

    S'arrête à profondeur_max pour éviter un affichage trop verbeux.
    """
    if profondeur > profondeur_max:
        return

    try:
        items = sorted(dossier.iterdir())
    except PermissionError:
        return

    # Limiter l'affichage à 20 items par dossier
    items_affiches = items[:20]
    tronque = len(items) > 20

    for i, item in enumerate(items_affiches):
        connecteur = "└── " if i == len(items_affiches) - 1 and not tronque else "├── "
        print(f"{prefixe}{connecteur}{item.name}")
        if item.is_dir():
            extension = "    " if connecteur == "└── " else "│   "
            afficher_arborescence(item, profondeur_max,
                                  prefixe + extension, profondeur + 1)

    if tronque:
        print(f"{prefixe}└── … ({len(items) - 20} éléments supplémentaires)")


def detecter_structure_repo(arkindex_dir: Path) -> dict:
    """
    Analyse la structure du repo pour identifier comment les fichiers
    texte sont organisés et nommés.

    Retourne un dict avec :
    - 'extensions' : extensions trouvées
    - 'exemple_fichiers' : quelques chemins d'exemple
    - 'nb_fichiers' : nombre total de fichiers texte
    """
    if not arkindex_dir.exists():
        return {"erreur": f"Dossier introuvable : {arkindex_dir}"}

    extensions = {}
    exemples = []
    nb_total = 0

    for fichier in arkindex_dir.rglob("*"):
        if fichier.is_file():
            ext = fichier.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
            nb_total += 1
            if len(exemples) < 10:
                exemples.append(str(fichier.relative_to(arkindex_dir)))

    return {
        "extensions": extensions,
        "exemple_fichiers": exemples,
        "nb_fichiers": nb_total,
    }


# ─── Liaison CSV ↔ fichiers texte ────────────────────────────────────────────

def trouver_fichier_texte(doc_id: str, arkindex_dir: Path,
                           index_fichiers: dict) -> Path | None:
    """
    Cherche le fichier texte correspondant à un document CSV.

    Stratégies (dans l'ordre) :
    1. Correspondance exacte sur l'id
    2. L'id est contenu dans le nom du fichier
    3. Le nom du fichier est contenu dans l'id
    """
    if not doc_id:
        return None

    doc_id_str = str(doc_id).strip()

    # Stratégie 1 : correspondance exacte
    for ext in [".txt", ".text", ""]:
        cle = doc_id_str + ext
        if cle in index_fichiers:
            return index_fichiers[cle]

    # Stratégie 2 : l'id est contenu dans le nom de fichier
    doc_id_lower = doc_id_str.lower()
    for nom, chemin in index_fichiers.items():
        if doc_id_lower in nom.lower():
            return chemin

    # Stratégie 3 : le nom de fichier (sans extension) est contenu dans l'id
    for nom, chemin in index_fichiers.items():
        nom_sans_ext = Path(nom).stem.lower()
        if nom_sans_ext and nom_sans_ext in doc_id_lower:
            return chemin

    return None


def construire_index_fichiers(arkindex_dir: Path,
                               extensions: list[str] | None = None) -> dict:
    """
    Construit un index {nom_relatif: Path} pour tous les fichiers texte.

    Utilisé pour accélérer la recherche de fichiers.
    """
    if extensions is None:
        extensions = [".txt", ".text", ""]

    index = {}
    for fichier in arkindex_dir.rglob("*"):
        if fichier.is_file() and fichier.suffix.lower() in extensions:
            # Clé = nom du fichier seul (avec extension)
            index[fichier.name] = fichier
            # Clé = chemin relatif complet
            index[str(fichier.relative_to(arkindex_dir))] = fichier

    return index


def lire_fichier_texte(chemin: Path) -> str:
    """
    Lit un fichier texte avec gestion des encodages (UTF-8 puis latin-1).
    """
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return chemin.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise IOError(f"Impossible de lire {chemin} avec les encodages testés.")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script de chargement des textes."""
    print("=" * 60)
    print("ÉTAPE 2 — Chargement et nettoyage des textes")
    print("=" * 60)

    # ── 1. Vérification du CSV filtré ───────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"[ERREUR] CSV introuvable : {CSV_PATH}")
        print("Lancez d'abord le script 01_explore_data.py")
        raise SystemExit(1)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"\nCSV chargé : {len(df)} documents")

    # ── 2. Exploration du repo arkindex ─────────────────────────────────────
    print(f"\nRecherche du repo arkindex dans : {ARKINDEX_DIR}")
    if not ARKINDEX_DIR.exists():
        print(f"\n[ERREUR] Repo introuvable : {ARKINDEX_DIR}")
        print("Clonez le repo avec :")
        print("  git clone https://gitlab.teklia.com/ckermorvant/arkindex_archelec")
        print(f"  (dans le dossier parent : {BASE_DIR.parent})")
        raise SystemExit(1)

    print("\nArborescence du repo (3 niveaux) :")
    print(str(ARKINDEX_DIR))
    afficher_arborescence(ARKINDEX_DIR, profondeur_max=3)

    info_repo = detecter_structure_repo(ARKINDEX_DIR)
    if "erreur" in info_repo:
        print(f"[ERREUR] {info_repo['erreur']}")
        raise SystemExit(1)

    print(f"\nExtensions trouvées : {info_repo['extensions']}")
    print(f"Nombre total de fichiers : {info_repo['nb_fichiers']}")
    print("Exemples de chemins :")
    for ex in info_repo["exemple_fichiers"]:
        print(f"  {ex}")

    # ── 3. Construction de l'index de fichiers ──────────────────────────────
    print("\nConstruction de l'index des fichiers texte…")
    extensions_texte = [".txt", ".text", ""]
    index_fichiers = construire_index_fichiers(ARKINDEX_DIR, extensions_texte)
    print(f"  → {len(index_fichiers)} entrées dans l'index")

    # ── 4. Détection de la colonne id dans le CSV ───────────────────────────
    id_candidates = ["id", "doc_id", "identifiant", "uuid", "document_id", "arkindex_id"]
    col_id = None
    for cand in id_candidates:
        matches = [c for c in df.columns if c.lower().strip() == cand]
        if matches:
            col_id = matches[0]
            break

    if col_id is None:
        print("\n[WARN] Colonne 'id' non trouvée. Utilisation de l'index de ligne comme id.")
        df["_doc_id"] = df.index.astype(str)
        col_id = "_doc_id"
    else:
        print(f"\nColonne id détectée : '{col_id}'")

    # ── 5. Chargement du tokenizer ──────────────────────────────────────────
    print(f"\nChargement du tokenizer CamemBERT ({CAMEMBERT_MODEL})…")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_MODEL)
        print("  → Tokenizer chargé.")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le tokenizer : {e}")
        print("Vérifiez votre connexion internet ou installez les dépendances.")
        raise SystemExit(1)

    # ── 6. Traitement document par document ─────────────────────────────────
    resultats = []
    non_trouves = 0
    longueurs_tokens = []

    print(f"\nTraitement de {len(df)} documents…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chargement textes"):
        doc_id = str(row[col_id]).strip()

        # Recherche du fichier texte
        chemin = trouver_fichier_texte(doc_id, ARKINDEX_DIR, index_fichiers)

        if chemin is None:
            non_trouves += 1
            resultats.append({
                "doc_id": doc_id,
                "texte_brut": None,
                "texte_clean": None,
                "longueur_tokens": 0,
                "fichier_source": None,
                "erreur": "fichier_non_trouve",
            })
            continue

        # Lecture et nettoyage
        try:
            texte_brut = lire_fichier_texte(chemin)
            texte_clean = nettoyer_texte(texte_brut)
            texte_tronque, nb_tokens = tronquer_a_512_tokens(texte_clean, tokenizer)
            longueurs_tokens.append(nb_tokens)

            resultats.append({
                "doc_id": doc_id,
                "texte_brut": texte_brut,
                "texte_clean": texte_tronque,
                "longueur_tokens": nb_tokens,
                "fichier_source": str(chemin.relative_to(ARKINDEX_DIR)),
                "erreur": None,
            })

        except Exception as e:
            warnings.warn(f"[WARN] Erreur sur document {doc_id} : {e}")
            resultats.append({
                "doc_id": doc_id,
                "texte_brut": None,
                "texte_clean": None,
                "longueur_tokens": 0,
                "fichier_source": str(chemin.relative_to(ARKINDEX_DIR)),
                "erreur": str(e),
            })

    # ── 7. Statistiques ─────────────────────────────────────────────────────
    docs_ok = [r for r in resultats if r["erreur"] is None]
    print("\n" + "=" * 60)
    print("STATISTIQUES DE CHARGEMENT")
    print("=" * 60)
    print(f"Documents traités    : {len(resultats)}")
    print(f"Documents chargés   : {len(docs_ok)}")
    print(f"Fichiers non trouvés : {non_trouves}")
    print(f"Autres erreurs       : {len(resultats) - len(docs_ok) - non_trouves}")

    if longueurs_tokens:
        import numpy as np
        arr = longueurs_tokens
        print(f"\nLongueur des textes (en tokens) :")
        print(f"  Moyenne : {sum(arr)/len(arr):.1f}")
        print(f"  Médiane : {sorted(arr)[len(arr)//2]}")
        print(f"  Min     : {min(arr)}")
        print(f"  Max     : {max(arr)}")
        print(f"  > 510   : {sum(1 for x in arr if x >= MAX_TOKENS)} documents tronqués")

    # ── 8. Sauvegarde ───────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Textes sauvegardés : {OUT_JSON}")
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Terminée avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
