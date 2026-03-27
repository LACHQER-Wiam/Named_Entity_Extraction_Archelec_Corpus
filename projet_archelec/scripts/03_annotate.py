"""
Script 03 — Annotation automatique par distant supervision
==========================================================
Cherche les entités du CSV (noms, partis, communes, etc.)
dans les textes OCRisés et produit des annotations au format JSON.
"""

import re
import json
import logging
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# fuzzywuzzy avec Levenshtein rapide si disponible
try:
    from rapidfuzz import fuzz as fuzzy_fuzz
    FUZZY_LIB = "rapidfuzz"
except ImportError:
    try:
        from fuzzywuzzy import fuzz as fuzzy_fuzz
        FUZZY_LIB = "fuzzywuzzy"
    except ImportError:
        fuzzy_fuzz = None
        FUZZY_LIB = None

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
PROC_DIR     = BASE_DIR / "data" / "processed"
ANN_DIR      = BASE_DIR / "data" / "annotated"
CSV_PATH     = PROC_DIR / "archelec_1973_1978.csv"
TEXTS_PATH   = PROC_DIR / "texts_clean.json"
OUT_JSON     = ANN_DIR  / "archelec_annotated.json"
LOG_PATH     = PROC_DIR / "not_found.log"

ANN_DIR.mkdir(parents=True, exist_ok=True)

# Seuil de similarité pour la recherche fuzzy (en %)
FUZZY_THRESHOLD = 85

# ─── Abréviations connues des partis français ─────────────────────────────────
PARTI_ABBREVIATIONS: dict[str, list[str]] = {
    "parti socialiste":                   ["PS", "P.S."],
    "parti communiste français":           ["PCF", "PC", "P.C.F."],
    "rassemblement pour la république":    ["RPR", "R.P.R."],
    "union pour la démocratie française":  ["UDF", "U.D.F."],
    "union des démocrates pour la république": ["UDR", "U.D.R."],
    "mouvement républicain populaire":     ["MRP", "M.R.P."],
    "fédération de la gauche démocrate et socialiste": ["FGDS", "F.G.D.S."],
    "front national":                     ["FN", "F.N."],
    "les républicains indépendants":       ["RI", "R.I."],
    "centre démocrate et progrès":         ["CDP"],
    "réformateurs":                        ["MRG"],
    "mouvement radical de gauche":         ["MRG"],
    "gaulliste":                           ["UNR", "UDT"],
    "union nationale inter-universitaire": ["UNI"],
    "lutte ouvrière":                     ["LO", "L.O."],
    "ligue communiste révolutionnaire":    ["LCR", "L.C.R."],
}


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class Entite:
    """Représente une entité annotée dans un texte."""
    texte: str
    tag: str
    debut: int
    fin: int

    def chevauche(self, autre: "Entite") -> bool:
        """Vérifie si deux entités se chevauchent."""
        return not (self.fin <= autre.debut or autre.fin <= self.debut)


@dataclass
class DocumentAnnote:
    """Document avec ses entités annotées."""
    doc_id: str
    texte: str
    annee: Optional[int]
    entites: list[Entite] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "texte": self.texte,
            "annee": self.annee,
            "entites": [
                {"texte": e.texte, "tag": e.tag, "debut": e.debut, "fin": e.fin}
                for e in sorted(self.entites, key=lambda x: x.debut)
            ],
        }


# ─── Normalisation ────────────────────────────────────────────────────────────

def normaliser(texte: str) -> str:
    """
    Normalise un texte pour la comparaison (minuscules, sans accents).
    Utilisé uniquement pour le matching, pas pour l'annotation finale.
    """
    if not isinstance(texte, str):
        return ""
    texte = texte.lower().strip()
    # Suppression des accents
    nfkd = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in nfkd if not unicodedata.combining(c))
    return texte


# ─── Recherche d'entités ──────────────────────────────────────────────────────

def chercher_exact(texte: str, motif: str, tag: str,
                   word_boundary: bool = True) -> list[Entite]:
    """
    Cherche toutes les occurrences exactes (case-insensitive) d'un motif.

    word_boundary : si True, exige des frontières de mot (évite les faux positifs).
    Retourne la liste des entités trouvées avec leur position dans le texte original.
    """
    if not motif or not texte or len(motif.strip()) < 2:
        return []

    motif_escaped = re.escape(motif.strip())
    if word_boundary:
        pattern = r"(?<!\w)" + motif_escaped + r"(?!\w)"
    else:
        pattern = motif_escaped

    entites = []
    for match in re.finditer(pattern, texte, flags=re.IGNORECASE):
        entites.append(Entite(
            texte=texte[match.start():match.end()],
            tag=tag,
            debut=match.start(),
            fin=match.end(),
        ))
    return entites


def chercher_fuzzy(texte: str, motif: str, tag: str,
                   seuil: int = FUZZY_THRESHOLD) -> list[Entite]:
    """
    Cherche un motif par correspondance approximative dans le texte.

    Fenêtre glissante sur des segments de longueur comparable au motif.
    Utilisé en fallback quand la recherche exacte échoue.
    """
    if fuzzy_fuzz is None or not motif or len(motif) < 4:
        return []

    motif_norm = normaliser(motif)
    longueur_motif = len(motif)
    # Fenêtre légèrement plus large pour absorber les erreurs OCR
    fenetre_min = max(4, longueur_motif - 5)
    fenetre_max = longueur_motif + 5

    texte_norm = normaliser(texte)
    entites = []

    for taille in range(fenetre_min, fenetre_max + 1):
        for i in range(len(texte_norm) - taille + 1):
            segment_norm = texte_norm[i:i + taille]
            score = fuzzy_fuzz.ratio(motif_norm, segment_norm)
            if score >= seuil:
                segment_original = texte[i:i + taille]
                entites.append(Entite(
                    texte=segment_original,
                    tag=tag,
                    debut=i,
                    fin=i + taille,
                ))

    # Dédupliquer : garder la meilleure occurrence par position
    entites_dedup = []
    positions_vues = set()
    for e in sorted(entites, key=lambda x: x.debut):
        if e.debut not in positions_vues:
            entites_dedup.append(e)
            positions_vues.add(e.debut)

    return entites_dedup


def chercher_departement(texte: str, nom_dept: str, num_dept: str) -> list[Entite]:
    """
    Cherche un département par son nom ET son numéro (ex: Rhône, 69).
    """
    entites = []
    if nom_dept:
        entites.extend(chercher_exact(texte, nom_dept, "LOC"))
    if num_dept:
        # Numéro seul en frontière de mot
        entites.extend(chercher_exact(texte, str(num_dept).zfill(2), "LOC"))
    return entites


def chercher_parti_avec_abreviations(texte: str, nom_parti: str) -> list[Entite]:
    """
    Cherche un parti par son nom complet ET ses abréviations connues.
    """
    entites = chercher_exact(texte, nom_parti, "ORG")
    nom_norm = normaliser(nom_parti)

    for parti_ref, abreviations in PARTI_ABBREVIATIONS.items():
        if parti_ref in nom_norm or nom_norm in parti_ref:
            for abrev in abreviations:
                entites.extend(chercher_exact(texte, abrev, "ORG"))

    return entites


# ─── Résolution des chevauchements ────────────────────────────────────────────

def resoudre_chevauchements(entites: list[Entite]) -> list[Entite]:
    """
    Résout les chevauchements entre entités selon les règles de priorité :
    1. Entité plus longue > entité plus courte
    2. Première occurrence si même longueur

    Trie d'abord par longueur décroissante pour que les entités longues
    prennent priorité sur les courtes.
    """
    # Trier : longueur décroissante, puis position croissante
    entites_triees = sorted(entites, key=lambda e: (-(e.fin - e.debut), e.debut))
    retenues: list[Entite] = []

    for candidat in entites_triees:
        conflit = any(candidat.chevauche(r) for r in retenues)
        if not conflit:
            retenues.append(candidat)

    return sorted(retenues, key=lambda e: e.debut)


# ─── Annotation d'un document ─────────────────────────────────────────────────

def annoter_document(doc_id: str, texte: str, annee: Optional[int],
                     nom: str, prenom: str, parti: str,
                     commune: str, departement: str, numero_dept: str,
                     profession: str,
                     logger: logging.Logger) -> DocumentAnnote:
    """
    Annote un document en cherchant toutes les entités dans le texte.

    Retourne un DocumentAnnote avec la liste des entités trouvées.
    """
    doc = DocumentAnnote(doc_id=doc_id, texte=texte, annee=annee)
    entites_brutes: list[Entite] = []

    # ── Nom complet (PER) ────────────────────────────────────────────────────
    nom_complet = f"{prenom} {nom}".strip() if prenom and nom else nom or prenom
    if nom_complet:
        trouvees = chercher_exact(texte, nom_complet, "PER")
        # Nom seul
        if nom:
            trouvees += chercher_exact(texte, nom, "PER")
        # Prénom seul (seulement si prénom > 3 caractères pour éviter faux positifs)
        if prenom and len(prenom) > 3:
            trouvees += chercher_exact(texte, prenom, "PER")

        if not trouvees and nom_complet:
            # Fallback fuzzy
            trouvees = chercher_fuzzy(texte, nom_complet, "PER")
            if trouvees:
                logger.info(f"[FUZZY-PER] {doc_id}: '{nom_complet}' → trouvé approx.")
            else:
                logger.warning(f"[NOT_FOUND-PER] {doc_id}: '{nom_complet}'")

        entites_brutes.extend(trouvees)

    # ── Parti (ORG) ───────────────────────────────────────────────────────────
    if parti:
        trouvees = chercher_parti_avec_abreviations(texte, parti)
        if not trouvees:
            trouvees = chercher_fuzzy(texte, parti, "ORG")
            if trouvees:
                logger.info(f"[FUZZY-ORG] {doc_id}: '{parti}' → trouvé approx.")
            else:
                logger.warning(f"[NOT_FOUND-ORG] {doc_id}: '{parti}'")
        entites_brutes.extend(trouvees)

    # ── Commune (LOC) ─────────────────────────────────────────────────────────
    if commune:
        trouvees = chercher_exact(texte, commune, "LOC")
        if not trouvees:
            trouvees = chercher_fuzzy(texte, commune, "LOC")
            if trouvees:
                logger.info(f"[FUZZY-LOC-commune] {doc_id}: '{commune}' → approx.")
            else:
                logger.warning(f"[NOT_FOUND-LOC-commune] {doc_id}: '{commune}'")
        entites_brutes.extend(trouvees)

    # ── Département (LOC) ─────────────────────────────────────────────────────
    if departement or numero_dept:
        trouvees = chercher_departement(texte, departement, numero_dept)
        if not trouvees and departement:
            logger.warning(f"[NOT_FOUND-LOC-dept] {doc_id}: '{departement}'")
        entites_brutes.extend(trouvees)

    # ── Profession (MISC) ─────────────────────────────────────────────────────
    if profession and len(profession.strip()) > 3:
        trouvees = chercher_exact(texte, profession, "MISC")
        if not trouvees:
            trouvees = chercher_fuzzy(texte, profession, "MISC")
            if trouvees:
                logger.info(f"[FUZZY-MISC] {doc_id}: '{profession}' → approx.")
            else:
                logger.warning(f"[NOT_FOUND-MISC] {doc_id}: '{profession}'")
        entites_brutes.extend(trouvees)

    # ── Résolution des chevauchements ─────────────────────────────────────────
    doc.entites = resoudre_chevauchements(entites_brutes)
    return doc


# ─── Extraction des valeurs CSV ───────────────────────────────────────────────

def extraire_valeur(row: pd.Series, colonnes_candidates: list[str]) -> str:
    """
    Extrait la valeur d'une ligne CSV en testant plusieurs noms de colonnes.
    Retourne une chaîne vide si aucune colonne n'est trouvée.
    """
    for col in colonnes_candidates:
        if col in row.index and pd.notna(row[col]):
            return str(row[col]).strip()
    return ""


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script d'annotation."""
    print("=" * 60)
    print("ÉTAPE 3 — Annotation par distant supervision")
    print("=" * 60)

    if fuzzy_fuzz is None:
        print("[WARN] Aucune librairie fuzzy disponible. "
              "Installez rapidfuzz ou fuzzywuzzy.")
    else:
        print(f"[INFO] Librairie fuzzy : {FUZZY_LIB}")

    # ── 1. Chargement des données ────────────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"[ERREUR] CSV introuvable : {CSV_PATH}")
        raise SystemExit(1)
    if not TEXTS_PATH.exists():
        print(f"[ERREUR] Textes introuvables : {TEXTS_PATH}")
        raise SystemExit(1)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    with open(TEXTS_PATH, encoding="utf-8") as f:
        textes_data = json.load(f)

    # Index des textes par doc_id
    index_textes = {d["doc_id"]: d for d in textes_data if d.get("texte_clean")}
    print(f"\nCSV : {len(df)} documents")
    print(f"Textes chargés : {len(index_textes)} documents avec texte valide")

    # ── 2. Configuration du logger ───────────────────────────────────────────
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
        filemode="w",
        encoding="utf-8",
    )
    logger = logging.getLogger("annotation")
    # Aussi logger les INFO pour les fuzzy match
    logger.setLevel(logging.INFO)

    # ── 3. Colonnes CSV → mapping ─────────────────────────────────────────────
    COL_NOM        = ["nom", "name", "nom_candidat", "candidate", "lastname", "surname"]
    COL_PRENOM     = ["prenom", "prénom", "firstname", "first_name"]
    COL_PARTI      = ["parti", "party", "liste", "organisation", "parti_politique"]
    COL_COMMUNE    = ["commune", "ville", "city", "localite"]
    COL_DEPT       = ["departement", "département", "dept", "dep", "dpt"]
    COL_NUMDEPT    = ["num_dept", "numero_dept", "code_dept", "dpt_num"]
    COL_PROFESSION = ["profession", "metier", "métier", "job", "occupation", "csp"]
    COL_ANNEE      = ["annee", "année", "year", "election_year"]
    COL_ID         = ["id", "doc_id", "identifiant", "uuid", "document_id", "arkindex_id"]

    # ── 4. Compteurs pour les statistiques ────────────────────────────────────
    stats = {
        "total": 0,
        "avec_texte": 0,
        "nom_trouve": 0,
        "parti_trouve": 0,
        "commune_trouvee": 0,
        "profession_trouvee": 0,
        "nb_entites": [],
    }

    documents_annotes: list[dict] = []

    # ── 5. Annotation document par document ──────────────────────────────────
    print("\nAnnotation en cours…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Annotation"):
        stats["total"] += 1

        doc_id = extraire_valeur(row, COL_ID) or str(row.name)
        texte_entry = index_textes.get(doc_id)
        if texte_entry is None:
            continue

        texte = texte_entry["texte_clean"]
        if not texte:
            continue

        stats["avec_texte"] += 1
        annee_str = extraire_valeur(row, COL_ANNEE)
        try:
            annee = int(float(annee_str)) if annee_str else None
        except ValueError:
            annee = None

        nom        = extraire_valeur(row, COL_NOM)
        prenom     = extraire_valeur(row, COL_PRENOM)
        parti      = extraire_valeur(row, COL_PARTI)
        commune    = extraire_valeur(row, COL_COMMUNE)
        departement = extraire_valeur(row, COL_DEPT)
        num_dept   = extraire_valeur(row, COL_NUMDEPT)
        profession = extraire_valeur(row, COL_PROFESSION)

        doc = annoter_document(
            doc_id=doc_id,
            texte=texte,
            annee=annee,
            nom=nom,
            prenom=prenom,
            parti=parti,
            commune=commune,
            departement=departement,
            numero_dept=num_dept,
            profession=profession,
            logger=logger,
        )

        # Mise à jour des statistiques
        tags_trouves = {e.tag for e in doc.entites}
        if "PER" in tags_trouves:
            stats["nom_trouve"] += 1
        if "ORG" in tags_trouves:
            stats["parti_trouve"] += 1
        if "LOC" in tags_trouves:
            stats["commune_trouvee"] += 1
        if "MISC" in tags_trouves:
            stats["profession_trouvee"] += 1
        stats["nb_entites"].append(len(doc.entites))

        documents_annotes.append(doc.to_dict())

    # ── 6. Statistiques d'annotation ─────────────────────────────────────────
    n = stats["avec_texte"]
    print("\n" + "=" * 60)
    print("STATISTIQUES D'ANNOTATION")
    print("=" * 60)
    print(f"Documents totaux        : {stats['total']}")
    print(f"Documents avec texte    : {n}")
    print(f"Nom (PER) trouvé        : {stats['nom_trouve']} "
          f"({100*stats['nom_trouve']/n:.1f}%)" if n else "N/A")
    print(f"Parti (ORG) trouvé      : {stats['parti_trouve']} "
          f"({100*stats['parti_trouve']/n:.1f}%)" if n else "N/A")
    print(f"Commune (LOC) trouvée   : {stats['commune_trouvee']} "
          f"({100*stats['commune_trouvee']/n:.1f}%)" if n else "N/A")
    print(f"Profession (MISC) trouvée: {stats['profession_trouvee']} "
          f"({100*stats['profession_trouvee']/n:.1f}%)" if n else "N/A")
    if stats["nb_entites"]:
        moy = sum(stats["nb_entites"]) / len(stats["nb_entites"])
        print(f"Entités/document (moy)  : {moy:.1f}")
    print(f"\nLog des non-trouvés     : {LOG_PATH}")

    # ── 7. Sauvegarde ────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(documents_annotes, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Données annotées sauvegardées : {OUT_JSON}")
    print(f"  → {len(documents_annotes)} documents annotés")

    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Terminée avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
