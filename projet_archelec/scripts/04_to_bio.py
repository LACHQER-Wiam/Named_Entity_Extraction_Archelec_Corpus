"""
Script 04 — Conversion au format BIO / CoNLL
=============================================
Tokenise les textes annotés avec CamemBERT, aligne les entités
sur les tokens (gestion du subword tokenization), produit les formats
CoNLL et HuggingFace Dataset, et effectue un split train/val/test.
"""

import json
import random
import collections
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
ANN_DIR   = BASE_DIR / "data" / "annotated"
BIO_DIR   = BASE_DIR / "data" / "bio"
ANN_JSON  = ANN_DIR  / "archelec_annotated.json"

BIO_DIR.mkdir(parents=True, exist_ok=True)

# ─── Paramètres ───────────────────────────────────────────────────────────────
CAMEMBERT_MODEL = "camembert-base"
MAX_LENGTH      = 512     # limite dure CamemBERT (tokens incluant [CLS] et [SEP])
SPLIT_RATIOS    = (0.80, 0.10, 0.10)   # train / val / test
RANDOM_SEED     = 42

# ─── Mapping labels → indices ─────────────────────────────────────────────────
LABEL2ID: dict[str, int] = {
    "O":      0,
    "B-PER":  1,
    "I-PER":  2,
    "B-ORG":  3,
    "I-ORG":  4,
    "B-LOC":  5,
    "I-LOC":  6,
    "B-MISC": 7,
    "I-MISC": 8,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


# ─── Conversion annotation → séquence BIO ────────────────────────────────────

def construire_carte_caracteres(entites: list[dict], longueur_texte: int) -> list[str]:
    """
    Construit un tableau de tags au niveau caractère.

    Chaque position i dans le tableau correspond au tag du caractère i du texte.
    Utilisé pour aligner les entités avec les tokens CamemBERT.
    """
    carte = ["O"] * longueur_texte

    # Trier les entités par début, puis par longueur décroissante
    entites_triees = sorted(entites, key=lambda e: (e["debut"], -(e["fin"] - e["debut"])))

    for ent in entites_triees:
        debut = ent["debut"]
        fin   = ent["fin"]
        tag   = ent["tag"]

        # Vérification des bornes
        if debut < 0 or fin > longueur_texte or debut >= fin:
            continue

        # Marquer le premier caractère en B-, les suivants en I-
        carte[debut] = f"B-{tag}"
        for i in range(debut + 1, fin):
            carte[i] = f"I-{tag}"

    return carte


def aligner_tokens_avec_entites(
    texte: str,
    entites: list[dict],
    tokenizer,
) -> tuple[list[str], list[str]]:
    """
    Tokenise un texte avec CamemBERT et aligne les entités sur les tokens.

    Stratégie :
    - On encode le texte avec return_offsets_mapping=True pour obtenir
      la position de chaque token dans le texte original.
    - Pour chaque token, on regarde le tag du premier caractère non-espace
      de son span dans la carte caractère.
    - Premier sous-mot d'une entité → B-TAG
    - Sous-mots suivants           → I-TAG
    - Tokens spéciaux ([CLS], [SEP]) → ignorés (label -100 dans HuggingFace)

    Retourne (tokens_texte, bio_tags) sans les tokens spéciaux.
    """
    # Encodage avec les offsets
    encoding = tokenizer(
        texte,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    token_ids      = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    # Carte des tags au niveau caractère
    carte = construire_carte_caracteres(entites, len(texte))

    tokens_texte: list[str]  = []
    bio_tags:     list[str]  = []

    for token_id, (debut_char, fin_char) in zip(token_ids, offset_mapping):
        # Tokens spéciaux (offset (0,0) dans CamemBERT/RoBERTa)
        if debut_char == fin_char:
            continue

        # Texte du token dans le texte original
        token_texte = texte[debut_char:fin_char]

        # Détermination du tag :
        # On prend le tag du premier caractère non-espace du span
        tag = "O"
        for i in range(debut_char, fin_char):
            if i < len(carte) and texte[i] != " ":
                tag = carte[i]
                break

        tokens_texte.append(token_texte)
        bio_tags.append(tag)

    return tokens_texte, bio_tags


def corriger_tags_bio(tags: list[str]) -> list[str]:
    """
    Corrige les séquences BIO invalides :
    - I-TAG en début de séquence ou après O → remplacé par B-TAG
    - I-TAG après un B-/I- d'un tag différent → remplacé par B-TAG
    """
    tags_corriges = []
    tag_precedent = "O"

    for tag in tags:
        if tag.startswith("I-"):
            entite = tag[2:]
            # Vérifier que le tag précédent est cohérent
            if tag_precedent in (f"B-{entite}", f"I-{entite}"):
                tags_corriges.append(tag)
            else:
                # Transformer I- en B- pour corriger
                tags_corriges.append(f"B-{entite}")
        else:
            tags_corriges.append(tag)

        tag_precedent = tags_corriges[-1]

    return tags_corriges


# ─── Écriture des formats de sortie ───────────────────────────────────────────

def ecrire_conll(exemples: list[dict], chemin: Path) -> None:
    """
    Écrit les données au format CoNLL :
    token<TAB>tag
    [ligne vide entre documents]
    """
    with open(chemin, "w", encoding="utf-8") as f:
        for exemple in exemples:
            for token, tag in zip(exemple["tokens"], exemple["ner_tags_str"]):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")   # séparateur de document


def ecrire_hf_json(exemples: list[dict], chemin: Path) -> None:
    """
    Écrit les données au format HuggingFace Dataset (JSON lines).
    Chaque ligne = un document avec 'tokens' et 'ner_tags' (indices).
    """
    with open(chemin, "w", encoding="utf-8") as f:
        for exemple in exemples:
            ligne = {
                "tokens":   exemple["tokens"],
                "ner_tags": exemple["ner_tags"],
            }
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")


# ─── Split stratifié ──────────────────────────────────────────────────────────

def split_stratifie(exemples: list[dict],
                    ratios: tuple[float, float, float] = SPLIT_RATIOS,
                    seed: int = RANDOM_SEED) -> tuple[list, list, list]:
    """
    Divise les exemples en train/val/test en stratifiant par année.

    Garantit que la proportion 1973/1978 est respectée dans chaque split.
    """
    rng = random.Random(seed)

    # Regrouper par année
    par_annee: dict[Optional[int], list] = collections.defaultdict(list)
    for ex in exemples:
        par_annee[ex.get("annee")].append(ex)

    train_all, val_all, test_all = [], [], []

    for annee, docs in par_annee.items():
        docs_copy = docs.copy()
        rng.shuffle(docs_copy)
        n = len(docs_copy)

        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        # Le reste va en test
        n_test  = n - n_train - n_val

        train_all.extend(docs_copy[:n_train])
        val_all.extend(docs_copy[n_train:n_train + n_val])
        test_all.extend(docs_copy[n_train + n_val:])

    # Mélanger chaque split
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    return train_all, val_all, test_all


# ─── Statistiques ─────────────────────────────────────────────────────────────

def afficher_stats_split(nom: str, exemples: list[dict]) -> None:
    """Affiche les statistiques d'un split (tokens, tags)."""
    if not exemples:
        print(f"  {nom}: 0 documents")
        return

    compteur_tags: dict[str, int] = collections.Counter()
    nb_tokens_total = 0

    for ex in exemples:
        nb_tokens_total += len(ex["tokens"])
        for tag in ex["ner_tags_str"]:
            compteur_tags[tag] += 1

    nb_docs = len(exemples)
    print(f"\n  {nom.upper()} — {nb_docs} docs | {nb_tokens_total} tokens")

    for tag, count in sorted(compteur_tags.items()):
        if tag != "O":
            print(f"    {tag:10s}: {count}")
    print(f"    {'O':10s}: {compteur_tags.get('O', 0)}")


def verifier_leakage(train: list, val: list, test: list) -> None:
    """
    Vérifie qu'il n'y a pas de data leakage entre les splits
    (même doc_id dans deux splits différents).
    """
    ids_train = {ex.get("doc_id") for ex in train}
    ids_val   = {ex.get("doc_id") for ex in val}
    ids_test  = {ex.get("doc_id") for ex in test}

    leakage_tv = ids_train & ids_val
    leakage_tt = ids_train & ids_test
    leakage_vt = ids_val   & ids_test

    if leakage_tv or leakage_tt or leakage_vt:
        print(f"  [WARN] Data leakage détecté !")
        if leakage_tv: print(f"    Train ∩ Val  : {len(leakage_tv)} docs")
        if leakage_tt: print(f"    Train ∩ Test : {len(leakage_tt)} docs")
        if leakage_vt: print(f"    Val ∩ Test   : {len(leakage_vt)} docs")
    else:
        print("  ✓ Aucun data leakage entre les splits")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script de conversion BIO."""
    print("=" * 60)
    print("ÉTAPE 4 — Conversion au format BIO / CoNLL")
    print("=" * 60)

    # ── 1. Chargement ────────────────────────────────────────────────────────
    if not ANN_JSON.exists():
        print(f"[ERREUR] Fichier annoté introuvable : {ANN_JSON}")
        raise SystemExit(1)

    with open(ANN_JSON, encoding="utf-8") as f:
        documents = json.load(f)
    print(f"\n{len(documents)} documents annotés chargés")

    # ── 2. Tokenizer ─────────────────────────────────────────────────────────
    print(f"\nChargement du tokenizer CamemBERT ({CAMEMBERT_MODEL})…")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_MODEL)
    except Exception as e:
        print(f"[ERREUR] {e}")
        raise SystemExit(1)
    print("  → Tokenizer chargé")

    # ── 3. Conversion BIO ────────────────────────────────────────────────────
    exemples_bruts: list[dict] = []
    erreurs = 0

    print("\nConversion en tokens BIO…")
    for doc in tqdm(documents, desc="Tokenisation"):
        texte   = doc.get("texte", "")
        entites = doc.get("entites", [])

        if not texte:
            continue

        try:
            tokens, tags = aligner_tokens_avec_entites(texte, entites, tokenizer)
            tags_corriges = corriger_tags_bio(tags)
            ner_ids = [LABEL2ID.get(t, 0) for t in tags_corriges]

            if len(tokens) == 0:
                continue

            exemples_bruts.append({
                "doc_id":      doc.get("doc_id", ""),
                "annee":       doc.get("annee"),
                "tokens":      tokens,
                "ner_tags_str": tags_corriges,
                "ner_tags":    ner_ids,
            })
        except Exception as e:
            erreurs += 1
            print(f"  [WARN] Erreur doc {doc.get('doc_id', '?')} : {e}")

    print(f"  → {len(exemples_bruts)} exemples convertis ({erreurs} erreurs)")

    # ── 4. Split train/val/test ───────────────────────────────────────────────
    print("\nSplit stratifié par année (80/10/10)…")
    train, val, test = split_stratifie(exemples_bruts)
    print(f"  Train : {len(train)} | Val : {len(val)} | Test : {len(test)}")

    # ── 5. Vérification du leakage ────────────────────────────────────────────
    print("\nVérification data leakage :")
    verifier_leakage(train, val, test)

    # ── 6. Écriture des fichiers ──────────────────────────────────────────────
    print("\nÉcriture des fichiers…")
    splits = [("train", train), ("val", val), ("test", test)]
    for nom, data in splits:
        # CoNLL
        ecrire_conll(data, BIO_DIR / f"{nom}.conll")
        # HuggingFace JSON
        ecrire_hf_json(data, BIO_DIR / f"{nom}.json")
        print(f"  → {nom}.conll + {nom}.json sauvegardés")

    # label2id
    label2id_path = BIO_DIR / "label2id.json"
    with open(label2id_path, "w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)
    print(f"  → label2id.json sauvegardé")

    # ── 7. Statistiques finales ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STATISTIQUES FINALES")
    print("=" * 60)
    for nom, data in splits:
        afficher_stats_split(nom, data)

    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Terminée avec succès")
    print("=" * 60)
    print(f"\nFichiers générés dans : {BIO_DIR}")
    for f in sorted(BIO_DIR.iterdir()):
        taille_kb = f.stat().st_size // 1024
        print(f"  {f.name:20s} ({taille_kb} Ko)")


if __name__ == "__main__":
    main()
