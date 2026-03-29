"""
Script 06 — Baseline NER avec spaCy fr_core_news_lg
=====================================================
Évalue le modèle spaCy français sans fine-tuning sur notre test set.
Sert de borne inférieure pour comparer avec CamemBERT fine-tuné.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import spacy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

# Mapping spaCy → nos tags
# spaCy fr_core_news_lg utilise : PER, ORG, LOC, MISC
# Nos tags : PER, ORG, LOC, MISC → mapping direct
SPACY_TO_OUR = {
    "PER":  "PER",
    "ORG":  "ORG",
    "LOC":  "LOC",
    "MISC": "MISC",
    "GPE":  "LOC",   # pays/villes détectés comme GPE → LOC
    "NORP": "ORG",   # nationalités/partis → ORG
}

LABEL2ID = {"O":0,"B-PER":1,"I-PER":2,"B-ORG":3,"I-ORG":4,
            "B-LOC":5,"I-LOC":6,"B-MISC":7,"I-MISC":8}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ──────────────────────────────────────────
# PARTIE 2 — Charger spaCy et les données
# ──────────────────────────────────────────

# Charger le modèle spaCy français
print("Chargement du modèle spaCy fr_core_news_lg...")
try:
    nlp = spacy.load("fr_core_news_lg")
    print("Modèle chargé.")
except OSError:
    print("Modèle non trouvé. Installation...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_lg"])
    nlp = spacy.load("fr_core_news_lg")

# Charger les textes bruts
with open(BASE_DIR / "data/processed/documents_clean.json", encoding="utf-8") as f:
    docs_clean = json.load(f)
docs_by_id = {d["id"]: d for d in docs_clean}

# Charger les labels gold (vérité terrain) depuis test.json
with open(BASE_DIR / "data/bio/test.json", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Documents test : {len(test_data)}")

# ──────────────────────────────────────────
# PARTIE 3 — Prédictions spaCy
# ──────────────────────────────────────────

def extraire_entites_gold(tokens_gold, tags_gold):
    """
    Extrait les entités gold sous forme de (type, texte_normalise).
    Reconstruit le texte de chaque entité depuis les tokens CamemBERT
    en retirant le préfixe SentencePiece ▁.
    """
    entites = set()
    i = 0
    while i < len(tags_gold):
        tag = tags_gold[i]
        if tag.startswith("B-"):
            entite_type = tag[2:]
            tokens_entite = [tokens_gold[i].lstrip("▁")]
            i += 1
            while i < len(tags_gold) and tags_gold[i] == f"I-{entite_type}":
                # Les sous-mots sans ▁ sont collés au précédent
                t = tokens_gold[i]
                if t.startswith("▁"):
                    tokens_entite.append(" " + t.lstrip("▁"))
                else:
                    tokens_entite.append(t)
                i += 1
            texte_entite = "".join(tokens_entite).strip().lower()
            if texte_entite:
                entites.add((entite_type, texte_entite))
        else:
            i += 1
    return entites


def extraire_entites_spacy(doc_spacy):
    """
    Extrait les entités spaCy sous forme de (type_mappe, texte_normalise).
    Ignore les types non mappés dans SPACY_TO_OUR.
    """
    entites = set()
    for ent in doc_spacy.ents:
        tag = SPACY_TO_OUR.get(ent.label_, None)
        if tag is None:
            continue
        texte_norm = ent.text.strip().lower()
        if texte_norm:
            entites.add((tag, texte_norm))
    return entites

# ──────────────────────────────────────────
# PARTIE 4 — Évaluation au niveau entité-texte
# ──────────────────────────────────────────
# On compare (type, texte_normalisé) entre gold et spaCy.
# C'est l'évaluation standard pour une baseline hors-domaine :
# elle évite les problèmes d'alignement entre tokenizers différents.

# Accumulateurs par type d'entité
stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
stats_global = {"tp": 0, "fp": 0, "fn": 0}

print("\nApplication de spaCy sur le test set...")
erreurs_spacy = 0

for doc_gold in tqdm(test_data):
    doc_id = doc_gold["id"]
    tokens_gold = doc_gold["tokens"]
    tags_gold = [ID2LABEL[t] for t in doc_gold["ner_tags"]]

    # Récupérer le texte brut
    if doc_id not in docs_by_id:
        erreurs_spacy += 1
        continue

    texte = docs_by_id[doc_id]["texte"]

    # Entités gold : (type, texte_normalisé)
    ents_gold = extraire_entites_gold(tokens_gold, tags_gold)

    # Prédictions spaCy sur le texte brut (limité à 5000 chars)
    doc_spacy = nlp(texte[:5000])
    ents_pred = extraire_entites_spacy(doc_spacy)

    # TP, FP, FN par type
    for (entite, texte_ent) in ents_pred:
        if (entite, texte_ent) in ents_gold:
            stats[entite]["tp"] += 1
            stats_global["tp"] += 1
        else:
            stats[entite]["fp"] += 1
            stats_global["fp"] += 1

    for (entite, texte_ent) in ents_gold:
        if (entite, texte_ent) not in ents_pred:
            stats[entite]["fn"] += 1
            stats_global["fn"] += 1

# Calculer P, R, F1
def compute_prf(tp, fp, fn):
    """Calcule Précision, Rappel et F1 en pourcentage."""
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return round(p*100, 2), round(r*100, 2), round(f1*100, 2)

# ──────────────────────────────────────────
# PARTIE 5 — Affichage des résultats
# ──────────────────────────────────────────

print("\n" + "=" * 65)
print("RESULTATS BASELINE spaCy fr_core_news_lg")
print("=" * 65)
print(f"{'Entite':<10} {'Precision':>12} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8}")
print("-" * 65)

resultats = {}
for entite in ["PER", "ORG", "LOC", "MISC"]:
    s = stats[entite]
    p, r, f1 = compute_prf(s["tp"], s["fp"], s["fn"])
    resultats[entite] = {"precision": p, "recall": r, "f1": f1,
                          "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}
    print(f"{entite:<10} {p:>11}% {r:>9}% {f1:>9}% {s['tp']:>8} {s['fp']:>8} {s['fn']:>8}")

print("-" * 65)
p_g, r_g, f1_g = compute_prf(
    stats_global["tp"], stats_global["fp"], stats_global["fn"]
)
print(f"{'GLOBAL':<10} {p_g:>11}% {r_g:>9}% {f1_g:>9}%")
print("=" * 65)

print(f"\nDocuments traites : {len(test_data) - erreurs_spacy}/{len(test_data)}")
print(f"Modele utilise    : fr_core_news_lg (sans fine-tuning)")
print(f"Note              : Ce score est la BORNE INFERIEURE (baseline)")
print(f"                    CamemBERT fine-tune devrait depasser ce score")

# Sauvegarder les résultats
resultats_finaux = {
    "modele": "spacy_fr_core_news_lg",
    "type": "baseline_sans_finetuning",
    "global": {"precision": p_g, "recall": r_g, "f1": f1_g},
    "par_entite": resultats
}

out_path = BASE_DIR / "data/processed/baseline_spacy_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(resultats_finaux, f, ensure_ascii=False, indent=2)

print(f"\nResultats sauvegardes -> {out_path}")
print("\nProchaine etape : fine-tuning CamemBERT sur Google Colab")
