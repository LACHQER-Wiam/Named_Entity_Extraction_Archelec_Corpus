"""
Script 05 — Vérification qualité des données BIO
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

LABEL2ID = {
    "O":0,"B-PER":1,"I-PER":2,"B-ORG":3,"I-ORG":4,
    "B-LOC":5,"I-LOC":6,"B-MISC":7,"I-MISC":8
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

erreurs_totales = 0
rapport = []

def log(msg):
    print(msg)
    rapport.append(msg)

log("=" * 60)
log("RAPPORT DE QUALITE -- DONNEES NER ARCHELEC")
log("=" * 60)

# ─── CHARGER LES 3 SPLITS ───────────────────────────────
splits = {}
for nom in ["train", "val", "test"]:
    with open(BASE_DIR / f"data/bio/{nom}.json", encoding='utf-8') as f:
        splits[nom] = json.load(f)
    log(f"\n{nom.upper()} : {len(splits[nom])} documents charges")

# ─── VÉRIFICATION 1 : pas de I- sans B- précédent ───────
log("\n--- VERIFICATION 1 : coherence BIO ---")
for nom, data in splits.items():
    erreurs_bio = 0
    for doc in data:
        tags = doc['ner_tags']
        for i, tag_id in enumerate(tags):
            label = ID2LABEL[tag_id]
            if label.startswith("I-"):
                if i == 0:
                    erreurs_bio += 1
                else:
                    prev = ID2LABEL[tags[i-1]]
                    entite = label[2:]
                    if not (prev == f"B-{entite}" or prev == f"I-{entite}"):
                        erreurs_bio += 1
    if erreurs_bio == 0:
        log(f"  OK {nom} : aucune erreur BIO")
    else:
        log(f"  ERREUR {nom} : {erreurs_bio} erreurs BIO")
        erreurs_totales += erreurs_bio

# ─── VÉRIFICATION 2 : longueur séquences ────────────────
log("\n--- VERIFICATION 2 : longueur sequences (max 512) ---")
for nom, data in splits.items():
    trop_longs = [d for d in data if len(d['tokens']) > 512]
    longueurs = [len(d['tokens']) for d in data]
    moy = sum(longueurs) // len(longueurs)
    log(f"  {nom} -- moy: {moy} tokens | max: {max(longueurs)} | >512: {len(trop_longs)}")
    if trop_longs:
        erreurs_totales += len(trop_longs)
        log(f"  ERREUR {len(trop_longs)} sequences trop longues")
    else:
        log(f"  OK toutes les sequences sont dans la limite")

# ─── VÉRIFICATION 3 : pas de valeurs nulles ─────────────
log("\n--- VERIFICATION 3 : valeurs nulles ---")
for nom, data in splits.items():
    nulls = 0
    for doc in data:
        if not doc.get('tokens') or not doc.get('ner_tags'):
            nulls += 1
        if len(doc['tokens']) != len(doc['ner_tags']):
            nulls += 1
    if nulls == 0:
        log(f"  OK {nom} : aucune valeur nulle, tokens/tags alignes")
    else:
        log(f"  ERREUR {nom} : {nulls} problemes detectes")
        erreurs_totales += nulls

# ─── VÉRIFICATION 4 : distribution des labels ────────────
log("\n--- VERIFICATION 4 : distribution labels (train) ---")
all_tags = [ID2LABEL[t] for d in splits['train'] for t in d['ner_tags']]
counts = Counter(all_tags)
total = sum(counts.values())
for label in ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]:
    n = counts.get(label, 0)
    pct = 100 * n / total
    log(f"  {label:<10} : {n:>8} ({pct:.2f}%)")

# ─── VÉRIFICATION 5 : pas de data leakage ───────────────
log("\n--- VERIFICATION 5 : data leakage ---")
ids_train = set(d['id'] for d in splits['train'])
ids_val   = set(d['id'] for d in splits['val'])
ids_test  = set(d['id'] for d in splits['test'])

overlap_tv = ids_train & ids_val
overlap_tt = ids_train & ids_test
overlap_vt = ids_val   & ids_test

if not overlap_tv and not overlap_tt and not overlap_vt:
    log("  OK Aucun data leakage entre train/val/test")
else:
    log(f"  ERREUR Leakage train/val : {len(overlap_tv)}")
    log(f"  ERREUR Leakage train/test: {len(overlap_tt)}")
    log(f"  ERREUR Leakage val/test  : {len(overlap_vt)}")
    erreurs_totales += len(overlap_tv) + len(overlap_tt) + len(overlap_vt)

# ─── AFFICHER 5 EXEMPLES ANNOTÉS LISIBLES ───────────────
log("\n--- 5 EXEMPLES ANNOTES (train) ---")
random.seed(42)
exemples = random.sample(splits['train'], 5)

for i, doc in enumerate(exemples):
    log(f"\n  Exemple {i+1} -- ID: {doc['id']} | Annee: {doc['annee']}")
    entites_trouvees = []
    j = 0
    while j < len(doc['ner_tags']):
        label = ID2LABEL[doc['ner_tags'][j]]
        if label.startswith("B-"):
            tag_type = label[2:]
            tokens_entite = [doc['tokens'][j]]
            j += 1
            while j < len(doc['ner_tags']) and ID2LABEL[doc['ner_tags'][j]] == f"I-{tag_type}":
                tokens_entite.append(doc['tokens'][j])
                j += 1
            entite_str = ''.join(tokens_entite).replace('▁', ' ').strip()
            entites_trouvees.append(f"[{tag_type}] {entite_str}")
        else:
            j += 1

    if entites_trouvees:
        for e in entites_trouvees[:8]:
            log(f"    {e}")
    else:
        log("    (aucune entite trouvee dans cet exemple)")

# ─── RÉSUMÉ FINAL ───────────────────────────────────────
log("\n" + "=" * 60)
log("RESUME FINAL")
log("=" * 60)
log(f"Total documents    : {sum(len(v) for v in splits.values())}")
log(f"Train/Val/Test     : {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
log(f"Erreurs detectees  : {erreurs_totales}")

if erreurs_totales == 0:
    log("\nDONNEES PRETES POUR L'ENTRAINEMENT CAMEMBERT !")
    log("   Fichiers dans data/bio/ :")
    log("   - train.json  (entrainement)")
    log("   - val.json    (validation)")
    log("   - test.json   (evaluation finale)")
    log("   - label2id.json")
else:
    log(f"\n{erreurs_totales} erreurs a corriger avant l'entrainement")

# ─── SAUVEGARDER LE RAPPORT ─────────────────────────────
(BASE_DIR / "data/processed/quality_report.txt").write_text(
    '\n'.join(rapport), encoding='utf-8'
)
print("\nRapport sauvegarde -> data/processed/quality_report.txt")
