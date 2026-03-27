"""
Script 04 — Conversion au format BIO / CoNLL
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ─── CONFIG ─────────────────────────────────────────────
LABEL2ID = {
    "O":      0,
    "B-PER":  1, "I-PER":  2,
    "B-ORG":  3, "I-ORG":  4,
    "B-LOC":  5, "I-LOC":  6,
    "B-MISC": 7, "I-MISC": 8
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 512
MODEL_NAME = "camembert-base"

# ─── CHARGER TOKENIZER ──────────────────────────────────
print("Chargement du tokenizer CamemBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─── CHARGER DONNÉES ────────────────────────────────────
with open(BASE_DIR / "data/annotated/archelec_annotated.json", encoding='utf-8') as f:
    documents = json.load(f)
print(f"Documents chargés : {len(documents)}")

# ─── FONCTION PRINCIPALE ────────────────────────────────
def convertir_en_bio(doc):
    """
    Convertit un document annoté en format BIO aligné avec CamemBERT.
    Retourne None si le document est trop long ou vide.
    """
    texte   = doc['texte']
    entites = doc['entites']

    # Tokeniser avec offset_mapping pour aligner entités
    encoding = tokenizer(
        texte,
        max_length=MAX_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    tokens     = encoding['input_ids']
    offset_map = encoding['offset_mapping']

    # Initialiser tous les tags à O
    ner_tags = [LABEL2ID["O"]] * len(tokens)

    # Pour chaque entité, trouver les tokens qui la chevauchent
    # et assigner B- au premier, I- aux suivants.
    # On itère par entité (et non par token) pour gérer correctement
    # le décalage d'offset SentencePiece (▁Paul → offset inclut l'espace).
    for ent in entites:
        ent_debut = ent['debut']
        ent_fin   = ent['fin']
        tag       = ent['tag']
        premier   = True

        for i, (debut_tok, fin_tok) in enumerate(offset_map):
            # Ignorer tokens spéciaux (<s>, </s>)
            if debut_tok == 0 and fin_tok == 0:
                continue

            # Chevauchement réel : token intersecte la span de l'entité
            # (couvre aussi le cas où ▁Paul a debut_tok=81 < ent_debut=82)
            if debut_tok < ent_fin and fin_tok > ent_debut:
                if premier:
                    # Ne surécrit que si le token n'a pas déjà été assigné
                    # par une entité précédente (évite les I- orphelins)
                    if ner_tags[i] == LABEL2ID["O"]:
                        ner_tags[i] = LABEL2ID[f"B-{tag}"]
                    premier = False
                else:
                    if ner_tags[i] == LABEL2ID["O"]:
                        ner_tags[i] = LABEL2ID[f"I-{tag}"]

    # Récupérer les tokens lisibles
    tokens_str = tokenizer.convert_ids_to_tokens(tokens)

    # Passe de correction BIO : tout I-TAG sans B-TAG précédent cohérent
    # est promu en B-TAG (cas de chevauchement d'entités en script 03)
    for i in range(len(ner_tags)):
        label = ID2LABEL[ner_tags[i]]
        if label.startswith("I-"):
            entite = label[2:]
            prev = ID2LABEL[ner_tags[i-1]] if i > 0 else "O"
            if prev not in (f"B-{entite}", f"I-{entite}"):
                ner_tags[i] = LABEL2ID[f"B-{entite}"]

    return {
        "id":        doc['id'],
        "annee":     doc['annee'],
        "tokens":    tokens_str,
        "input_ids": tokens,
        "ner_tags":  ner_tags
    }

# ─── CONVERTIR TOUS LES DOCUMENTS ───────────────────────
print("Conversion en format BIO...")
dataset = []
erreurs = 0

for doc in tqdm(documents):
    try:
        result = convertir_en_bio(doc)
        if result:
            dataset.append(result)
    except Exception as e:
        erreurs += 1

print(f"Documents convertis : {len(dataset)}")
print(f"Erreurs             : {erreurs}")

# ─── AFFICHER UN EXEMPLE LISIBLE ────────────────────────
print("\n=== EXEMPLE BIO (20 premiers tokens) ===")
ex = dataset[0]
for token, tag_id in zip(ex['tokens'][:20], ex['ner_tags'][:20]):
    label = ID2LABEL[tag_id]
    print(f"  {token:<20} -> {label}")

# ─── SPLIT TRAIN / VAL / TEST ───────────────────────────
print("\nSplit train/val/test...")

# Extraire les années pour stratification
annees = [d['annee'] for d in dataset]

# Split 80% train, 20% temp
train_data, temp_data, train_annees, temp_annees = train_test_split(
    dataset, annees,
    test_size=0.2,
    random_state=42,
    stratify=annees
)

# Split 50% val, 50% test sur le temp (= 10% / 10% du total)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_annees
)

print(f"Train : {len(train_data)} docs")
print(f"Val   : {len(val_data)} docs")
print(f"Test  : {len(test_data)} docs")

# Vérifier proportion 1973/1978 dans chaque split
for nom, split in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
    n73 = sum(1 for d in split if d['annee'] == '1973')
    n78 = sum(1 for d in split if d['annee'] == '1978')
    print(f"{nom} -- 1973: {n73} ({100*n73//len(split)}%) | 1978: {n78} ({100*n78//len(split)}%)")

# ─── STATS DISTRIBUTION DES TAGS ────────────────────────
all_tags = [ID2LABEL[t] for d in train_data for t in d['ner_tags']]
tag_counts = Counter(all_tags)
print(f"\n=== DISTRIBUTION DES TAGS (train) ===")
for tag, count in sorted(tag_counts.items()):
    print(f"  {tag:<10} : {count:>8}")

# ─── SAUVEGARDER ────────────────────────────────────────
(BASE_DIR / "data/bio").mkdir(exist_ok=True)

def sauvegarder(data, chemin):
    with open(chemin, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Sauvegarde -> {chemin} ({len(data)} docs)")

sauvegarder(train_data, BASE_DIR / "data/bio/train.json")
sauvegarder(val_data,   BASE_DIR / "data/bio/val.json")
sauvegarder(test_data,  BASE_DIR / "data/bio/test.json")

# Label mapping
with open(BASE_DIR / "data/bio/label2id.json", "w") as f:
    json.dump(LABEL2ID, f, indent=2)
print("Sauvegarde -> data/bio/label2id.json")

# Format CoNLL
def to_conll(data, chemin):
    lines = []
    for doc in data:
        for token, tag_id in zip(doc['tokens'], doc['ner_tags']):
            lines.append(f"{token}\t{ID2LABEL[tag_id]}")
        lines.append("")
    Path(chemin).write_text('\n'.join(lines), encoding='utf-8')
    print(f"Sauvegarde -> {chemin}")

to_conll(train_data, BASE_DIR / "data/bio/train.conll")
to_conll(val_data,   BASE_DIR / "data/bio/val.conll")
to_conll(test_data,  BASE_DIR / "data/bio/test.conll")

print("\nETAPE 4 TERMINEE -- donnees pretes pour fine-tuning CamemBERT")
