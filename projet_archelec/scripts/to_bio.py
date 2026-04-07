"""
Conversion au format BIO compatible spaCy/Hugging Face
Lit les fichiers split (train.json, test_before_2000.json, test_after_2000.json)
et crée les fichiers BIO avec tokens et ner_tags
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path
from transformers import AutoTokenizer

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
splits_dir = BASE_DIR / "data" / "splits"
output_dir = splits_dir  # Sauvegarder dans le même dossier

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

print("Chargement du tokenizer CamemBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ─── FONCTION DE CONVERSION BIO ──────────────────────────────────
def convertir_en_bio(doc):
    """
    Convertit un document annoté en format BIO aligné avec CamemBERT.
    """
    texte   = doc['texte']
    entites = doc.get('entites', [])

    # Tokeniser avec offset_mapping
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

    # Pour chaque entité, assigner B- et I- tags
    for ent in entites:
        ent_debut = ent['debut']
        ent_fin   = ent['fin']
        tag       = ent['tag']
        premier   = True

        for i, (debut_tok, fin_tok) in enumerate(offset_map):
            # Ignorer tokens spéciaux
            if debut_tok == 0 and fin_tok == 0:
                continue

            # Chevauchement réel
            if debut_tok < ent_fin and fin_tok > ent_debut:
                if premier:
                    if ner_tags[i] == LABEL2ID["O"]:
                        ner_tags[i] = LABEL2ID[f"B-{tag}"]
                    premier = False
                else:
                    if ner_tags[i] == LABEL2ID["O"]:
                        ner_tags[i] = LABEL2ID[f"I-{tag}"]

    # Récupérer les tokens lisibles
    tokens_str = tokenizer.convert_ids_to_tokens(tokens)

    # Passe de correction BIO
    for i in range(len(ner_tags)):
        label = ID2LABEL[ner_tags[i]]
        if label.startswith("I-"):
            if i == 0 or ID2LABEL[ner_tags[i-1]].startswith("O") or ID2LABEL[ner_tags[i-1]].split("-")[1] != label.split("-")[1]:
                ner_tags[i] = LABEL2ID[f"B-{label.split('-')[1]}"]

    return {
        "id": doc['id'],
        "annee": doc['annee'],
        "tokens": tokens_str,
        "input_ids": tokens,
        "ner_tags": ner_tags
    }

# ─── CHARGER ET CONVERTIR ──────────────────────────────────
input_files = [
    ('train.json', 'train_bio.json'),
    ('test_before_2000.json', 'test_before_2000_bio.json'),
    ('test_after_2000.json', 'test_after_2000_bio.json'),
]

for input_name, output_name in input_files:
    input_path = splits_dir / input_name
    output_path = output_dir / output_name

    if not input_path.exists():
        print(f" {input_name} non trouvé, ignoré")
        continue

    print(f"\nLecture {input_name}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Conversion de {len(documents)} documents...")
    bio_docs = []
    for doc in documents:
        bio_doc = convertir_en_bio(doc)
        bio_docs.append(bio_doc)

    print(f"Sauvegarde {output_name}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bio_docs, f, ensure_ascii=False, indent=2)

    print(f"✓ {output_name}: {len(bio_docs)} documents")

# ─── SAUVEGARDER LABEL2ID ──────────────────────────────────
label2id_path = output_dir / "label2id.json"
with open(label2id_path, 'w', encoding='utf-8') as f:
    json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)
print(f"\n✓ label2id.json sauvegardé")

print(f"\nTous les fichiers BIO sont dans: {output_dir}")
