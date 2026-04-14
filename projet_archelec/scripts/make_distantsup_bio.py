"""
Génère les fichiers BIO pour la distant supervision (8670 docs, 1973/1978)
depuis archelec_annotated.json → data/bio_distantsup/
Split stratifié 80/10/10 par année.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

BASE_DIR   = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "data" / "annotated" / "archelec_annotated.json"
output_dir = BASE_DIR / "data" / "bio_distantsup"
output_dir.mkdir(exist_ok=True)

LABEL2ID = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6,
    "B-MISC": 7, "I-MISC": 8
}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 512
MODEL_NAME = "camembert-base"

print("Chargement du tokenizer CamemBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def convertir_en_bio(doc):
    texte   = doc["texte"]
    entites = doc.get("entites", [])

    encoding = tokenizer(
        texte,
        max_length=MAX_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=True
    )

    tokens     = encoding["input_ids"]
    offset_map = encoding["offset_mapping"]
    ner_tags   = [LABEL2ID["O"]] * len(tokens)

    for ent in entites:
        ent_debut, ent_fin, tag = ent["debut"], ent["fin"], ent["tag"]
        premier = True
        for i, (debut_tok, fin_tok) in enumerate(offset_map):
            if debut_tok == 0 and fin_tok == 0:
                continue
            if debut_tok < ent_fin and fin_tok > ent_debut:
                if premier:
                    if ner_tags[i] == LABEL2ID["O"]:
                        ner_tags[i] = LABEL2ID[f"B-{tag}"]
                    premier = False
                else:
                    if ner_tags[i] == LABEL2ID["O"]:
                        ner_tags[i] = LABEL2ID[f"I-{tag}"]

    tokens_str = tokenizer.convert_ids_to_tokens(tokens)

    # Correction BIO
    for i in range(len(ner_tags)):
        label = ID2LABEL[ner_tags[i]]
        if label.startswith("I-"):
            entite = label[2:]
            prev   = ID2LABEL[ner_tags[i-1]] if i > 0 else "O"
            if prev not in (f"B-{entite}", f"I-{entite}"):
                ner_tags[i] = LABEL2ID[f"B-{entite}"]

    return {
        "id":       doc["id"],
        "annee":    doc["annee"],
        "tokens":   tokens_str,
        "input_ids": tokens,
        "ner_tags": ner_tags
    }

# Charger
print(f"Chargement {input_path.name}...")
with open(input_path, encoding="utf-8") as f:
    documents = json.load(f)
print(f"{len(documents)} documents chargés")

# Split stratifié 80/10/10 par année
annees = [d["annee"] for d in documents]
train_docs, temp_docs = train_test_split(documents, test_size=0.2, random_state=42, stratify=annees)
annees_temp = [d["annee"] for d in temp_docs]
val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42, stratify=annees_temp)

print(f"Split : train={len(train_docs)} | val={len(val_docs)} | test={len(test_docs)}")

# Convertir et sauvegarder
for nom, docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
    print(f"Conversion {nom} ({len(docs)} docs)...")
    bio_docs = [convertir_en_bio(d) for d in docs]
    out = output_dir / f"{nom}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bio_docs, f, ensure_ascii=False)
    print(f"  ✓ {nom}.json sauvegardé ({out.stat().st_size/1024/1024:.1f} MB)")

# label2id
with open(output_dir / "label2id.json", "w", encoding="utf-8") as f:
    json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)

print(f"\nTous les fichiers dans : {output_dir}")
print(f"  train : {len(train_docs)} docs")
print(f"  val   : {len(val_docs)} docs")
print(f"  test  : {len(test_docs)} docs")
