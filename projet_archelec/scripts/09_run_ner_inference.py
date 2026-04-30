"""
Inférence NER sur le corpus Archelec avec le modèle Stage 2.
Lit les textes déjà téléchargés dans data/annotated/, produit data/results/entities.jsonl
"""
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path("/home/onyxia/work/Named_Entity_Extraction_Archelec_Corpus/projet_archelec")
STAGE2_MODEL = PROJECT_DIR / "models" / "stage2_model"
OUTPUT_FILE  = PROJECT_DIR / "data" / "results" / "entities.jsonl"
BATCH_SIZE   = 32
MAX_LENGTH   = 512

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ── Chargement modèle ─────────────────────────────────────────────────────────
print(f"Chargement {STAGE2_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model     = AutoModelForTokenClassification.from_pretrained(str(STAGE2_MODEL))
model.to(device).eval()

LABEL2ID = model.config.label2id
ID2LABEL = model.config.id2label
print(f"Labels : {ID2LABEL}")

# ── Chargement des docs avec texte ────────────────────────────────────────────
print("Chargement des textes...")
docs = []
seen_ids = set()
for fname in [
    "archelec_annotated.json",
    "archelec_annotated_1973_1981.json",
    "archelec_annotated_1981.json",
    "archelec_annotated_1988_1993.json",
    "archelec_annotated_2015_2020.json",
]:
    fpath = PROJECT_DIR / "data" / "annotated" / fname
    raw = json.load(open(fpath, encoding="utf-8"))
    for d in raw:
        if d["id"] not in seen_ids:
            seen_ids.add(d["id"])
            docs.append({"id": d["id"], "annee": d["annee"], "texte": d["texte"]})

print(f"{len(docs)} docs chargés")

# ── Extraction de spans BIO → entités ─────────────────────────────────────────
def extraire_entites(tokens, labels, offsets, texte):
    entites = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            tag   = lab[2:]
            debut = offsets[i][0]
            j     = i + 1
            while j < len(labels) and labels[j] == f"I-{tag}":
                j += 1
            fin        = offsets[j - 1][1]
            texte_ent  = texte[debut:fin].strip()
            if texte_ent:
                entites.append({"texte": texte_ent, "type": tag, "debut": int(debut), "fin": int(fin)})
            i = j
        else:
            i += 1
    return entites

# ── Inférence par batch ────────────────────────────────────────────────────────
print(f"Inférence (batch={BATCH_SIZE})...")
total = len(docs)
written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for batch_start in range(0, total, BATCH_SIZE):
        batch = docs[batch_start : batch_start + BATCH_SIZE]

        encodings = tokenizer(
            [d["texte"] for d in batch],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mappings = encodings.pop("offset_mapping")
        input_ids      = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        for i, doc in enumerate(batch):
            mask    = attention_mask[i].cpu().numpy()
            offsets = offset_mappings[i].numpy()
            preds   = predictions[i]
            labels  = [ID2LABEL[p] for p, m in zip(preds, mask) if m == 1]
            offs    = [tuple(o) for o, m in zip(offsets, mask) if m == 1]

            # Ignorer tokens spéciaux (offset (0,0))
            labels_clean = [l for l, o in zip(labels, offs) if o != (0, 0)]
            offs_clean   = [o for o in offs if o != (0, 0)]

            entites = extraire_entites(None, labels_clean, offs_clean, doc["texte"])

            record = {
                "id":      doc["id"],
                "annee":   doc["annee"],
                "entites": entites,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

        if batch_start % (BATCH_SIZE * 20) == 0:
            pct = (batch_start + len(batch)) / total * 100
            print(f"  {batch_start + len(batch):>6}/{total} ({pct:.1f}%)", flush=True)

print(f"\nTerminé : {written} docs → {OUTPUT_FILE}")

# ── Stats rapides ─────────────────────────────────────────────────────────────
print("\nStats entités extraites :")
from collections import Counter
counts = Counter()
with open(OUTPUT_FILE, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        for e in rec["entites"]:
            counts[e["type"]] += 1

total_ents = sum(counts.values())
for tag, n in sorted(counts.items()):
    print(f"  {tag:<8} : {n:>8} ({n/total_ents*100:.1f}%)")
print(f"  TOTAL    : {total_ents:>8}")
