import json
from collections import defaultdict
from pathlib import Path

# Chemins des fichiers
data_dir = Path(__file__).parent.parent / "data"
annotated_files = [
    data_dir / "annotated" / "archelec_annotated_1973_1981.json",
    data_dir / "annotated" / "archelec_annotated_1988_1993.json",
    data_dir / "annotated" / "archelec_annotated_2015_2020.json",
]
output_dir = data_dir / "splits"
output_dir.mkdir(exist_ok=True)

# Charger tous les documents
all_docs = []
for file_path in annotated_files:
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
            all_docs.extend(docs)
            print(f"✓ Chargé {len(docs)} documents de {file_path.name}")

print(f"\nTotal: {len(all_docs)} documents\n")

# Organiser par année
docs_by_year = defaultdict(list)
for doc in all_docs:
    year = int(doc['annee'])
    docs_by_year[year].append(doc)

print("Documents par année:")
for year in sorted(docs_by_year.keys()):
    print(f"  {year}: {len(docs_by_year[year])} documents")

# Split selon les règles
train = []
test_before_2000 = []
test_after_2000 = []

for year in sorted(docs_by_year.keys()):
    docs = docs_by_year[year]

    if year < 2000:
        # Avant 2000: 10 en train, 10 en test
        train.extend(docs[:10])
        test_before_2000.extend(docs[10:20])

        # Les documents restants aussi dans train
        if len(docs) > 20:
            train.extend(docs[20:])

        print(f"{year}: {len(docs)} docs → {min(10, len(docs))} train, {min(10, len(docs[10:20]))} test_before_2000")
    else:
        # Après 2000: tout dans test_after_2000
        test_after_2000.extend(docs)
        print(f"{year}: {len(docs)} docs → tous dans test_after_2000")

# Sauvegarder
output_files = {
    'train.json': train,
    'test_before_2000.json': test_before_2000,
    'test_after_2000.json': test_after_2000,
}

for filename, data in output_files.items():
    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {filename}: {len(data)} documents")

print(f"\n Fichiers sauvegardés dans: {output_dir}")
print(f"\nRésumé:")
print(f"  - train.json: {len(train)} documents")
print(f"  - test_before_2000.json: {len(test_before_2000)} documents")
print(f"  - test_after_2000.json: {len(test_after_2000)} documents")
print(f"  - Total: {len(train) + len(test_before_2000) + len(test_after_2000)} documents")
