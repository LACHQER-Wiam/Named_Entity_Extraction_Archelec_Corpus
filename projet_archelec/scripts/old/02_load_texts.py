"""
Script 02 — Chargement et nettoyage des textes OCRises
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import re
import json
import pandas as pd
import ftfy
from pathlib import Path
from tqdm import tqdm

# ─── Années ───────────────────────────────────────────────────────────────────
YEAR_START = 1973
YEAR_END   = 1993

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Charger le CSV filtré
df = pd.read_csv(
    BASE_DIR / f"data/processed/archelec_{YEAR_START}_{YEAR_END}.csv",
    encoding="utf-8",
    low_memory=False,
)
print(f"Documents à traiter : {len(df)}")

# 3. Pour chaque document
documents = []
manquants = []

def split_valeur(val):
    if pd.isna(val):
        return []
    return [
        v.strip()
        for v in str(val).split(";")
        if v.strip() and v.strip() != "non mentionné"
    ]

for _, row in tqdm(df.iterrows(), total=len(df)):

    doc_id = str(row["id"])
    annee  = str(row["date"])[:4]
    chemin = BASE_DIR / f"data/raw/arkindex_archelec/text_files/{annee}/legislatives/{doc_id}.txt"

    # Lire le fichier texte
    texte_brut = None
    if chemin.exists():
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                texte_brut = chemin.read_text(encoding=encoding)
                break
            except Exception:
                continue

    if texte_brut is None:
        manquants.append(doc_id)
        continue

    # Corriger l'encodage avec ftfy
    texte_clean = ftfy.fix_text(texte_brut)

    # Nettoyer
    texte_clean = re.sub(r"\s+", " ", texte_clean).strip()
    texte_clean = re.sub(r"\n{3,}", "\n\n", texte_clean)

    # Ajouter le leader national
    if annee =="1973" :
        leader_national = "Georges Pompidou"
    elif annee =="1978" :
        leader_national = "Valéry Giscard d'Estaing"
    elif annee in ["1981", "1988", "1993"]:
        leader_national = "François Mitterrand" 


    documents.append({
        "id":          doc_id,
        "annee":       annee,
        "texte":       texte_clean,
        "nb_chars":    len(texte_clean),
        "nom":         str(row.get("titulaire-nom", "")),
        "prenom":      str(row.get("titulaire-prenom", "")),
        "nom_complet": f"{str(row.get('titulaire-prenom',''))} {str(row.get('titulaire-nom',''))}".strip(),
        "partis":      split_valeur(row.get("titulaire-soutien")),
        "departement": str(row.get("departement-nom", "")),
        "professions": split_valeur(row.get("titulaire-profession")),
        "suppleant_nom": str(row.get("suppleant-nom", "")),
        "suppleant_prenom": str(row.get("suppleant-prenom", "")),
        "suppleant_nom_complet": f"{str(row.get('suppleant-prenom',''))} {str(row.get('suppleant-nom',''))}".strip(),
        "id_circ": str(row.get("identifiant de circonscription", "")),
        "leader_national": leader_national
    })

# 4. Stats
print(f"\nDocuments charges    : {len(documents)}")
print(f"Fichiers manquants   : {len(manquants)}")
if documents:
    print(f"Longueur moyenne     : {sum(d['nb_chars'] for d in documents) // len(documents)} chars")
    print(f"Longueur min         : {min(d['nb_chars'] for d in documents)} chars")
    print(f"Longueur max         : {max(d['nb_chars'] for d in documents)} chars")

# 5. Afficher 2 exemples pour vérification visuelle
if len(documents) > 0:
    print("\n=== EXEMPLE 1 ===")
    print(f"ID       : {documents[0]['id']}")
    print(f"Nom      : {documents[0]['nom_complet']}")
    print(f"Suppleant: {documents[0]['suppleant_nom_complet']}")
    print(f"Partis   : {documents[0]['partis']}")
    print(f"Dept     : {documents[0]['departement']}")
    print(f"id_circ  : {documents[0]['id_circ']}")
    print(f"Profess. : {documents[0]['professions']}")
    print(f"Texte    : {documents[0]['texte'][:300]}")

if len(documents) > 100:
    print("\n=== EXEMPLE 2 ===")
    print(f"ID       : {documents[100]['id']}")
    print(f"Nom      : {documents[100]['nom_complet']}")
    print(f"Suppleant: {documents[100]['suppleant_nom_complet']}")
    print(f"Partis   : {documents[100]['partis']}")
    print(f"Dept     : {documents[100]['departement']}")
    print(f"id_circ  : {documents[100]['id_circ']}")
    print(f"Profess. : {documents[100]['professions']}")
    print(f"Texte    : {documents[100]['texte'][:300]}")

# 6. Sauvegarder les fichiers manquants
if manquants:
    (BASE_DIR / "data/processed/missing_files.log").write_text(
        "\n".join(manquants), encoding="utf-8"
    )
    print(f"\nListe des manquants -> data/processed/missing_files.log")

# 7. Sauvegarder
out_path = BASE_DIR / "data/processed/documents_clean.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"\nSauvegarde -> {out_path}")
