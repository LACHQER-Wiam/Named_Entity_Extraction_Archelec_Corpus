"""
Script 03 — Annotation automatique par distant supervision
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


def normalise(texte):
    """Supprime accents et met en minuscule pour comparaison floue."""
    texte = texte.lower()
    texte = unicodedata.normalize('NFD', texte)
    texte = ''.join(c for c in texte if unicodedata.category(c) != 'Mn')
    texte = re.sub(r'[-\s]+', ' ', texte).strip()
    return texte


def chercher_entite(texte, valeur, tag):
    """
    Cherche 'valeur' dans 'texte' de façon robuste.
    Retourne liste de {texte, tag, debut, fin} ou [] si pas trouvé.
    """
    if not valeur or valeur.lower() in ['non mentionné', 'nan', '']:
        return []

    resultats = []

    # Méthode 1 : recherche exacte (case insensitive)
    pattern = re.compile(re.escape(valeur), re.IGNORECASE)
    for m in pattern.finditer(texte):
        resultats.append({
            "texte": texte[m.start():m.end()],
            "tag": tag,
            "debut": m.start(),
            "fin": m.end()
        })

    # Méthode 2 : recherche normalisée si rien trouvé
    if not resultats:
        texte_norm = normalise(texte)
        valeur_norm = normalise(valeur)
        pattern_norm = re.compile(re.escape(valeur_norm), re.IGNORECASE)
        for m in pattern_norm.finditer(texte_norm):
            # Récupérer la position dans le texte original
            resultats.append({
                "texte": valeur,
                "tag": tag,
                "debut": m.start(),
                "fin": m.end()
            })

    return resultats


def annoter_document(doc):
    """Annote un document avec toutes ses entités."""
    texte = doc['texte']
    entites = []
    stats = {"per": 0, "org": 0, "loc": 0, "misc": 0}

    # --- PER : nom du candidat ---
    nom_complet = doc.get('nom_complet', '')
    nom = doc.get('nom', '')
    prenom = doc.get('prenom', '')

    # Chercher nom complet d'abord
    trovato = chercher_entite(texte, nom_complet, 'PER')
    if trovato:
        entites.extend(trovato)
        stats['per'] += len(trovato)
    else:
        # Chercher nom seul en majuscule
        if nom:
            trovato = chercher_entite(texte, nom.upper(), 'PER')
            if trovato:
                entites.extend(trovato)
                stats['per'] += len(trovato)
            else:
                trovato = chercher_entite(texte, nom, 'PER')
                if trovato:
                    entites.extend(trovato)
                    stats['per'] += len(trovato)

    # --- PER : nom du suppleant ---
    nom_complet = doc.get('suppleant_nom_complet', '')
    nom = doc.get('suppleant_nom', '')
    prenom = doc.get('suppleant_prenom', '')

    # Chercher nom complet d'abord
    trovato = chercher_entite(texte, nom_complet, 'PER')
    if trovato:
        entites.extend(trovato)
        stats['per'] += len(trovato)
    else:
        # Chercher nom seul en majuscule
        if nom:
            trovato = chercher_entite(texte, nom.upper(), 'PER')
            if trovato:
                entites.extend(trovato)
                stats['per'] += len(trovato)
            else:
                trovato = chercher_entite(texte, nom, 'PER')
                if trovato:
                    entites.extend(trovato)
                    stats['per'] += len(trovato)

    # --- ORG : partis politiques ---
    for parti in doc.get('partis', []):
        trovato = chercher_entite(texte, parti, 'ORG')
        entites.extend(trovato)
        stats['org'] += len(trovato)

    # Chercher aussi abréviations connues
    abreviations = {
        'PS': 'Parti socialiste',
        'PCF': 'Parti communiste français',
        'UDR': 'Union pour la défense de la République',
        'RPR': 'Rassemblement pour la République',
        'UDF': 'Union pour la démocratie française',
        'URP': 'Union des républicains de progrès',
        'MRG': 'Mouvement des radicaux de gauche',
        'PSU': 'Parti socialiste unifié',
        'P.S.U.': 'Parti socialiste unifié',
        'P.S.': 'Parti socialiste',
        'P.C.F.': 'Parti communiste français',
        'U.D.R.': 'Union pour la défense de la République',
        'R.P.R.': 'Rassemblement pour la République',
        'U.D.F.': 'Union pour la démocratie française',
        'U.R.P.': 'Union des républicains de progrès',
        'M.R.G.': 'Mouvement des radicaux de gauche',
    }
    for abrev, nom_long in abreviations.items():
        if abrev in texte:
            entites.append({
                "texte": abrev,
                "tag": "ORG",
                "debut": texte.index(abrev),
                "fin": texte.index(abrev) + len(abrev)
            })
            stats['org'] += 1

    # --- LOC : département ---
    dept = doc.get('departement', '')
    if dept and dept.lower() not in ['nan', '']:
        trovato = chercher_entite(texte, dept, 'LOC')
        entites.extend(trovato)
        # Chercher aussi en majuscule
        trovato_maj = chercher_entite(texte, dept.upper(), 'LOC')
        entites.extend(trovato_maj)
        stats['loc'] += len(trovato) + len(trovato_maj)

    # --- LOC : circonscription ---
    id_circ = doc.get('id_circ', '')
    if id_circ  not in ['nan', '']:
        # Regex : capture "id_circ ... circonscription"
        pattern = rf"\b{id_circ}\s*\w*(?:\s+\w+)*\s+circonscription\b"
        matches = re.finditer(pattern, texte, flags=re.IGNORECASE)
        for m in matches:
            entites.append({
                "texte": texte[m.start():m.end()],
                "tag": "LOC",
                "debut": m.start(),
                "fin": m.end()
            })
        # Update stats
        stats['loc'] += len(list(re.finditer(pattern, texte, flags=re.IGNORECASE)))

    # --- MISC : professions ---
    for prof in doc.get('professions', []):
        trovato = chercher_entite(texte, prof, 'MISC')
        entites.extend(trovato)
        stats['misc'] += len(trovato)

    # --- PER : leaders nationaux ---
    leader_national = doc.get('leader_national', '')
    if leader_national:
        trovato = chercher_entite(texte, leader_national, 'PER')
        entites.extend(trovato)
        stats['per'] += len(trovato)


    # Supprimer doublons (même début et fin)
    seen = set()
    entites_uniques = []
    for e in entites:
        key = (e['debut'], e['fin'])
        if key not in seen:
            seen.add(key)
            entites_uniques.append(e)

    # Trier par position dans le texte
    entites_uniques.sort(key=lambda x: x['debut'])

    return entites_uniques, stats


# ─── MAIN ───────────────────────────────────────────────
with open(BASE_DIR / "data/processed/documents_clean.json", encoding='utf-8') as f:
    documents = json.load(f)

print(f"Documents à annoter : {len(documents)}")

resultats = []
stats_globales = {"per": 0, "org": 0, "loc": 0, "misc": 0, "total_entites": 0}
docs_avec_per  = 0
docs_avec_org  = 0
docs_avec_loc  = 0

for doc in tqdm(documents):
    entites, stats = annoter_document(doc)

    resultats.append({
        "id":      doc['id'],
        "annee":   doc['annee'],
        "texte":   doc['texte'],
        "entites": entites
    })

    if stats['per'] > 0: docs_avec_per += 1
    if stats['org'] > 0: docs_avec_org += 1
    if stats['loc'] > 0: docs_avec_loc += 1

    for k in stats_globales:
        if k in stats:
            stats_globales[k] += stats[k]
    stats_globales['total_entites'] += len(entites)

n = len(documents)
print(f"\n=== STATS D'ANNOTATION ===")
print(f"Docs avec PER  : {docs_avec_per}/{n} ({100*docs_avec_per//n}%)")
print(f"Docs avec ORG  : {docs_avec_org}/{n} ({100*docs_avec_org//n}%)")
print(f"Docs avec LOC  : {docs_avec_loc}/{n} ({100*docs_avec_loc//n}%)")
print(f"Total entités  : {stats_globales['total_entites']}")
print(f"Moyenne/doc    : {stats_globales['total_entites']//n}")

# Afficher 2 exemples annotés
print(f"\n=== EXEMPLE ANNOTÉ 1 ===")
ex = resultats[0]
print(f"Texte (300c) : {ex['texte'][:300]}")
print(f"Entités      :")
for e in ex['entites'][:10]:
    print(f"  [{e['tag']}] '{e['texte']}' (pos {e['debut']}-{e['fin']})")

print(f"\n=== EXEMPLE ANNOTÉ 2 ===")
ex = resultats[50]
print(f"Texte (300c) : {ex['texte'][:300]}")
print(f"Entités      :")
for e in ex['entites'][:10]:
    print(f"  [{e['tag']}] '{e['texte']}' (pos {e['debut']}-{e['fin']})")

# Sauvegarder
out_path = BASE_DIR / "data/annotated/archelec_annotated.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding='utf-8') as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

print(f"\nSauvegardé -> {out_path}")
