"""
Script 04 — Téléchargement + annotation distant supervision
Utilise 10 workers en parallèle pour accélérer le téléchargement Archive.org.
Output : data/annotated/archelec_annotated_<annees>.json
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import re
import time
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent.parent
CSV_PATH   = BASE_DIR / 'data' / 'processed' / 'archelec_1958_2019.csv'
OUT_PATH   = BASE_DIR / 'data' / 'annotated' / 'archelec_annotated_new_years.json'
CHECKPOINT = BASE_DIR / 'data' / 'annotated' / 'archelec_annotated_new_years_checkpoint.json'

ANNEES_CIBLES = [1986, 1988, 1993, 2002, 2007, 2012]
MAX_PAR_ANNEE = 300
N_WORKERS     = 3
TIMEOUT       = 30

ABREVIATIONS = {
    'PS':    'Parti socialiste',
    'PCF':   'Parti communiste français',
    'RPR':   'Rassemblement pour la République',
    'UDF':   'Union pour la démocratie française',
    'FN':    'Front national',
    'MRG':   'Mouvement des radicaux de gauche',
    'PSU':   'Parti socialiste unifié',
    'PC':    'Parti communiste',
    'UDR':   'Union pour la défense de la République',
    'URP':   'Union des républicains de progrès',
    'P.S.':  'Parti socialiste',
    'P.C.F.':'Parti communiste français',
    'R.P.R.':'Rassemblement pour la République',
    'U.D.F.':'Union pour la démocratie française',
    'F.N.':  'Front national',
    'M.R.G.':'Mouvement des radicaux de gauche',
}


# ── Fonctions d'annotation ────────────────────────────────────────────────────

def normalise(texte):
    texte = texte.lower()
    texte = unicodedata.normalize('NFD', texte)
    texte = ''.join(c for c in texte if unicodedata.category(c) != 'Mn')
    texte = re.sub(r'[-\s]+', ' ', texte).strip()
    return texte


def chercher_entite(texte, valeur, tag):
    if not valeur or str(valeur).lower() in ['non mentionné', 'nan', '', 'none']:
        return []
    valeur = str(valeur).strip()
    resultats = []
    pattern = re.compile(re.escape(valeur), re.IGNORECASE)
    for m in pattern.finditer(texte):
        resultats.append({'texte': texte[m.start():m.end()], 'tag': tag,
                          'debut': m.start(), 'fin': m.end()})
    if not resultats:
        texte_norm  = normalise(texte)
        valeur_norm = normalise(valeur)
        if valeur_norm:
            for m in re.compile(re.escape(valeur_norm), re.IGNORECASE).finditer(texte_norm):
                resultats.append({'texte': valeur, 'tag': tag,
                                  'debut': m.start(), 'fin': m.end()})
    return resultats


def annoter_document(doc):
    texte   = doc['texte']
    entites = []

    for val in [doc.get('nom_complet'), doc.get('nom'), doc.get('nom_upper')]:
        trovato = chercher_entite(texte, val, 'PER')
        if trovato:
            entites.extend(trovato)
            break

    for val in [doc.get('suppleant_nom_complet'), doc.get('suppleant_nom')]:
        trovato = chercher_entite(texte, val, 'PER')
        if trovato:
            entites.extend(trovato)
            break

    for parti in doc.get('partis', []):
        entites.extend(chercher_entite(texte, parti, 'ORG'))

    for abrev in ABREVIATIONS:
        if abrev in texte:
            try:
                pos = texte.index(abrev)
                entites.append({'texte': abrev, 'tag': 'ORG',
                                'debut': pos, 'fin': pos + len(abrev)})
            except ValueError:
                pass

    dept = doc.get('departement', '')
    if dept and str(dept).lower() not in ['nan', '', 'none']:
        for val in [dept, str(dept).upper()]:
            entites.extend(chercher_entite(texte, val, 'LOC'))

    for prof in doc.get('professions', []):
        entites.extend(chercher_entite(texte, prof, 'MISC'))

    seen, entites_uniques = set(), []
    for e in entites:
        key = (e['debut'], e['fin'])
        if key not in seen:
            seen.add(key)
            entites_uniques.append(e)
    entites_uniques.sort(key=lambda x: x['debut'])
    return entites_uniques


def nettoyer_texte(texte):
    texte = re.sub(r'[ \t]{2,}', ' ', texte)
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    return texte.strip()


def csv_row_to_doc(row, texte):
    def safe(val):
        return '' if pd.isna(val) else str(val).strip()

    nom    = safe(row.get('titulaire-nom'))
    prenom = safe(row.get('titulaire-prenom'))
    nom_complet = f'{prenom} {nom}'.strip() if prenom else nom

    s_nom    = safe(row.get('suppleant-nom'))
    s_prenom = safe(row.get('suppleant-prenom'))
    s_complet = f'{s_prenom} {s_nom}'.strip() if s_prenom else s_nom

    liste = safe(row.get('titulaire-liste'))
    partis = [liste] if liste and liste.lower() not in ['non mentionné', ''] else []

    prof = safe(row.get('titulaire-profession'))
    professions = [prof] if prof and prof.lower() not in ['non mentionné', ''] else []

    return {
        'id':                    safe(row.get('id')),
        'annee':                 int(row['annee']),
        'texte':                 texte,
        'nom_complet':           nom_complet,
        'nom':                   nom,
        'nom_upper':             nom.upper() if nom else '',
        'prenom':                prenom,
        'suppleant_nom_complet': s_complet,
        'suppleant_nom':         s_nom,
        'suppleant_prenom':      s_prenom,
        'partis':                partis,
        'departement':           safe(row.get('departement-nom')),
        'professions':           professions,
    }


# ── Worker : télécharge + annote un doc ───────────────────────────────────────

_session_local = threading.local()

def get_session():
    if not hasattr(_session_local, 'session'):
        s = requests.Session()
        s.headers.update({'User-Agent': 'archelec-ner-research/1.0'})
        _session_local.session = s
    return _session_local.session


def traiter_row(row):
    doc_id = str(row.get('id', '')).strip()
    url    = str(row.get('ocr_url', '')).strip()
    if not url or url == 'nan':
        return None

    # Fallback via URL canonique Archive.org si le serveur direct timeout
    item_id   = doc_id
    url_fallback = f'https://archive.org/download/{item_id}/{item_id}_djvu.txt'
    urls_a_essayer = [url, url_fallback]

    session = get_session()
    for url_essai in urls_a_essayer:
        for tentative in range(2):
            try:
                resp = session.get(url_essai, timeout=TIMEOUT)
                if resp.status_code == 200:
                    texte = nettoyer_texte(resp.text)
                    doc   = csv_row_to_doc(row, texte)
                    entites = annoter_document(doc)
                    return {'id': doc['id'], 'annee': doc['annee'],
                            'texte': texte, 'entites': entites}
                if resp.status_code == 429:
                    time.sleep(10 * (tentative + 1))
                else:
                    time.sleep(2 + tentative)
            except Exception:
                time.sleep(3 + tentative * 2)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH, low_memory=False)
df_cible = df[df['annee'].isin(ANNEES_CIBLES)].copy()

# Limiter à MAX_PAR_ANNEE docs par année
df_cible = df_cible.groupby('annee').head(MAX_PAR_ANNEE).reset_index(drop=True)
print(f'Docs cibles ({MAX_PAR_ANNEE} max/année) : {len(df_cible)}')
for annee, n in df_cible['annee'].value_counts().sort_index().items():
    print(f'  {annee} : {n} docs')

# Reprendre depuis checkpoint
resultats  = []
ids_faits  = set()
if CHECKPOINT.exists():
    with open(CHECKPOINT, encoding='utf-8') as f:
        resultats = json.load(f)
    ids_faits = {r['id'] for r in resultats}
    print(f'\nCheckpoint : {len(resultats)} docs déjà traités, reprise...')

rows_a_traiter = [row for _, row in df_cible.iterrows()
                  if str(row.get('id', '')).strip() not in ids_faits]
print(f'À télécharger : {len(rows_a_traiter)} docs avec {N_WORKERS} workers\n')

lock = threading.Lock()
erreurs = 0
checkpoint_lock = threading.Lock()

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(traiter_row, row): row for row in rows_a_traiter}
    with tqdm(total=len(rows_a_traiter)) as pbar:
        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result:
                    resultats.append(result)
                else:
                    erreurs += 1
                pbar.update(1)
                pbar.set_postfix(ok=len(resultats), err=erreurs)

                if len(resultats) % 50 == 0:
                    with open(CHECKPOINT, 'w', encoding='utf-8') as f:
                        json.dump(resultats, f, ensure_ascii=False)

# Sauvegarde finale
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

if CHECKPOINT.exists():
    CHECKPOINT.unlink()

print(f'\n=== RÉSULTAT ===')
print(f'Docs annotés : {len(resultats)}')
print(f'Erreurs/skip : {erreurs}')
print(f'Sauvegardé → {OUT_PATH}')
