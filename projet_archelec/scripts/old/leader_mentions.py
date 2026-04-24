"""
Analyse des mentions de leaders nationaux dans les professions de foi.

Pour chaque document, compte combien de fois chaque leader est mentionné.
Résultat : mentions par doc, par année, normalisées par longueur du texte.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Leaders nationaux avec variantes orthographiques ─────────────────────────
LEADERS = {
    # Présidents / figures majeures
    "De Gaulle":       [r"de gaulle", r"degaulle", r"gaulliste", r"gaullisme"],
    "Pompidou":        [r"pompidou"],
    "Giscard":         [r"giscard", r"giscard d.estaing", r"vge"],
    "Mitterrand":      [r"mitterrand", r"mitterandiste"],
    "Chirac":          [r"chirac"],
    "Rocard":          [r"rocard"],
    "Marchais":        [r"marchais"],
    "Fabius":          [r"fabius"],
    "Jospin":          [r"jospin"],
    "Le Pen J-M":      [r"le pen", r"jean-marie le pen", r"front national"],
    # Pour les années récentes
    "Sarkozy":         [r"sarkozy"],
    "Hollande":        [r"hollande"],
    "Macron":          [r"macron", r"en marche", r"lrem"],
    "Le Pen Marine":   [r"marine le pen"],
    "Melenchon":       [r"m.lenchon", r"france insoumise", r"lfi"],
}

def compter_mentions(texte: str) -> dict:
    """Compte les mentions de chaque leader dans un texte (insensible à la casse)."""
    texte_low = texte.lower()
    counts = {}
    for leader, patterns in LEADERS.items():
        total = 0
        for pat in patterns:
            total += len(re.findall(pat, texte_low))
        if total > 0:
            counts[leader] = total
    return counts

def compter_phrases(texte: str) -> int:
    """Estime le nombre de phrases (découpage sur . ! ?)."""
    phrases = re.split(r'[.!?]+', texte)
    return max(1, len([p for p in phrases if len(p.strip()) > 10]))

# ── Chargement des données ────────────────────────────────────────────────────
splits_dir = BASE_DIR / "data" / "splits"
all_docs = []
for fname in ["train.json", "test_before_2000.json", "test_after_2000.json"]:
    path = splits_dir / fname
    if path.exists():
        with open(path, encoding="utf-8") as f:
            docs = json.load(f)
        all_docs.extend(docs)
        print(f"Chargé {fname} : {len(docs)} docs")

print(f"\nTotal : {len(all_docs)} docs\n")

# ── Analyse ───────────────────────────────────────────────────────────────────
resultats = []
mentions_par_annee = defaultdict(lambda: defaultdict(int))
docs_par_annee     = defaultdict(int)

for doc in all_docs:
    annee   = doc["annee"]
    texte   = doc["texte"]
    n_mots  = max(1, len(texte.split()))
    n_phrases = compter_phrases(texte)
    mentions  = compter_mentions(texte)

    docs_par_annee[annee] += 1

    for leader, count in mentions.items():
        mentions_par_annee[annee][leader] += count

    resultats.append({
        "id":       doc["id"],
        "annee":    annee,
        "n_mots":   n_mots,
        "n_phrases": n_phrases,
        "mentions": mentions,
        "total_mentions": sum(mentions.values()),
        # Taux : mentions pour 1000 mots
        "taux_mentions": round(sum(mentions.values()) / n_mots * 1000, 2)
    })

# ── Affichage par année ───────────────────────────────────────────────────────
print("=" * 60)
print("MENTIONS PAR ANNÉE (total sur tous les docs de l'année)")
print("=" * 60)

for annee in sorted(mentions_par_annee.keys()):
    n_docs   = docs_par_annee[annee]
    mentions = mentions_par_annee[annee]
    if not mentions:
        continue
    print(f"\n{annee} ({n_docs} docs) :")
    for leader, count in sorted(mentions.items(), key=lambda x: -x[1]):
        moy = count / n_docs
        print(f"  {leader:<20} {count:>4} mentions  ({moy:.2f}/doc)")

# ── Leaders les plus cités globalement ───────────────────────────────────────
print("\n" + "=" * 60)
print("CLASSEMENT GLOBAL DES LEADERS")
print("=" * 60)
global_counts = defaultdict(int)
for doc in resultats:
    for leader, count in doc["mentions"].items():
        global_counts[leader] += count

for leader, count in sorted(global_counts.items(), key=lambda x: -x[1]):
    print(f"  {leader:<20} {count:>5} mentions")

# ── Docs avec le plus de mentions ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOP 10 DOCS PAR NOMBRE DE MENTIONS DE LEADERS")
print("=" * 60)
top_docs = sorted(resultats, key=lambda x: -x["total_mentions"])[:10]
for d in top_docs:
    leaders_str = ", ".join(f"{l}({c})" for l, c in d["mentions"].items())
    print(f"  {d['annee']} | {d['id'][:30]:<30} | {d['total_mentions']} mentions : {leaders_str}")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
output_dir = BASE_DIR / "data" / "results" / "leader_mentions"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "mentions_par_doc.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

# Résumé par année
resume = {}
for annee in sorted(mentions_par_annee.keys()):
    resume[annee] = {
        "n_docs": docs_par_annee[annee],
        "mentions": dict(mentions_par_annee[annee])
    }
with open(output_dir / "resume_par_annee.json", "w", encoding="utf-8") as f:
    json.dump(resume, f, ensure_ascii=False, indent=2)

print(f"\nRésultats sauvegardés dans {output_dir}")
