# Archelec NER — Préparation des données pour CamemBERT

Pipeline de préparation de données d'entraînement pour la reconnaissance d'entités nommées (NER) sur le corpus **Archelec** — une collection de professions de foi de candidats aux élections législatives françaises de la Ve République.

L'objectif est de produire des données annotées prêtes à utiliser pour **fine-tuner CamemBERT** sur une tâche NER spécialisée en histoire politique française.

---

## Contexte du projet

Le corpus Archelec est constitué de professions de foi numérisées et OCRisées, accompagnées de métadonnées structurées (nom du candidat, parti, département, profession) issues du portail [Sciences Po / CEVIPOF](https://archelec.sciencespo.fr/explorer).

Ce projet se concentre sur les **élections législatives de 1973 et 1978**, deux scrutins clés de la Ve République (Union de la Gauche vs majorité giscardienne).

---

## Entités annotées

| Tag | Description | Source CSV |
|-----|-------------|-----------|
| `PER` | Nom du candidat | `titulaire-nom` + `titulaire-prenom` |
| `ORG` | Parti / soutien politique | `titulaire-soutien` |
| `LOC` | Département | `departement-nom` |
| `MISC` | Profession du candidat | `titulaire-profession` |

---

## Sources de données

| Source | Description | Accès |
|--------|-------------|-------|
| `archelec.csv` | Métadonnées des candidats (8 673 lignes, 42 colonnes) | [archelec.sciencespo.fr/explorer](https://archelec.sciencespo.fr/explorer) — téléchargement manuel |
| `arkindex_archelec/` | Textes OCRisés des professions de foi | `git clone https://gitlab.teklia.com/ckermorvant/arkindex_archelec` |

Les fichiers texte sont organisés ainsi :
```
data/raw/arkindex_archelec/text_files/
├── 1973/legislatives/*.txt   (3 921 fichiers)
└── 1978/legislatives/*.txt   (5 030 fichiers)
```

La clé de liaison CSV ↔ texte est la colonne `id` du CSV, qui correspond exactement au nom du fichier `.txt` sans extension (ex: `EL065_L_1973_03_001_01_1_PF_01`).

---

## Structure du projet

```
projet_archelec/
├── data/
│   ├── raw/
│   │   ├── archelec.csv                  ← CSV brut (latin-1, 8673 lignes)
│   │   └── arkindex_archelec/            ← repo GitLab cloné
│   ├── processed/
│   │   ├── archelec_1973_1978.csv        ← CSV filtré et re-encodé UTF-8
│   │   ├── documents_clean.json          ← 8670 textes nettoyés + métadonnées
│   │   ├── missing_files.log             ← 3 fichiers texte introuvables
│   │   ├── quality_report.txt            ← rapport de vérification qualité
│   │   └── distribution_*.png            ← visualisations exploratoires
│   ├── annotated/
│   │   └── archelec_annotated.json       ← 8670 docs avec entités annotées (51 MB)
│   └── bio/
│       ├── train.json                    ← 6 936 docs — entraînement (129 MB)
│       ├── val.json                      ← 867 docs — validation (16 MB)
│       ├── test.json                     ← 867 docs — évaluation finale (16 MB)
│       ├── label2id.json                 ← mapping labels → indices
│       ├── train.conll / val.conll / test.conll  ← format CoNLL pour inspection
├── scripts/
│   ├── 01_explore_data.py
│   ├── 02_load_texts.py
│   ├── 03_annotate.py
│   ├── 04_to_bio.py
│   └── 05_quality_check.py
├── models/                               ← modèles fine-tunés (à venir)
├── notebooks/                            ← exploration et analyse
├── pyproject.toml
└── README.md
```

---

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) comme gestionnaire d'environnement.

```bash
# Cloner ce repo et se placer dans le dossier
cd projet_archelec

# Créer le venv et installer toutes les dépendances (dont fr_core_news_lg)
uv sync

# Lancer un script
uv run python scripts/01_explore_data.py
```

### Dépendances principales

| Paquet | Usage |
|--------|-------|
| `transformers` | Tokenizer CamemBERT + futur Trainer |
| `datasets` | Format HuggingFace Dataset |
| `spacy` + `fr_core_news_lg` | NLP français |
| `ftfy` | Correction des encodages OCR cassés |
| `rapidfuzz` | Matching flou pour la distant supervision |
| `scikit-learn` | Split stratifié train/val/test |
| `pandas` | Manipulation du CSV |

---

## Pipeline — étape par étape

### Étape 1 — Exploration du CSV (`01_explore_data.py`)

- Charge `archelec.csv` (encodage latin-1, 8 673 lignes × 42 colonnes)
- Re-encode les chaînes latin-1 → UTF-8 pour corriger les accents
- Filtre sur les années **1973** (3 843 docs) et **1978** (4 830 docs)
- Génère des visualisations de distribution (partis, professions, départements)
- Sauvegarde `data/processed/archelec_1973_1978.csv`

Points notables sur les données :
- `titulaire-soutien` contient souvent plusieurs partis séparés par `;`
- `titulaire-profession` a 16 valeurs manquantes (négligeable)
- Top partis : PCF (942), PS (746), LO (630), RPR (409)
- Top département : Paris (718), Nord (432)

### Étape 2 — Chargement des textes (`02_load_texts.py`)

- Construit le chemin de chaque fichier texte depuis l'`id` CSV
- Essaie les encodages UTF-8, latin-1, cp1252 dans l'ordre
- Corrige les artéfacts OCR d'encodage avec **ftfy** (`Ã©` → `é`)
- Sépare les valeurs multiples de `titulaire-soutien` et `titulaire-profession` sur `;`, en excluant `"non mentionné"`
- Résultat : **8 670 documents** chargés (3 fichiers manquants)
- Longueur moyenne : **5 141 caractères** (min 139, max 25 120)
- Sauvegarde `data/processed/documents_clean.json`

### Étape 3 — Annotation par distant supervision (`03_annotate.py`)

La méthode de **distant supervision** consiste à utiliser les métadonnées structurées du CSV comme source d'annotations : chaque valeur connue (nom, parti, département, profession) est recherchée dans le texte correspondant.

Stratégie de recherche par entité :

| Tag | Stratégie |
|-----|-----------|
| `PER` | Recherche du nom complet, puis nom seul en majuscules, puis nom seul |
| `ORG` | Recherche exacte + abréviations connues (PS, PCF, RPR, UDF, MRG…) |
| `LOC` | Recherche exacte + version en majuscules (ex: `Ain` et `AIN`) |
| `MISC` | Recherche exacte du libellé de profession |

La recherche est **case-insensitive** en méthode 1, puis **normalisée** (sans accents, tirets ignorés) en méthode 2 si rien n'est trouvé — ce qui permet de retrouver `"CENTRE - PROGRES ET DEMOCRATIE MODERNE"` depuis `"Centre progrès et démocratie moderne"`.

Résultats :

| Tag | Couverture |
|-----|-----------|
| PER | **97%** des documents |
| ORG | **84%** des documents |
| LOC | **83%** des documents |

- **61 233 entités** annotées au total (moyenne 7 par document)
- Sauvegarde `data/annotated/archelec_annotated.json`

### Étape 4 — Conversion BIO (`04_to_bio.py`)

- Tokenise chaque texte avec le tokenizer **CamemBERT** (`camembert-base`), troncature à 512 tokens
- Aligne les entités sur les tokens via `offset_mapping` de HuggingFace
- Gère le subword tokenization SentencePiece : le token `▁Paul` a un offset décalé d'un espace par rapport à la position de l'entité → l'alignement utilise un test de chevauchement réel (`debut_tok < ent_fin and fin_tok > ent_debut`) plutôt qu'une égalité stricte
- Passe de correction post-tokenisation : tout `I-TAG` sans `B-TAG` précédent cohérent est promu en `B-TAG`
- **Split stratifié par année** (proportion 1973/1978 conservée dans chaque split) :
  - Train : 80% — **6 936 documents**
  - Val : 10% — **867 documents**
  - Test : 10% — **867 documents**

Format de sortie JSON (HuggingFace-compatible) :
```json
{
  "id": "EL065_L_1973_03_001_01_1_PF_01",
  "annee": "1973",
  "tokens": ["<s>", "▁Paul", "▁BAR", "BER", "OT", "▁est", "..."],
  "input_ids": [5, 1234, 567, 89, 10, 42, ...],
  "ner_tags": [0, 1, 2, 2, 2, 0, ...]
}
```

### Étape 5 — Vérification qualité (`05_quality_check.py`)

| Vérification | Résultat |
|---|---|
| Cohérence BIO (aucun `I-` sans `B-` précédent) | **0 erreur** |
| Longueur séquences ≤ 512 tokens | **0 dépassement** (moy. 503) |
| Valeurs nulles / tokens non alignés | **0** |
| Data leakage entre splits | **Aucun** |

---

## Label mapping

```json
{
  "O":      0,
  "B-PER":  1,  "I-PER":  2,
  "B-ORG":  3,  "I-ORG":  4,
  "B-LOC":  5,  "I-LOC":  6,
  "B-MISC": 7,  "I-MISC": 8
}
```

---

## Statistiques finales du dataset

| Split | Documents | Tokens | 1973 | 1978 |
|-------|-----------|--------|------|------|
| Train | 6 936 | ~3.49M | 44% | 56% |
| Val | 867 | ~436K | 44% | 56% |
| Test | 867 | ~436K | 44% | 56% |
| **Total** | **8 670** | **~4.36M** | | |

Distribution des entités (tous splits) :

| Tag | Occurrences B- | Occurrences I- |
|-----|---------------|---------------|
| PER | 10 778 | 39 492 |
| ORG | 13 358 | 59 052 |
| LOC | 10 181 | 17 474 |
| MISC | 3 150 | 4 832 |

---

## Prochaine étape — Fine-tuning CamemBERT

Les fichiers `data/bio/train.json`, `val.json`, `test.json` et `label2id.json` sont directement consommables par HuggingFace `Trainer` :

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import json

label2id = json.load(open("data/bio/label2id.json"))
id2label = {v: k for k, v in label2id.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "camembert-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)
```

---

## Limites connues

- **Distant supervision** : les annotations sont automatiques et peuvent contenir du bruit (entités non trouvées, faux positifs sur des noms courts ou ambigus). Un sous-ensemble de validation manuelle est recommandé avant publication.
- **OCR** : les textes sont issus de numérisation. Certains mots sont mal reconnus, ce qui peut faire échouer la recherche exacte.
- **Partis multiples** : `titulaire-soutien` liste parfois 2–3 partis séparés par `;`. Chaque parti est cherché indépendamment.
- **Commune absente** : la commune de circonscription n'est pas dans le CSV — seul le département est annoté comme `LOC`.

---

## Reproduction

```bash
uv run python scripts/01_explore_data.py   # → data/processed/archelec_1973_1978.csv
uv run python scripts/02_load_texts.py     # → data/processed/documents_clean.json
uv run python scripts/03_annotate.py       # → data/annotated/archelec_annotated.json
uv run python scripts/04_to_bio.py         # → data/bio/{train,val,test}.{json,conll}
uv run python scripts/05_quality_check.py  # → data/processed/quality_report.txt
```
