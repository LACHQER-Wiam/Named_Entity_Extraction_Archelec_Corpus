# Extraction d'entités nommées appliquée au corpus Archelec
### Analyse du discours des candidats aux législatives françaises 1973–1993

**Ossama Oualy — ENSAE Paris**

Projet de recherche en TAL appliqué au corpus **Archelec** — une collection de 10 598 professions de foi numérisées issues des cinq élections législatives françaises de 1973 à 1993.

L'étude s'articule autour de trois axes :
1. **Comparaison NER** — quatre architectures (SpaCy, CamemBERT en deux étapes, GLiNER zero-shot et fine-tuné) évaluées en correspondance exacte de span
2. **Sentiment envers le président** — classification binaire des mentions du président en exercice (6 342 mentions classifiées)
3. **Échelle spatiale du discours** — classification local / national / supranational de chaque lieu détecté par NER

---

## Corpus

Le corpus Archelec rassemble des professions de foi numérisées et transcrites par OCR, accompagnées de métadonnées structurées (département, bloc politique, parti, année) issues du portail [Sciences Po / CEVIPOF](https://archelec.sciencespo.fr/explorer).

### Répartition annuelle

| Année | Élection | Documents | Note |
|-------|----------|-----------|------|
| 1973 | Législatives | 3 843 | |
| 1978 | Législatives | 4 827 | |
| 1981 | Législatives | 1 500 | Collecte partielle |
| 1988 | Législatives | 305 | Sous-ensemble non aléatoire |
| 1993 | Législatives | 123 | Sous-ensemble non aléatoire |
| **Total** | | **10 598** | |

> Les analyses longitudinales doivent tenir compte du déséquilibre temporel : Sciences Po n'a pas systématiquement archivé les professions de foi de 1981, 1988 et 1993.

### Entités annotées

| Tag | Description | Exemple |
|-----|-------------|---------|
| `PER` | Candidat ou leader politique | `François Mitterrand` |
| `ORG` | Parti ou mouvement politique | `Parti Socialiste`, `RPR` |
| `LOC` | Lieu géographique | `Nord`, `Paris`, `Europe` |
| `MISC` | Profession du candidat | `Agriculteur`, `Instituteur` |

### Données d'entraînement et d'évaluation

| Ensemble | Période | Documents | Usage |
|---------|---------|-----------|-------|
| Annotations manuelles — train | 1973–1993 | **46 docs** | Fine-tuning GLiNER, CamemBERT stage 2 |
| Annotations manuelles — val | 1973–1993 | **30 docs** | Sélection modèle |
| Supervision distante — train | 1973, 1978 | **6 936 docs** | CamemBERT stage 1 |
| `test_before_2000` | 1973–1993 | 50 docs | Évaluation in-domain |
| `test_after_2000` | 2015–2020 | 50 docs | Évaluation out-of-domain (Regards Citoyens) |

---

## Modèles NER

### SpaCy
Pipeline français `fr_core_news_lg` (architecture transformer). Version zero-shot appliquée sans entraînement. Version fine-tunée ré-entraînée sur les 46 documents manuels via `spacy train`. Mapping des labels : `PERSON → PER`, `GPE/LOC → LOC`, `MISC → MISC`.

### CamemBERT en deux étapes
Modèle RoBERTa pré-entraîné sur 138 Go de français, adapté par classification de tokens avec un curriculum en deux étapes :

- **Étape 1 — Supervision distante** : `camembert-base` + tête linéaire (9 étiquettes BIO), 6 936 docs annotés automatiquement, split 80/10/10, 5 époques, lr = 2×10⁻⁵, batch 16, fp16, early stopping sur F1 globale
- **Étape 2 — Fine-tuning gold** : reprise du modèle stage 1, 46 docs manuels, lr = 1×10⁻⁵, jusqu'à 20 époques, batch 8, early stopping patience = 5

### GLiNER zero-shot
Modèle `urchade/gliner_multi-v2.1` (195 M paramètres, backbone DeBERTa). Labels en langage naturel : `["person", "political party/political movement", "location", "profession"]`. Documents longs découpés en blocs de 300 caractères sur les frontières de phrase (fenêtre de 384 tokens). Seuil balayé de 0.1 à 0.9 — **optimum retenu : 0.5** (compromis cross-split).

### GLiNER fine-tuné (checkpoint-800)
Même base, fine-tuné sur 46 docs manuels :

| Paramètre | Valeur |
|-----------|--------|
| Backbone gelé | Couches 0–9 (DeBERTa) |
| Paramètres entraînés | Couches 10–11 + tête span = **22 M / 195 M** |
| Fonction de perte | Focal loss α=0.75, γ=2 |
| Learning rate | 1×10⁻⁴ |
| Steps | 800 (~6 époques), batch 2 |
| Warmup | 20 pas, planificateur linéaire |

---

## Résultats NER

**Métrique** : F1 stricte au niveau du span (correspondance exacte de frontière et de label). Seuil GLiNER = 0.5.

### Tableau complet (P, R, F1 globaux + F1 par type)

| Modèle | Split | P | R | **F1** | PER | ORG | LOC | MISC |
|--------|-------|---|---|--------|-----|-----|-----|------|
| SpaCy zero-shot | < 2000 | .077 | .245 | .118 | .304 | .138 | .092 | .016 |
| SpaCy fine-tuné | < 2000 | .292 | .208 | .243 | .521 | .178 | .413 | .000 |
| GLiNER zero-shot | < 2000 | .157 | .336 | .214 | .348 | .189 | .152 | .178 |
| GLiNER fine-tuné | < 2000 | .203 | .477 | **.285** | .577 | .241 | .284 | .186 |
| CamemBERT 2-stage | < 2000 | **.630** | .366 | **.463** | **.740** | **.360** | **.552** | .000 |
| SpaCy zero-shot | > 2000 | .098 | .377 | .156 | .178 | .207 | .238 | .000 |
| SpaCy fine-tuné | > 2000 | .038 | .018 | .025 | .036 | .027 | .000 | .000 |
| GLiNER zero-shot | > 2000 | .130 | .361 | .191 | .167 | .230 | .285 | .023 |
| GLiNER fine-tuné | > 2000 | .150 | **.377** | **.214** | .206 | .223 | **.348** | **.036** |
| CamemBERT 2-stage | > 2000 | .194 | .185 | .189 | **.240** | **.215** | .050 | .000 |

### Observations clés

**CamemBERT domine en F1 sur les données in-domain (< 2000)** — F1 = 0.463, précision élevée (0.630) — mais s'effondre sur les données post-2000 (F1 = 0.189) : fort sur-apprentissage au domaine 1973–1993.

**GLiNER fine-tuné est le modèle de production retenu** pour les analyses aval, pour trois raisons :
1. **Meilleur rappel** sur l'avant 2000 (0.477 vs 0.366) — crucial pour capturer un maximum de mentions du président
2. **Meilleure généralisation** hors période (F1 0.214 vs 0.189) — meilleure robustesse linguistique
3. **Seul modèle à gérer MISC** (F1 0.186 vs 0.000 pour CamemBERT) — grâce à l'architecture par spans

**SpaCy fine-tuné s'effondre sur l'après 2000** (F1 0.025) — pire que la version zero-shot — révélant un sur-apprentissage critique au vocabulaire des années 1970–1990.

**PER est la classe la plus facile**, MISC la plus difficile (professions très variées, rarement mentionnées textuellement de façon exacte).

---

## Analyse du sentiment — mentions du président en exercice

### Question de recherche
Un candidat local préfère-t-il se rapprocher de la figure présidentielle ou s'en distancier ? La réponse varie selon le président et le bloc politique.

### Classifieur de sentiment

Tâche binaire (positif / négatif). 12 combinaisons d'embeddings × classifieurs testées :
- **TF-IDF** (unigrammes, bigrammes) × régression logistique, SVM linéaire, random forest
- **CamemBERT** ([CLS] ou mean pooling) × mêmes classifieurs
- CamemBERT fine-tuné cross-entropy

**Meilleure configuration retenue : TF-IDF unigrammes + régression logistique (seuil 0.5)**

| Métrique | Score |
|---------|-------|
| F1 macro (test, 39 docs) | **0.88** |
| F1 classe rare (négatif) | **0.84** |
| Erreurs (1 FP + 1 FN sur 39 docs) | 2 |

> Le CamemBERT fine-tuné échoue sur la classe rare (200 lignes d'entraînement, déséquilibre 73%/27%) — les indices lexicaux locaux suffisent pour cette tâche.

### Résultats

Le classifieur est appliqué aux **6 342 mentions du président** extraites par GLiNER fine-tuné.

#### Sentiment par président

| Président | Année | Mentions | % Positif | % Négatif |
|-----------|-------|----------|-----------|-----------|
| Pompidou | 1973 | 343 | **92%** | 8% |
| Giscard | 1978 | 157 | **84%** | 16% |
| Mitterrand | 1981–1993 | 5 842 | 45% | **55%** |

#### Évolution du sentiment négatif

| Année | % Négatif |
|-------|-----------|
| 1973 (Pompidou) | 8% |
| 1978 (Giscard) | 16% |
| 1981 (Mitterrand I) | 52% |
| 1988 (Mitterrand II) | 56% |
| 1993 (Mitterrand III) | **76%** |

#### Décomposition par bloc politique (période Mitterrand)

| Bloc | % Négatif | Note |
|------|-----------|------|
| Droite | **100%** | Systématique sur toute la période |
| Centre | **100%** | En 1993 |
| Gauche | 55% | Varie de 50% (1981) à 69% (1988) |

#### Décomposition par parti (mentions de Mitterrand, % négatif)

| Parti | Mentions | % Négatif |
|-------|----------|-----------|
| LO (Lutte Ouvrière) | — | **100%** |
| PT (Parti des Travailleurs) | — | **100%** |
| FN | 605 | **100%** |
| RPR | 63 | 94% |
| UDF | 55 | 85% |
| PCF | 1 107 | 73% |
| PS (parti du président) | 1 537 | 28% |

#### Signal Front National

| Année | Mentions FN | % du corpus annuel |
|-------|------------|-------------------|
| 1973 | 14 | 4.1% |
| 1978 | 51 | 1.8% |
| 1981 | — | — |
| 1988 | **551** | **19.3%** |
| 1993 | 3 | — |

#### Géographie du sentiment négatif

- **Plus critiques** : Puy-de-Dôme (68%), Charente-Maritime (68%), Vosges (67%), Haute-Garonne (67%)
- **Moins critiques** : Haut-Rhin (26%), Bas-Rhin (31%), Pas-de-Calais (30%), Moselle (30%)

> La variation géographique reflète la composition partisane locale plus que l'opinion publique : l'Alsace et la Moselle ont davantage de candidats centristes au ton moins polarisé ; Puy-de-Dôme et Haute-Garonne ont une forte présence PCF et courants de gauche non-PS.

---

## Analyse spatiale — échelle géographique du discours

### Méthodologie

Les entités LOC produites par GLiNER fine-tuné sont confrontées à un **gazetteer de 278 lieux français + 118 lieux internationaux**. Score d'échelle : 0 = local (département du candidat), 1 = national (autre lieu français), 2 = supranational (hors France). Le score d'un document est la moyenne de ses mentions.

### Résultats globaux

| Échelle | Part annuelle |
|---------|--------------|
| Locale | 28–42% |
| **Nationale** | **50–62%** (dominante) |
| Supranationale | 9–10% |

**Évolution du score d'échelle moyen :**

| Année | Score moyen |
|-------|-------------|
| 1973 | 0.77 |
| 1978 | — |
| 1981 | 0.78 |
| 1988 | **0.94** (pic) |
| 1993 | 0.80 |

**Par bloc politique :**

| Bloc | Score moyen | n |
|------|-------------|---|
| Centre | **0.84** | 72 |
| Non classé | 0.79 | 8 492 |
| Droite | 0.74 | 10 |
| Gauche | 0.69 | 1 601 |

> Le centre cite plus souvent les institutions européennes (positionnement pro-européen UDF/Giscard). La gauche reste ancrée sur les enjeux locaux dans les professions de foi, même si l'européanisme existe au niveau présidentiel.

### Top lieux détectés

| Échelle | Lieu | Mentions |
|---------|------|----------|
| Local | Paris | 1 599 |
| Local | Nord | 385 |
| National | France | **14 277** |
| National | Bretagne | 993 |
| National | Marseille | 583 |
| Supranational | **Europe** | **2 536** |
| Supranational | Tiers-Monde | 443 |
| Supranational | URSS | 285 |
| Supranational | Hanoï | 155 |

### Évolution des références européennes

| Année | Mentions Europe / 100 docs |
|-------|---------------------------|
| 1973 | 33.6 |
| 1981 | **8.4** (creux — agenda intérieur Mitterrand) |
| 1988 | **53.0** (pic — après Acte unique européen 1986, avant Maastricht 1992) |
| 1993 | 35.2 |

> Le pic de 1988 est le signal spatial le plus marquant : après l'Acte unique (1986), les candidats de tous les blocs mentionnent l'Europe — pour la défendre (centre, PS) ou la critiquer (FN, PCF).

---

## Structure du projet

```
projet_archelec/
├── data/
│   ├── raw/recent_pdf/              ← PDFs sources Regards Citoyens (test >2000)
│   ├── processed/
│   │   ├── archelec_1973_1993.csv   ← CSV filtré UTF-8 (corpus principal)
│   │   └── archelec_1958_2019.csv   ← CSV complet
│   ├── annotated/
│   │   ├── archelec_annotated.json          ← 8 670 docs supervision distante (52 MB)
│   │   └── archelec_gliner800_loc.json      ← 10 598 docs corpus NER production (27 MB)
│   ├── bio/                         ← données BIO CamemBERT (train/val/test)
│   ├── bio_distantsup_extended/     ← supervision distante étendue
│   ├── bio_manual/                  ← données CoNLL manuelles
│   ├── splits/
│   │   ├── train.json
│   │   ├── test_before_2000.json    ← 50 docs 1973–1993
│   │   └── test_after_2000.json     ← 50 docs 2015–2020 (Regards Citoyens)
│   └── results/
│       ├── Gliner/raw/              ← métriques zero-shot (seuils 0.1→0.9)
│       ├── Gliner/trained/          ← métriques fine-tuné (seuils 0.1→0.9)
│       ├── CamemBERT/               ← métriques (avant/après 2000)
│       ├── Spacy/raw/               ← métriques zero-shot
│       ├── Spacy/trained/           ← métriques fine-tuné
│       ├── sentiment_model_best/    ← modèle sentiment CamemBERT (non retenu)
│       └── output_best_model/
│           ├── leaders_sentiment_corpus.xlsx  ← 6 342 mentions + sentiment
│           ├── spatial_gliner800_results.csv  ← analyse spatiale complète
│           └── graphs/                        ← toutes les figures
├── notebooks/
│   ├── gliner.ipynb                 ← entraînement + évaluation GLiNER
│   ├── ner_comparison.ipynb         ← comparaison formelle des 4 modèles NER
│   ├── camembert_ner_archelec.ipynb ← fine-tuning CamemBERT NER (2 stages)
│   ├── sentiment_classification.ipynb ← comparaison 12 classifieurs sentiment
│   ├── leaders_detection.ipynb      ← extraction mentions leaders
│   ├── analyse_politique_finale.ipynb ← figures sentiment (fig1–6)
│   └── spatial_analysis.ipynb       ← figures spatiales (fig_spatial1–5)
├── scripts/
│   ├── 01_extract_text_arkindex.py        ← extraction texte corpus Arkindex (1973–1993)
│   ├── 02_extract_text_pdf_recent.py      ← extraction texte PDFs récents (Regards Citoyens)
│   ├── 03_split_train_test.py             ← création des splits temporels
│   ├── 04_annotate_distant_supervision.py ← annotation par supervision distante
│   ├── 05_download_annotate_extended.py   ← annotation corpus étendu
│   ├── 06_make_bio_distantsup.py          ← conversion au format BIO
│   ├── 07_train_camembert_stage1.py       ← CamemBERT stage 1 (supervision distante)
│   ├── 08_train_camembert_stage2.py       ← CamemBERT stage 2 (gold labels)
│   ├── 09_run_ner_inference.py            ← inférence GLiNER sur corpus complet
│   └── 10_evaluate_ner_models.py          ← F1/précision/rappel span-level
├── pyproject.toml
└── README.md
```

---

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) (Python 3.11, CUDA 12.4).

```bash
# Sur Onyxia : créer le venv dans un dossier persistant
uv venv /home/onyxia/work/.archelec_venv --python 3.11
ln -sf /home/onyxia/work/.archelec_venv .venv
uv sync

# Vérifier le GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
```

### Dépendances principales

| Paquet | Usage |
|--------|-------|
| `torch` + CUDA 12.4 | GPU Tesla T4 (15 GB) |
| `transformers` | CamemBERT fine-tuning |
| `gliner` | Modèle GLiNER |
| `spacy` + `fr_core_news_lg` | NLP français |
| `datasets` | Format HuggingFace |
| `ftfy` | Correction encodages OCR |
| `rapidfuzz` / `python-levenshtein` | Matching flou distant supervision |
| `scikit-learn` | Splits stratifiés, classifieurs sentiment |

### Récupérer le modèle GLiNER fine-tuné

Le checkpoint est stocké sur S3 MinIO (SSP Cloud) — les credentials expirent à chaque restart.

```bash
# Rafraîchir les credentials via le portail Onyxia, puis vérifier :
ls models/Gliner/

# Si vide, ré-entraîner (~10 min sur T4) :
jupyter nbconvert --to notebook --execute notebooks/gliner.ipynb
```

---

## Reproduction

```bash
# 1. Extraction du texte
uv run python scripts/01_extract_text_arkindex.py       # corpus Arkindex 1973–1993
uv run python scripts/02_extract_text_pdf_recent.py     # PDFs Regards Citoyens

# 2. Préparation des données
uv run python scripts/03_split_train_test.py            # splits temporels
uv run python scripts/04_annotate_distant_supervision.py # annotation automatique
uv run python scripts/05_download_annotate_extended.py  # corpus étendu
uv run python scripts/06_make_bio_distantsup.py         # format BIO

# 3. Entraînement CamemBERT
uv run python scripts/07_train_camembert_stage1.py      # stage 1 — supervision distante
uv run python scripts/08_train_camembert_stage2.py      # stage 2 — labels gold

# 4. GLiNER fine-tuning
# → notebooks/gliner.ipynb

# 5. Évaluation
uv run python scripts/09_run_ner_inference.py           # inférence corpus complet
uv run python scripts/10_evaluate_ner_models.py         # métriques NER
# → notebooks/ner_comparison.ipynb                      # comparaison formelle

# 6. Analyses du discours
# → notebooks/sentiment_classification.ipynb            # sélection classifieur sentiment
# → notebooks/leaders_detection.ipynb                   # extraction + classification
# → notebooks/analyse_politique_finale.ipynb            # figures fig1–6
# → notebooks/spatial_analysis.ipynb                    # figures fig_spatial1–5
```

---

## Limites

- **Annotations manuelles réduites** (46 docs entraînement) : les métriques NER restent modestes en absolu (F1 max = 0.46 sur avant 2000) ; les tendances qualitatives résistent au bruit mais les pourcentages exacts sont des estimations.
- **Classifieur sentiment petit** (200 mentions, 73%/27% déséquilibre) : F1 macro = 0.88 sur 39 docs test, mais l'incertitude statistique est non négligeable. Le modèle est opaque au niveau des décisions individuelles.
- **Gazetteer limité** (278 + 118 lieux) : les entités LOC non reconnues sont écartées de l'analyse spatiale.
- **Déséquilibre temporel** : 1988 (n=305) et 1993 (n=123) couvrent un sous-ensemble non aléatoire des circonscriptions — les tendances longitudinales doivent se lire en gardant les effectifs en tête.
- **Domain shift** : les modèles entraînés sur 1973–1993 généralisent mal aux données post-2000. GLiNER est le plus robuste à cet égard.
- **OCR** : les textes anciens (1973–1981) ont une qualité OCR variable qui peut affecter la détection NER.

---

## Infrastructure

| Ressource | Détail |
|-----------|--------|
| GPU | Tesla T4 (15 GB VRAM) |
| CUDA | 12.9 |
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| Plateforme | [SSP Cloud / Onyxia](https://datalab.sspcloud.fr/) |
| Stockage S3 | MinIO SSP Cloud (credentials session) |

---

## Références

1. Borthwick & Grishman (1999). *A maximum entropy approach to named entity recognition.*
2. Devlin et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.*
3. Grishman & Sundheim (1995). *Design of the MUC-6 evaluation.*
4. Hobbs et al. (1997). *FASTUS: A cascaded finite-state transducer for extracting information from natural-language text.*
5. Lafferty, McCallum & Pereira (2001). *Conditional random fields: Probabilistic models for segmenting and labeling sequence data.*
6. Lample et al. (2016). *Neural architectures for named entity recognition.*
7. Zaratiana et al. (2023). *GLiNER: Generalist model for named entity recognition using bidirectional transformer.*
