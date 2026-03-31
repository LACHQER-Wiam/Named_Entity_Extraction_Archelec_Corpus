"""
Script 01 — Chargement et exploration du CSV Archelec
======================================================
Colonnes réelles détectées :
  id, date, subject, title, contexte-election, contexte-tour, cote,
  departement, departement-nom, departement-insee,
  identifiant de circonscription, images, pdf, ocr_url,
  titulaire-nom, titulaire-prenom, titulaire-sexe, titulaire-age,
  titulaire-age-calcule, titulaire-age-tranche, titulaire-profession,
  titulaire-mandat-en-cours, titulaire-mandat-passe,
  titulaire-associations, titulaire-autres-statuts, titulaire-soutien,
  titulaire-liste, titulaire-decorations,
  suppleant-nom, suppleant-prenom, ...
"""

import sys
import io
from pathlib import Path

# Forcer UTF-8 sur la sortie standard (Windows cp1252 par defaut)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # pas de fenêtre graphique
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Années ───────────────────────────────────────────────────────────────────
YEAR_START = 1973
YEAR_END   = 1993

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "archelec.csv"
OUT_CSV  = BASE_DIR / "data" / "processed" / f"archelec_{YEAR_START}_{YEAR_END}.csv"
OUT_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colonnes clés (noms réels dans le CSV)
COL_ID         = "id"
COL_DATE       = "date"
COL_NOM        = "titulaire-nom"
COL_PRENOM     = "titulaire-prenom"
COL_SOUTIEN    = "titulaire-soutien"       # parti / soutien politique
COL_PROFESSION = "titulaire-profession"
COL_DEPT       = "departement"             # numéro département
COL_DEPT_NOM   = "departement-nom"
COL_ELECTION   = "contexte-election"
COL_SUPP_NOM   = "suppleant-nom"
COL_SUPP_PRENOM= "suppleant-prenom"
COL_ID_CIRC     = "identifiant de circonscription"


def main() -> None:
    print("=" * 60)
    print("ÉTAPE 1 — Exploration du CSV Archelec")
    print("=" * 60)

    # ── 1. Chargement ──────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"[ERREUR] CSV introuvable : {CSV_PATH}")
        sys.exit(1)

    print(f"\nChargement : {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
    # Re-encoder les colonnes string : latin-1 bytes -> UTF-8 correct
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: x.encode("latin-1").decode("utf-8") if isinstance(x, str) else x
        )
    df.columns = df.columns.str.strip()

    # ── 2. Informations générales ───────────────────────────────────────────
    print(f"\nDimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print("\nColonnes :")
    print(df.columns.tolist())
    print("\n10 premiers exemples :")
    print(df[[COL_ID, COL_DATE, COL_NOM, COL_SOUTIEN,
              COL_PROFESSION, COL_DEPT_NOM, COL_SUPP_NOM, COL_SUPP_PRENOM, COL_ID_CIRC]].head(10).to_string())
    print("\nTypes de données :")
    print(df.dtypes.to_string())
    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum().to_string())

    # ── 3. Filtre 1973 / 1978 ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"Filtrage sur les années entre {YEAR_START} et {YEAR_END}…")
    df[COL_DATE] = df[COL_DATE].astype(str).str.strip()
    df_final = df[
        df[COL_DATE].str.startswith(('1973', '1978', '1981', '1988', '1993'))
    ].copy().reset_index(drop=True)

    print(f"→ {len(df_final)} documents retenus (sur {len(df)} au total)")

    # ── 4. Statistiques du sous-ensemble ────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STATISTIQUES {YEAR_START} / {YEAR_END}")
    print("=" * 60)

    # Année extraite (4 premiers caractères de la date)
    df_final["annee"] = df_final[COL_DATE].str[:4]

    print("\nNombre de documents par année :")
    print(df_final["annee"].value_counts().sort_index().to_string())

    print("\nTop 10 — titulaire-soutien (parti) :")
    print(df_final[COL_SOUTIEN].value_counts().head(10).to_string())

    print("\nTop 10 — titulaire-profession :")
    print(df_final[COL_PROFESSION].value_counts().head(10).to_string())

    print("\nTop 10 — département :")
    print(df_final[COL_DEPT_NOM].value_counts().head(10).to_string())

    print("\nValeurs manquantes (colonnes clés) :")
    cols_cles = [COL_ID, COL_DATE, COL_NOM, COL_PRENOM,
                 COL_SOUTIEN, COL_PROFESSION, COL_DEPT_NOM, 
                 COL_ELECTION, COL_SUPP_NOM, 
                 COL_SUPP_PRENOM, COL_ID_CIRC]
    print(df_final[cols_cles].isnull().sum().to_string())

    # ── 5. Sauvegarde ────────────────────────────────────────────────────────
    df_final.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✓ Sauvegardé : {OUT_CSV}  ({len(df_final)} lignes)")

    # ── 6. Visualisations ────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", palette="muted")

    # Distribution par année
    fig, ax = plt.subplots(figsize=(5, 3))
    df_final["annee"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Documents par année ({YEAR_START}/{YEAR_END})")
    ax.set_xlabel("Année"); ax.set_ylabel("Nombre")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "distribution_annees.png", dpi=150)
    plt.close(fig)

    # Top 15 partis
    top_soutien = df_final[COL_SOUTIEN].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_soutien.plot(kind="barh", ax=ax, color="coral", edgecolor="white")
    ax.set_title("Top 15 partis / soutiens")
    ax.set_xlabel("Nombre de candidats")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "distribution_partis.png", dpi=150)
    plt.close(fig)

    # Top 20 professions
    top_prof = df_final[COL_PROFESSION].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_prof.plot(kind="barh", ax=ax, color="mediumseagreen", edgecolor="white")
    ax.set_title("Top 20 professions")
    ax.set_xlabel("Nombre de candidats")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "distribution_professions.png", dpi=150)
    plt.close(fig)

    print(f"\nVisualisations sauvegardées dans : {OUT_DIR}")
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Terminée")
    print("=" * 60)


if __name__ == "__main__":
    main()
