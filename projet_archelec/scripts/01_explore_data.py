"""
Script 01 — Chargement et exploration du CSV Archelec
======================================================
Charge le CSV brut, détecte les colonnes disponibles,
filtre sur les années 1973/1978, génère des stats et visualisations.
"""

import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
CSV_PATH   = RAW_DIR / "archelec.csv"
OUT_CSV    = PROC_DIR / "archelec_1973_1978.csv"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ─── Mapping des colonnes attendues ───────────────────────────────────────────
# Clé = nom interne utilisé dans ce projet
# Valeur = liste de noms possibles dans le CSV Archelec (ordre de priorité)
COLUMN_CANDIDATES = {
    "nom":         ["nom", "name", "candidat", "candidate", "nom_candidat", "lastname", "surname"],
    "prenom":      ["prenom", "prénom", "firstname", "first_name", "given_name"],
    "parti":       ["parti", "party", "liste", "organisation", "org", "parti_politique"],
    "commune":     ["commune", "ville", "city", "municipality", "localite", "localité"],
    "departement": ["departement", "département", "dept", "dep", "dpt"],
    "profession":  ["profession", "metier", "métier", "job", "occupation", "csp"],
    "annee":       ["annee", "année", "year", "election_year", "an", "date"],
    "id":          ["id", "doc_id", "identifiant", "uuid", "document_id", "arkindex_id"],
}


def detecter_colonnes(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Mappe les noms internes vers les noms réels des colonnes du CSV.

    Retourne un dict {nom_interne: nom_reel_ou_None}.
    Affiche un warning pour chaque colonne attendue non trouvée.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    for champ, candidats in COLUMN_CANDIDATES.items():
        trouve = None
        for candidat in candidats:
            if candidat.lower() in cols_lower:
                trouve = cols_lower[candidat.lower()]
                break
        if trouve is None:
            warnings.warn(
                f"[WARN] Colonne '{champ}' introuvable dans le CSV "
                f"(candidats testés : {candidats}). "
                "Cette colonne sera ignorée.",
                stacklevel=2,
            )
        mapping[champ] = trouve

    return mapping


def afficher_info_generale(df: pd.DataFrame) -> None:
    """Affiche les informations de base sur le DataFrame brut."""
    print("\n" + "=" * 60)
    print("INFORMATIONS GÉNÉRALES DU CSV BRUT")
    print("=" * 60)
    print(f"Dimensions      : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"\nColonnes ({len(df.columns)}) :")
    for col in df.columns:
        print(f"  - {col}")
    print("\n10 premiers exemples :")
    print(df.head(10).to_string())
    print("\nTypes de données :")
    print(df.dtypes.to_string())
    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum().to_string())


def filtrer_annees(df: pd.DataFrame, col_annee: str, annees: list[int]) -> pd.DataFrame:
    """
    Filtre le DataFrame sur les années spécifiées.

    Tente d'abord une conversion numérique de la colonne année.
    """
    df = df.copy()
    try:
        df[col_annee] = pd.to_numeric(df[col_annee], errors="coerce")
    except Exception as e:
        warnings.warn(f"[WARN] Impossible de convertir la colonne année : {e}")

    df_filtré = df[df[col_annee].isin(annees)].reset_index(drop=True)
    print(f"\n→ Filtre {annees} : {len(df_filtré)} documents retenus "
          f"(sur {len(df)} au total).")
    return df_filtré


def afficher_stats_subset(df: pd.DataFrame, mapping: dict) -> None:
    """Affiche les statistiques descriptives du sous-ensemble filtré."""
    print("\n" + "=" * 60)
    print("STATISTIQUES DU SOUS-ENSEMBLE 1973 / 1978")
    print("=" * 60)

    # Documents par année
    if mapping["annee"]:
        print("\nDocuments par année :")
        print(df[mapping["annee"]].value_counts().sort_index().to_string())

    # Documents par parti
    if mapping["parti"]:
        print("\nDocuments par parti (top 20) :")
        print(df[mapping["parti"]].value_counts().head(20).to_string())

    # Documents par département
    if mapping["departement"]:
        print("\nDocuments par département (top 20) :")
        print(df[mapping["departement"]].value_counts().head(20).to_string())

    # Distribution des professions
    if mapping["profession"]:
        print("\nProfessions les plus fréquentes (top 20) :")
        print(df[mapping["profession"]].value_counts().head(20).to_string())


def generer_visualisations(df: pd.DataFrame, mapping: dict, out_dir: Path) -> None:
    """Génère et sauvegarde les graphiques de distribution."""
    sns.set_theme(style="whitegrid", palette="muted")
    figures_créées = []

    # Histogramme par année
    if mapping["annee"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[mapping["annee"]].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="steelblue", edgecolor="white"
        )
        ax.set_title("Nombre de documents par année")
        ax.set_xlabel("Année")
        ax.set_ylabel("Nombre de documents")
        plt.tight_layout()
        path = out_dir / "distribution_annees.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_créées.append(str(path))

    # Bar chart top 15 partis
    if mapping["parti"]:
        top_partis = df[mapping["parti"]].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        top_partis.plot(kind="barh", ax=ax, color="coral", edgecolor="white")
        ax.set_title("Top 15 partis politiques")
        ax.set_xlabel("Nombre de candidats")
        plt.tight_layout()
        path = out_dir / "distribution_partis.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_créées.append(str(path))

    # Bar chart top 20 professions
    if mapping["profession"]:
        top_prof = df[mapping["profession"]].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        top_prof.plot(kind="barh", ax=ax, color="mediumseagreen", edgecolor="white")
        ax.set_title("Top 20 professions des candidats")
        ax.set_xlabel("Nombre de candidats")
        plt.tight_layout()
        path = out_dir / "distribution_professions.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_créées.append(str(path))

    # Bar chart top 20 départements
    if mapping["departement"]:
        top_dept = df[mapping["departement"]].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(8, 5))
        top_dept.plot(kind="barh", ax=ax, color="mediumpurple", edgecolor="white")
        ax.set_title("Top 20 départements")
        ax.set_xlabel("Nombre de candidats")
        plt.tight_layout()
        path = out_dir / "distribution_departements.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures_créées.append(str(path))

    if figures_créées:
        print(f"\nVisualisations sauvegardées ({len(figures_créées)}) :")
        for f in figures_créées:
            print(f"  → {f}")
    else:
        print("\n[WARN] Aucune visualisation générée (colonnes manquantes).")


def main() -> None:
    """Point d'entrée principal du script d'exploration."""
    print("=" * 60)
    print("ÉTAPE 1 — Exploration du CSV Archelec")
    print("=" * 60)

    # ── 1. Chargement ──────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"\n[ERREUR] Fichier CSV introuvable : {CSV_PATH}")
        print("Veuillez placer le fichier téléchargé depuis https://archelec.sciencespo.fr/explorer")
        print(f"dans le dossier : {RAW_DIR}")
        sys.exit(1)

    print(f"\nChargement de : {CSV_PATH}")
    try:
        # Essai UTF-8, puis latin-1 en fallback (encodages courants des CSV français)
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            print("[INFO] UTF-8 échoué, tentative en latin-1…")
            df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
        # Nettoyage des noms de colonnes (espaces parasites)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le CSV : {e}")
        sys.exit(1)

    # ── 2. Informations générales ───────────────────────────────────────────
    afficher_info_generale(df)

    # ── 3. Détection des colonnes ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DÉTECTION AUTOMATIQUE DES COLONNES")
    print("=" * 60)
    mapping = detecter_colonnes(df)
    print("\nMapping retenu :")
    for champ, col_reelle in mapping.items():
        statut = col_reelle if col_reelle else "⚠ NON TROUVÉE"
        print(f"  {champ:15s} → {statut}")

    # ── 4. Filtrage 1973 / 1978 ─────────────────────────────────────────────
    if mapping["annee"] is None:
        print("\n[ERREUR CRITIQUE] Colonne 'annee' introuvable — impossible de filtrer.")
        sys.exit(1)

    df_train = filtrer_annees(df, mapping["annee"], [1973, 1978])

    # ── 5. Statistiques du sous-ensemble ────────────────────────────────────
    afficher_stats_subset(df_train, mapping)

    # ── 6. Sauvegarde ───────────────────────────────────────────────────────
    df_train.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✓ Sous-ensemble sauvegardé : {OUT_CSV}")

    # ── 7. Visualisations ───────────────────────────────────────────────────
    generer_visualisations(df_train, mapping, PROC_DIR)

    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Terminée avec succès")
    print("=" * 60)


if __name__ == "__main__":
    main()
