"""
Script 05 — Vérification qualité des données BIO
=================================================
Charge les données BIO finales, vérifie leur cohérence,
affiche des exemples annotés et génère un rapport qualité.
"""

import json
import random
import collections
from pathlib import Path
from typing import Iterator

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
BIO_DIR   = BASE_DIR / "data" / "bio"
PROC_DIR  = BASE_DIR / "data" / "processed"
RAPPORT   = PROC_DIR / "quality_report.txt"

LABEL2ID_PATH = BIO_DIR / "label2id.json"
SPLITS = ["train", "val", "test"]

RANDOM_SEED = 42


# ─── Chargement des données ───────────────────────────────────────────────────

def charger_jsonlines(chemin: Path) -> list[dict]:
    """
    Charge un fichier JSON lines (un objet JSON par ligne).
    """
    exemples = []
    with open(chemin, encoding="utf-8") as f:
        for i, ligne in enumerate(f, 1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                exemples.append(json.loads(ligne))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Ligne {i} invalide dans {chemin.name} : {e}")
    return exemples


def charger_conll(chemin: Path) -> list[list[tuple[str, str]]]:
    """
    Charge un fichier CoNLL et retourne une liste de documents.
    Chaque document = liste de (token, tag).
    """
    documents = []
    doc_courant: list[tuple[str, str]] = []

    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.rstrip("\n")
            if ligne == "":
                if doc_courant:
                    documents.append(doc_courant)
                    doc_courant = []
            else:
                parties = ligne.split("\t")
                if len(parties) == 2:
                    doc_courant.append((parties[0], parties[1]))
                else:
                    print(f"  [WARN] Ligne CoNLL malformée : '{ligne}'")

    if doc_courant:
        documents.append(doc_courant)

    return documents


# ─── Vérifications ────────────────────────────────────────────────────────────

class RapportQualite:
    """Accumule les résultats de vérification et génère un rapport texte."""

    def __init__(self):
        self.lignes: list[str] = []
        self.erreurs: int = 0
        self.avertissements: int = 0

    def ajouter(self, message: str, niveau: str = "INFO") -> None:
        """Ajoute une ligne au rapport."""
        self.lignes.append(f"[{niveau}] {message}")
        if niveau == "ERREUR":
            self.erreurs += 1
        elif niveau == "WARN":
            self.avertissements += 1

    def titre(self, texte: str) -> None:
        """Ajoute un titre de section."""
        separateur = "─" * 60
        self.lignes.append("")
        self.lignes.append(separateur)
        self.lignes.append(texte)
        self.lignes.append(separateur)

    def sauvegarder(self, chemin: Path) -> None:
        """Sauvegarde le rapport dans un fichier."""
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lignes))
        print(f"\nRapport sauvegardé : {chemin}")

    def afficher(self) -> None:
        """Affiche le rapport dans la console."""
        for ligne in self.lignes:
            print(ligne)


def verifier_sequences_bio(exemples: list[dict], label2id: dict,
                            id2label: dict, rapport: RapportQualite,
                            nom_split: str) -> dict:
    """
    Vérifie la cohérence BIO des séquences.

    Checks :
    1. Pas de tag I- sans B- précédent cohérent
    2. Pas de valeurs nulles ou vides
    3. Longueur < 512 tokens (limite CamemBERT)
    4. Cohérence entre ner_tags (indices) et tokens

    Retourne les statistiques calculées.
    """
    rapport.titre(f"Vérification du split : {nom_split.upper()}")

    nb_docs            = len(exemples)
    nb_tokens_total    = 0
    nb_erreurs_bio     = 0
    nb_sequences_longues = 0
    nb_nulls           = 0
    compteur_tags      = collections.Counter()
    longueurs_seq      = []

    for i, ex in enumerate(exemples):
        tokens   = ex.get("tokens", [])
        ner_tags = ex.get("ner_tags", [])

        # Vérification longueur cohérente
        if len(tokens) != len(ner_tags):
            rapport.ajouter(
                f"Doc #{i}: tokens ({len(tokens)}) ≠ ner_tags ({len(ner_tags)})",
                "ERREUR"
            )
            continue

        # Vérification nulls
        if any(t is None or t == "" for t in tokens):
            nb_nulls += 1
            rapport.ajouter(f"Doc #{i}: token nul ou vide détecté", "WARN")

        if any(t is None for t in ner_tags):
            nb_nulls += 1
            rapport.ajouter(f"Doc #{i}: ner_tag nul détecté", "ERREUR")
            continue

        # Vérification longueur
        nb_tokens = len(tokens)
        longueurs_seq.append(nb_tokens)
        nb_tokens_total += nb_tokens

        if nb_tokens > 510:   # 512 − 2 tokens spéciaux
            nb_sequences_longues += 1

        # Vérification BIO
        tag_precedent = "O"
        for j, tag_id in enumerate(ner_tags):
            tag = id2label.get(tag_id, f"INCONNU({tag_id})")
            compteur_tags[tag] += 1

            if tag.startswith("I-"):
                entite = tag[2:]
                if tag_precedent not in (f"B-{entite}", f"I-{entite}"):
                    nb_erreurs_bio += 1
                    rapport.ajouter(
                        f"Doc #{i}, token #{j} ('{tokens[j]}'): "
                        f"I-{entite} sans B-{entite} précédent "
                        f"(précédent : {tag_precedent})",
                        "WARN"
                    )
            tag_precedent = tag

    # Résumé
    rapport.ajouter(f"Documents analysés   : {nb_docs}")
    rapport.ajouter(f"Tokens totaux        : {nb_tokens_total}")
    if longueurs_seq:
        rapport.ajouter(
            f"Longueur moy/min/max : "
            f"{sum(longueurs_seq)/len(longueurs_seq):.1f} / "
            f"{min(longueurs_seq)} / {max(longueurs_seq)}"
        )
    rapport.ajouter(f"Séquences > 510 tok  : {nb_sequences_longues}")
    rapport.ajouter(f"Erreurs BIO (I- orphelin): {nb_erreurs_bio}")
    rapport.ajouter(f"Valeurs nulles/vides : {nb_nulls}")

    rapport.ajouter("Distribution des tags :")
    for tag, count in sorted(compteur_tags.items()):
        rapport.ajouter(f"  {tag:10s}: {count}")

    if nb_erreurs_bio == 0 and nb_nulls == 0 and nb_sequences_longues == 0:
        rapport.ajouter(f"✓ Split '{nom_split}' — aucune anomalie détectée", "INFO")
    else:
        rapport.ajouter(
            f"⚠ Split '{nom_split}' — {nb_erreurs_bio} erreurs BIO, "
            f"{nb_nulls} nulls, {nb_sequences_longues} séquences longues",
            "WARN"
        )

    return {
        "nb_docs": nb_docs,
        "nb_tokens": nb_tokens_total,
        "erreurs_bio": nb_erreurs_bio,
        "nulls": nb_nulls,
        "sequences_longues": nb_sequences_longues,
        "compteur_tags": compteur_tags,
    }


def afficher_exemples(exemples: list[dict], id2label: dict,
                      n: int = 5, seed: int = RANDOM_SEED) -> None:
    """
    Affiche n exemples annotés aléatoires pour vérification visuelle.

    Formatte la sortie pour faciliter la lecture : tokens colorés
    selon leur tag (en mode texte).
    """
    rng = random.Random(seed)
    selection = rng.sample(exemples, min(n, len(exemples)))

    print("\n" + "=" * 60)
    print(f"EXEMPLES ANNOTÉS (tirage aléatoire de {len(selection)})")
    print("=" * 60)

    for i, ex in enumerate(selection, 1):
        tokens   = ex["tokens"]
        ner_tags = ex["ner_tags"]

        print(f"\n── Exemple {i} ──────────────────────────────────")

        # Affichage aligné token / tag
        entites_en_cours: list[tuple[str, str]] = []  # (texte_entite, tag)
        token_courant  = []
        tag_courant    = None

        for token, tag_id in zip(tokens, ner_tags):
            tag = id2label.get(tag_id, "O")

            if tag.startswith("B-"):
                if token_courant:
                    entites_en_cours.append((" ".join(token_courant), tag_courant))
                token_courant = [token]
                tag_courant   = tag[2:]
            elif tag.startswith("I-") and tag_courant:
                token_courant.append(token)
            else:
                if token_courant:
                    entites_en_cours.append((" ".join(token_courant), tag_courant))
                    token_courant = []
                    tag_courant   = None

        if token_courant:
            entites_en_cours.append((" ".join(token_courant), tag_courant))

        # Affichage CoNLL
        for token, tag_id in zip(tokens[:30], ner_tags[:30]):
            tag = id2label.get(tag_id, "O")
            if tag != "O":
                print(f"  {token:25s}  {tag}")

        if len(tokens) > 30:
            print(f"  … ({len(tokens) - 30} tokens supplémentaires)")

        # Entités extraites
        if entites_en_cours:
            print("\n  Entités détectées :")
            for texte_ent, tag_ent in entites_en_cours:
                print(f"    [{tag_ent:4s}] {texte_ent}")
        else:
            print("\n  (aucune entité dans cet exemple)")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal du script de vérification qualité."""
    print("=" * 60)
    print("ÉTAPE 5 — Vérification qualité des données BIO")
    print("=" * 60)

    # ── 1. Chargement du mapping labels ─────────────────────────────────────
    if not LABEL2ID_PATH.exists():
        print(f"[ERREUR] label2id.json introuvable : {LABEL2ID_PATH}")
        raise SystemExit(1)

    with open(LABEL2ID_PATH, encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    rapport = RapportQualite()
    rapport.lignes.append("RAPPORT DE QUALITÉ — DONNÉES BIO ARCHELEC NER")
    rapport.lignes.append("=" * 60)

    tous_les_exemples: list[dict] = []
    stats_globales: dict[str, int] = collections.Counter()

    # ── 2. Vérification par split ─────────────────────────────────────────
    for split in SPLITS:
        json_path  = BIO_DIR / f"{split}.json"
        conll_path = BIO_DIR / f"{split}.conll"

        if not json_path.exists():
            rapport.ajouter(f"Fichier manquant : {json_path}", "ERREUR")
            continue

        exemples = charger_jsonlines(json_path)
        tous_les_exemples.extend(exemples)

        stats = verifier_sequences_bio(exemples, label2id, id2label,
                                       rapport, split)

        # Vérification du fichier CoNLL correspondant
        if conll_path.exists():
            docs_conll = charger_conll(conll_path)
            rapport.ajouter(
                f"CoNLL '{split}' : {len(docs_conll)} documents lus"
            )
            if len(docs_conll) != len(exemples):
                rapport.ajouter(
                    f"Divergence JSON ({len(exemples)}) vs CoNLL ({len(docs_conll)})",
                    "WARN"
                )
        else:
            rapport.ajouter(f"Fichier CoNLL manquant : {conll_path}", "WARN")

        for k, v in stats.items():
            if k not in ("compteur_tags",):
                stats_globales[k] += v if isinstance(v, int) else 0

    # ── 3. Résumé global ─────────────────────────────────────────────────
    rapport.titre("RÉSUMÉ GLOBAL")
    rapport.ajouter(f"Erreurs totales      : {rapport.erreurs}")
    rapport.ajouter(f"Avertissements totaux: {rapport.avertissements}")
    rapport.ajouter(f"Docs totaux          : {stats_globales.get('nb_docs', 0)}")
    rapport.ajouter(f"Tokens totaux        : {stats_globales.get('nb_tokens', 0)}")

    if rapport.erreurs == 0:
        rapport.ajouter("✓ Données valides — prêtes pour l'entraînement NER", "INFO")
    else:
        rapport.ajouter(
            f"⚠ {rapport.erreurs} erreurs à corriger avant l'entraînement", "ERREUR"
        )

    # ── 4. Exemples visuels ───────────────────────────────────────────────
    if tous_les_exemples:
        # Ne prendre que des exemples qui ont au moins une entité
        exemples_avec_entites = [
            ex for ex in tous_les_exemples
            if any(t != 0 for t in ex.get("ner_tags", []))
        ]
        afficher_exemples(
            exemples_avec_entites if exemples_avec_entites else tous_les_exemples,
            id2label,
            n=5,
        )

    # ── 5. Sauvegarde et affichage du rapport ────────────────────────────
    rapport.sauvegarder(RAPPORT)

    print("\n" + "=" * 60)
    print("BILAN FINAL")
    print("=" * 60)
    print(f"  Erreurs   : {rapport.erreurs}")
    print(f"  Warnings  : {rapport.avertissements}")

    if rapport.erreurs == 0 and rapport.avertissements == 0:
        print("\n✓ Toutes les vérifications sont passées.")
        print("  Les données sont prêtes pour fine-tuner CamemBERT-NER.")
    else:
        print(f"\nConsultez le rapport détaillé : {RAPPORT}")

    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Terminée")
    print("=" * 60)


if __name__ == "__main__":
    main()
