from collections import defaultdict
from prettytable import PrettyTable


def evaluate(data, extracted_entities):
    """
    Évalue les entités NER extraites par rapport aux annotations gold standard.
    
    Métriques calculées :
    ─────────────────────────────────────────────────────
    TP (True Positive)  : entité prédite correcte (texte ET label = gold)
    FP (False Positive) : entité prédite mais absente du gold (sur-détection)
    FN (False Negative) : entité gold non détectée par le modèle (sous-détection)
    
    Precision = TP / (TP + FP)
        → Parmi toutes les entités prédites, quelle proportion est correcte ?
        → Une precision faible = beaucoup de fausses détections
    
    Recall = TP / (TP + FN)
        → Parmi toutes les entités gold, quelle proportion a été trouvée ?
        → Un recall faible = beaucoup d'entités manquées
    
    F1-Score = 2 * Precision * Recall / (Precision + Recall)
        → Moyenne harmonique entre precision et recall
        → Métrique de référence en NER (CoNLL, SemEval)
    
    Support = nombre d'entités dans le gold pour ce type
        → Permet d'interpréter les scores (type rare = score moins fiable)
    
    Partial TP = entité avec le bon label mais span incomplet
        → ex: "Hollande" prédit vs "François Hollande" gold
        → Partial Precision = (TP + Partial TP) / total prédit
        → Partial Recall    = (TP + Partial TP) / total gold
    
    Micro-average (global) : agrège tous les TP/FP/FN avant de calculer
        → Donne plus de poids aux types fréquents (ORG, PER, LOC)
    
    Note : MISC est exclu de tous les calculs car non présent dans le gold.
    ─────────────────────────────────────────────────────
    """


    EXCLUDED_TAGS = {"MISC"}

    extracted_by_id = {e["id"]: e for e in extracted_entities}

    tp_per_type      = defaultdict(int)
    fp_per_type      = defaultdict(int)
    fn_per_type      = defaultdict(int)
    support_per_type = defaultdict(int)
    partial_tp_per_type = defaultdict(int)

    for doc in data:
        doc_id = doc["id"]

        # Gold : on exclut MISC
        gold_ents = set(
            (ent["texte"].strip().lower(), ent["tag"].strip().upper())
            for ent in extracted_by_id[doc_id].get("entites", [])
#            if ent["tag"].strip().upper() not in EXCLUDED_TAGS
        )

        # Prédictions : on exclut MISC
        pred_ents = set(
            (ent["texte"].strip().lower(), ent["tag"].strip().upper())
            for ent in doc.get("predicted_entities", [])
            # if ent["tag"].strip().upper() not in EXCLUDED_TAGS
        )

        # Support par type (gold uniquement)
        for _, tag in gold_ents:
            support_per_type[tag] += 1

        # Exact match : TP / FP / FN par type
        all_types = set(t for _, t in gold_ents | pred_ents)
        for label in all_types:
            gold_label = {e for e in gold_ents if e[1] == label}
            pred_label = {e for e in pred_ents if e[1] == label}
            tp_per_type[label] += len(gold_label & pred_label)
            fp_per_type[label] += len(pred_label - gold_label)
            fn_per_type[label] += len(gold_label - pred_label)

        # Partial match : bon label, texte partiellement chevauchant
        for pred_text, pred_tag in pred_ents:
            if (pred_text, pred_tag) not in gold_ents:  # pas déjà un TP exact
                for gold_text, gold_tag in gold_ents:
                    if pred_tag == gold_tag and (
                        pred_text in gold_text or gold_text in pred_text
                    ):
                        partial_tp_per_type[pred_tag] += 1
                        break  # une seule correspondance par prédiction

    def compute_metrics(tp, fp, fn):
        """Calcule precision, recall, f1 à partir de TP, FP, FN."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    # Micro-average global (exact)
    total_tp = sum(tp_per_type.values())
    total_fp = sum(fp_per_type.values())
    total_fn = sum(fn_per_type.values())
    global_p, global_r, global_f1 = compute_metrics(total_tp, total_fp, total_fn)

    # Micro-average global (partial)
    total_partial = sum(partial_tp_per_type.values())
    total_pred    = total_tp + total_fp
    total_gold    = total_tp + total_fn
    partial_p  = (total_tp + total_partial) / total_pred if total_pred > 0 else 0.0
    partial_r  = (total_tp + total_partial) / total_gold if total_gold > 0 else 0.0
    partial_f1 = 2 * partial_p * partial_r / (partial_p + partial_r) if (partial_p + partial_r) > 0 else 0.0

    # ── Affichage ──────────────────────────────────────────────────────

    print("=" * 45)
    print("Global NER Performance (Exact Match)")
    print("=" * 45)
    global_table = PrettyTable()
    global_table.field_names = ["Metric", "Value"]
    global_table.add_row(["Precision", f"{global_p:.4f}"])
    global_table.add_row(["Recall",    f"{global_r:.4f}"])
    global_table.add_row(["F1-Score",  f"{global_f1:.4f}"])
    print(global_table)

    print("\nGlobal NER Performance (Partial Match)")
    partial_table = PrettyTable()
    partial_table.field_names = ["Metric", "Value"]
    partial_table.add_row(["Precision", f"{partial_p:.4f}"])
    partial_table.add_row(["Recall",    f"{partial_r:.4f}"])
    partial_table.add_row(["F1-Score",  f"{partial_f1:.4f}"])
    print(partial_table)

    print("\nPerformance by Tag — Exact Match (MISC excluded)")
    tag_table = PrettyTable()
    tag_table.field_names = ["Tag", "Precision", "Recall", "F1-Score", "Support", "Partial TP"]
    all_labels = sorted(set(list(tp_per_type.keys()) + list(support_per_type.keys())))
    for label in all_labels:
        p, r, f = compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])
        tag_table.add_row([
            label,
            f"{p:.4f}",
            f"{r:.4f}",
            f"{f:.4f}",
            support_per_type[label],
            partial_tp_per_type[label]
        ])
    print(tag_table)

    return {
        "exact":   {"precision": global_p,  "recall": global_r,  "f1": global_f1},
        "partial": {"precision": partial_p, "recall": partial_r, "f1": partial_f1},
        "per_type": {
            label: {
                "precision":  compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])[0],
                "recall":     compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])[1],
                "f1":         compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])[2],
                "support":    support_per_type[label],
                "partial_tp": partial_tp_per_type[label]
            }
            for label in all_labels
        }
    }




######
# def evaluate(data, extracted_entities):
#     from collections import defaultdict
#     from prettytable import PrettyTable

#     extracted_by_id = {e["id"]: e for e in extracted_entities}

#     tp_per_type = defaultdict(int)
#     fp_per_type = defaultdict(int)
#     fn_per_type = defaultdict(int)
#     support_per_type = defaultdict(int)

#     # Pour partial match
#     partial_tp_per_type = defaultdict(int)

#     for doc in data:
#         doc_id = doc["id"]

#         gold_ents = list(
#             (ent["texte"].strip().lower(), ent["tag"].strip().upper())
#             for ent in doc.get("entites", [])
#         )
#         gold_set = set(gold_ents)

#         pred_ents = []
#         if doc_id in extracted_by_id:
#             pred_ents = list(
#                 (ent["texte"].strip().lower(), ent["tag"].strip().upper())
#                 for ent in extracted_by_id[doc_id].get("entities", [])
#             )
#         pred_set = set(pred_ents)

#         # Support (nombre d'entités gold par type)
#         for _, tag in gold_ents:
#             support_per_type[tag] += 1

#         # Exact match TP/FP/FN par type
#         all_types = set(t for _, t in gold_set | pred_set)
#         for label in all_types:
#             gold_label = {e for e in gold_set if e[1] == label}
#             pred_label = {e for e in pred_set if e[1] == label}
#             tp_per_type[label] += len(gold_label & pred_label)
#             fp_per_type[label] += len(pred_label - gold_label)
#             fn_per_type[label] += len(gold_label - pred_label)

#         # Partial match : même label, et un texte est contenu dans l'autre
#         for pred_text, pred_tag in pred_set:
#             for gold_text, gold_tag in gold_set:
#                 if pred_tag == gold_tag and (pred_text, pred_tag) not in gold_set:
#                     if pred_text in gold_text or gold_text in pred_text:
#                         partial_tp_per_type[pred_tag] += 1
#                         break  # compter une seule fois par prédiction

#     def compute_metrics(tp, fp, fn):
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
#         return precision, recall, f1

#     # Globaux (micro-average) — exact
#     total_tp = sum(tp_per_type.values())
#     total_fp = sum(fp_per_type.values())
#     total_fn = sum(fn_per_type.values())
#     global_p, global_r, global_f1 = compute_metrics(total_tp, total_fp, total_fn)

#     # Partial global
#     total_partial = sum(partial_tp_per_type.values())
#     total_pred = total_tp + total_fp
#     total_gold = total_tp + total_fn
#     partial_p = (total_tp + total_partial) / total_pred if total_pred > 0 else 0.0
#     partial_r = (total_tp + total_partial) / total_gold if total_gold > 0 else 0.0
#     partial_f1 = 2 * partial_p * partial_r / (partial_p + partial_r) if (partial_p + partial_r) > 0 else 0.0

#     # ── Affichage ──────────────────────────────────────────

#     print("=" * 45)
#     print("Global NER Performance (Exact Match)")
#     print("=" * 45)
#     global_table = PrettyTable()
#     global_table.field_names = ["Metric", "Value"]
#     global_table.add_row(["Precision", f"{global_p:.4f}"])
#     global_table.add_row(["Recall",    f"{global_r:.4f}"])
#     global_table.add_row(["F1-Score",  f"{global_f1:.4f}"])
#     print(global_table)

#     print("\nGlobal NER Performance (Partial Match)")
#     partial_table = PrettyTable()
#     partial_table.field_names = ["Metric", "Value"]
#     partial_table.add_row(["Precision", f"{partial_p:.4f}"])
#     partial_table.add_row(["Recall",    f"{partial_r:.4f}"])
#     partial_table.add_row(["F1-Score",  f"{partial_f1:.4f}"])
#     print(partial_table)

#     print("\nPerformance by Tag (Exact Match + Support)")
#     tag_table = PrettyTable()
#     tag_table.field_names = ["Tag", "Precision", "Recall", "F1-Score", "Support", "Partial TP"]
#     all_labels = sorted(set(list(tp_per_type.keys()) + list(support_per_type.keys())))
#     for label in all_labels:
#         p, r, f = compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])
#         tag_table.add_row([
#             label,
#             f"{p:.4f}",
#             f"{r:.4f}",
#             f"{f:.4f}",
#             support_per_type[label],
#             partial_tp_per_type[label]
#         ])
#     print(tag_table)

#     return {
#         "exact":   {"precision": global_p,  "recall": global_r,  "f1": global_f1},
#         "partial": {"precision": partial_p, "recall": partial_r, "f1": partial_f1},
#         "per_type": {
#             label: {
#                 "precision": compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])[0],
#                 "recall":    compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])[1],
#                 "f1":        compute_metrics(tp_per_type[label], fp_per_type[label], fn_per_type[label])[2],
#                 "support":   support_per_type[label],
#                 "partial_tp": partial_tp_per_type[label]
#             }
#             for label in all_labels
#         }
#     }