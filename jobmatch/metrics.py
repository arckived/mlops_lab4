"""Ranking metrics. Labels are 1 (would apply) or 0 (not for me), in ranked order."""

import numpy as np


def precision_at_k(labels_in_rank_order, k: int) -> float:
    top = list(labels_in_rank_order)[:k]
    return float(np.mean(top)) if top else 0.0


def ndcg_at_k(labels_in_rank_order, k: int) -> float:
    """1.0 means every 'would apply' job is ranked above every 'not for me' job."""
    labels = np.asarray(list(labels_in_rank_order), dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, min(k, len(labels)) + 2))
    dcg = float(np.sum(labels[:k] * discounts))
    ideal = np.sort(labels)[::-1]
    idcg = float(np.sum(ideal[:k] * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(labels_in_rank_order) -> float:
    labels = list(labels_in_rank_order)
    hits, total = 0, 0.0
    for i, rel in enumerate(labels, start=1):
        if rel:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0
