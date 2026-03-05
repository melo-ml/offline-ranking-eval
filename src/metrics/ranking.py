import math

import numpy as np


def recall_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    top_k = ranked_items[:k]
    hits = len(set(top_k) & relevant_items)
    return float(hits / len(relevant_items))


def dcg_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    score = 0.0
    for idx, item_id in enumerate(ranked_items[:k], start=1):
        rel = 1.0 if item_id in relevant_items else 0.0
        score += rel / math.log2(idx + 1)
    return score


def ndcg_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    ideal_hits = min(len(relevant_items), k)
    ideal_dcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_items, relevant_items, k) / ideal_dcg


def average_precision_at_k(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for idx, item_id in enumerate(ranked_items[:k], start=1):
        if item_id in relevant_items:
            hits += 1
            precision_sum += hits / idx

    if hits == 0:
        return 0.0
    return precision_sum / min(len(relevant_items), k)


def auc_from_positive_score(positive_score: float, negative_scores: np.ndarray) -> float:
    if negative_scores.size == 0:
        return 1.0

    wins = np.sum(positive_score > negative_scores)
    ties = np.sum(positive_score == negative_scores)
    return float((wins + 0.5 * ties) / negative_scores.size)
