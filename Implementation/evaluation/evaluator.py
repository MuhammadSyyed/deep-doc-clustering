"""
evaluation/evaluator.py
────────────────────────
Clustering evaluation metrics.

Metrics
───────
- ACC  : Clustering accuracy via Hungarian (linear sum assignment) matching.
- NMI  : Normalised Mutual Information (arithmetic mean).
- ARI  : Adjusted Rand Index.
- Purity: Fraction of samples in their majority ground-truth class per cluster.
"""

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hungarian-matched clustering accuracy."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    k = max(y_true.max(), y_pred.max()) + 1
    cost = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cost[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    return cost[row_ind, col_ind].sum() / len(y_true)


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering purity."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total = 0
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        most_common = np.bincount(y_true[mask]).max()
        total += most_common
    return total / len(y_true)


class Evaluator:
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ["acc", "nmi", "ari", "purity"]

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        results = {}
        if "acc" in self.metrics:
            results["acc"] = round(clustering_accuracy(y_true, y_pred), 4)
        if "nmi" in self.metrics:
            results["nmi"] = round(
                normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"),
                4,
            )
        if "ari" in self.metrics:
            results["ari"] = round(adjusted_rand_score(y_true, y_pred), 4)
        if "purity" in self.metrics:
            results["purity"] = round(purity(y_true, y_pred), 4)

        results["acc"] = float(round(clustering_accuracy(y_true, y_pred), 4))
        results["nmi"] = float(round(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"), 4))
        results["ari"] = float(round(adjusted_rand_score(y_true, y_pred), 4))
        results["purity"] = float(round(purity(y_true, y_pred), 4))
        return results

    def print_results(self, results: Dict[str, float], prefix: str = ""):
        header = f"{'Metric':<12}{'Score':>8}"
        sep = "─" * 22
        lines = [sep, header, sep]
        for k, v in results.items():
            lines.append(f"{k.upper():<12}{v:>8.4f}")
        lines.append(sep)
        block = "\n".join(lines)
        if prefix:
            print(f"\n{prefix}")
        print(block)
        return block
