"""Detection metrics and validation threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_detect_metrics(
    y_true: np.ndarray,  # [N]  binary
    y_score: np.ndarray,  # [N]  probability
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute per-gene detection metrics."""
    y_pred = (y_score >= threshold).astype(int)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    result: dict[str, float] = {
        "n_samples": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if n_pos > 0 and n_neg > 0:
        result["auroc"] = float(roc_auc_score(y_true, y_score))
        result["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        result["auroc"] = 0.5
        result["auprc"] = 0.0

    result["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result["tn"] = int(tn)
        result["fp"] = int(fp)
        result["fn"] = int(fn)
        result["tp"] = int(tp)

    return result


def select_detection_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "mcc",
) -> tuple[float, float]:
    """Select a score threshold by validation MCC or F1 in O(N log N)."""
    if metric not in {"mcc", "f1"}:
        raise ValueError("threshold metric must be 'mcc' or 'f1'")
    if y_true.shape != y_score.shape or y_true.size == 0:
        raise ValueError("threshold selection requires matching non-empty arrays")
    truth = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if not np.isfinite(scores).all():
        raise ValueError("threshold selection requires finite scores")
    baseline_pred = scores >= 0.5
    if metric == "mcc":
        baseline_score = float(matthews_corrcoef(truth, baseline_pred))
    else:
        baseline_score = float(f1_score(truth, baseline_pred, zero_division=0))
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, truth.size]
    group_truth = np.add.reduceat(sorted_truth, boundaries[:-1])
    group_size = np.diff(boundaries)
    tp = np.cumsum(group_truth)
    fp = np.cumsum(group_size - group_truth)
    total_pos = int(truth.sum())
    total_neg = truth.size - total_pos
    fn = total_pos - tp
    tn = total_neg - fp
    if metric == "mcc":
        denominator = np.sqrt(
            (tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        values = np.divide(
            tp * tn - fp * fn,
            denominator,
            out=np.zeros_like(tp, dtype=float),
            where=denominator != 0,
        )
    else:
        denominator = 2 * tp + fp + fn
        values = np.divide(
            2 * tp,
            denominator,
            out=np.zeros_like(tp, dtype=float),
            where=denominator != 0,
        )
    candidate_scores = np.r_[values, 0.0]
    candidate_thresholds = np.r_[
        sorted_scores[boundaries[:-1]],
        np.nextafter(sorted_scores[-1], np.inf),
    ]
    improved = candidate_scores > baseline_score
    if not improved.any():
        return 0.5, baseline_score
    best_score = float(candidate_scores[improved].max())
    best_threshold = float(
        candidate_thresholds[improved & (candidate_scores == best_score)].min()
    )
    return best_threshold, best_score
