"""
metrics.py
Imbalance-aware metrics. Accuracy is deliberately NOT a headline metric.

Reported: PR-AUC (average precision), precision, recall, F1, confusion matrix at a
chosen threshold, a downsampled PR curve, a reliability (calibration) curve, and
the Brier score. ROC-AUC is included only as a secondary reference.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray,
                     policy: str, target_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns thresholds of length n-1
    if len(thresholds) == 0:
        return 0.5
    if policy == "target_recall":
        # smallest threshold achieving recall >= target (recall decreases as
        # threshold rises); walk from high recall down.
        best = thresholds[0]
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
            if r >= target_recall:
                best = t
        return float(best)
    # default: max F1
    f1s = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        f1s.append((f1, t))
    f1s.sort(key=lambda x: (x[0], -x[1]))  # max f1, tie -> lower threshold
    return float(f1s[-1][1])


def confusion_at(y_true: np.ndarray, scores: np.ndarray,
                 threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "threshold": round(float(threshold), 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def pr_curve_points(y_true: np.ndarray, scores: np.ndarray,
                    n: int = 60) -> list[dict]:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    # downsample deterministically to n points along recall
    idx = np.linspace(0, len(recall) - 1, num=min(n, len(recall))).astype(int)
    return [{"recall": round(float(recall[i]), 6),
             "precision": round(float(precision[i]), 6)} for i in idx]


def calibration_points(y_true: np.ndarray, scores: np.ndarray,
                       bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    pts = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (scores >= lo) & (scores < hi if b < bins - 1 else scores <= hi)
        if mask.sum() == 0:
            continue
        pts.append({
            "bin_lo": round(float(lo), 4),
            "bin_hi": round(float(hi), 4),
            "mean_pred": round(float(scores[mask].mean()), 6),
            "obs_freq": round(float(y_true[mask].mean()), 6),
            "count": int(mask.sum()),
        })
    return pts


def score_block(y_true: np.ndarray, scores: np.ndarray,
                threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    out = {
        "pr_auc": round(float(average_precision_score(y_true, scores)), 6),
        "base_rate": round(float(y_true.mean()), 6),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "brier": round(float(brier_score_loss(y_true, np.clip(scores, 0, 1))), 6),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 6)
    except ValueError:
        out["roc_auc"] = None
    out["confusion"] = confusion_at(y_true, scores, threshold)
    return out
