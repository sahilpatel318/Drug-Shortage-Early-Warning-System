"""
Early warning model and honest baselines.

Model A is a logistic regression over standardized numeric features plus
category indicators. It predicts the probability that an at risk drug enters
shortage within the next LEAD_HORIZON months. The same score ranks drugs for
the survival view. Baselines are deliberately simple so the reported lift is
attributable, not hand waved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from . import config
from .features import NUMERIC_FEATURES, CATEGORY_FEATURES


class EarlyWarningModel:
    def __init__(self, seed: int = config.SEED):
        self.seed = seed
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            solver="liblinear", C=1.0, random_state=seed, max_iter=1000
        )
        self.numeric = NUMERIC_FEATURES
        self.categorical = CATEGORY_FEATURES

    def _matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        num = df[self.numeric].to_numpy(dtype=np.float64)
        cat = df[self.categorical].to_numpy(dtype=np.float64)
        if fit:
            num_s = self.scaler.fit_transform(num)
        else:
            num_s = self.scaler.transform(num)
        return np.hstack([num_s, cat])

    def fit(self, train: pd.DataFrame) -> "EarlyWarningModel":
        x = self._matrix(train, fit=True)
        y = train["y"].to_numpy()
        self.clf.fit(x, y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = self._matrix(df, fit=False)
        return self.clf.predict_proba(x)[:, 1]

    def coefficients(self) -> list:
        """Return standardized log-odds contributions per feature, sorted by
        absolute magnitude. Numeric coefficients are comparable because inputs
        were standardized; category coefficients are per indicator."""
        names = self.numeric + self.categorical
        coefs = self.clf.coef_.ravel()
        pairs = [
            {"feature": n, "coef": round(float(c), 6), "abs": abs(float(c))}
            for n, c in zip(names, coefs)
        ]
        pairs.sort(key=lambda d: d["abs"], reverse=True)
        for p in pairs:
            del p["abs"]
        return pairs


def prevalence_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    rate = float(train["y"].mean())
    return np.full(len(test), rate)


def random_baseline(test: pd.DataFrame, seed: int = config.SEED) -> np.ndarray:
    rng = np.random.RandomState(seed + 11)
    return rng.rand(len(test))


def best_single_feature(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Pick the single numeric feature with the best train PR-AUC (either
    orientation) and return its test score plus the feature name."""
    y_train = train["y"].to_numpy()
    best_name, best_ap, best_sign = None, -1.0, 1.0
    for feat in NUMERIC_FEATURES:
        col = train[feat].to_numpy(dtype=np.float64)
        for sign in (1.0, -1.0):
            ap = average_precision_score(y_train, sign * col)
            if ap > best_ap:
                best_ap, best_name, best_sign = ap, feat, sign
    test_score = best_sign * test[best_name].to_numpy(dtype=np.float64)
    return test_score, best_name


def choose_threshold(train_scores: np.ndarray, y_train: np.ndarray) -> float:
    """Pick the probability cutoff on TRAIN that maximizes F1. Applied
    unchanged to test. Never tuned on test."""
    order = np.argsort(train_scores)
    best_thr, best_f1 = 0.5, -1.0
    candidates = np.unique(np.round(train_scores, 4))
    for thr in candidates:
        pred = (train_scores >= thr).astype(int)
        tp = int(((pred == 1) & (y_train == 1)).sum())
        fp = int(((pred == 1) & (y_train == 0)).sum())
        fn = int(((pred == 0) & (y_train == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr
