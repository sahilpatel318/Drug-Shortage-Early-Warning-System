"""
model.py
Interpretable forward-window risk model.

Headline model: L2-regularised logistic regression on standardised features.
Chosen for interpretability: every risk score decomposes into per-feature
contributions (coef * standardised value), so each flag is explainable by its top
drivers. No opaque deep model is added for show.

Secondary (reported only in validation for comparison): gradient-boosted trees,
with feature importances. It is NOT the headline and never overrides the LR
explanation.

Probabilities are used as-is from the logistic model (no class reweighting) so the
scores stay calibrated; imbalance is handled at the decision threshold and via
PR-based metrics, not by distorting the probabilities. Calibration is then
reported honestly via a reliability curve (see metrics.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_COLUMNS
from .util import log


class RiskModel:
    def __init__(self, seed: int):
        self.seed = seed
        self.features = list(FEATURE_COLUMNS)
        self.mean_ = None
        self.std_ = None
        self.lr = None
        self.gbt = None

    # ------------------------------------------------------------------ #
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, train_df: pd.DataFrame) -> None:
        X = train_df[self.features].to_numpy(dtype=float)
        y = train_df["label"].to_numpy(dtype=int)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0  # guard constant columns
        self.std_ = std
        Xs = self._standardize(X)

        # L2 is the solver default; we do not pass penalty= explicitly because
        # scikit-learn 1.8 deprecated that argument. C=1.0 keeps regularisation.
        self.lr = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=self.seed,
        )
        self.lr.fit(Xs, y)

        self.gbt = GradientBoostingClassifier(random_state=self.seed)
        self.gbt.fit(Xs, y)
        log(f"model fit: LR + GBT on {len(train_df)} train rows, "
            f"{int(y.sum())} positives")

    # ------------------------------------------------------------------ #
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        Xs = self._standardize(df[self.features].to_numpy(dtype=float))
        return self.lr.predict_proba(Xs)[:, 1]

    def predict_proba_gbt(self, df: pd.DataFrame) -> np.ndarray:
        Xs = self._standardize(df[self.features].to_numpy(dtype=float))
        return self.gbt.predict_proba(Xs)[:, 1]

    # ------------------------------------------------------------------ #
    def coefficients(self) -> dict[str, float]:
        return {f: float(c) for f, c in zip(self.features, self.lr.coef_[0])}

    def gbt_importances(self) -> dict[str, float]:
        return {f: float(i)
                for f, i in zip(self.features, self.gbt.feature_importances_)}

    def contributions(self, row: pd.Series, top_k: int = 4) -> list[dict]:
        """Per-row explanation: contribution_i = coef_i * standardised_value_i.
        Positive contributions push risk UP. Returns top_k by absolute value."""
        contribs = []
        for i, f in enumerate(self.features):
            z = (float(row[f]) - self.mean_[i]) / self.std_[i]
            c = float(self.lr.coef_[0][i]) * z
            contribs.append({
                "feature": f,
                "value": float(row[f]),
                "contribution": round(c, 4),
                "direction": "raises" if c >= 0 else "lowers",
            })
        contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return contribs[:top_k]
