"""
baselines.py
Two honest naive baselines. The model is only ever reported in comparison to
these. If it does not beat them, the README says so plainly.

  1) base_rate      : constant score = training-period positive rate. Its ranking
                      PR-AUC equals the evaluation set's positive fraction.
  2) persistence    : rank by prior_shortage_count ("it happened before, it will
                      happen again"). A strong, cheap, interpretable baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_rate_scores(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    rate = float(train_df["label"].mean())
    return np.full(len(eval_df), rate, dtype=float)


def persistence_scores(eval_df: pd.DataFrame) -> np.ndarray:
    # Rank by prior shortage count; add recency as a deterministic tie-breaker
    # (more recent -> slightly higher) without inventing signal.
    pc = eval_df["prior_shortage_count"].to_numpy(dtype=float)
    recency = 1.0 / (1.0 + eval_df["days_since_last_shortage"].to_numpy(dtype=float))
    return pc + 1e-3 * recency
