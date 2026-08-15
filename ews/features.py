"""
Temporal feature engineering.

Every feature for a row anchored at month t is a function of the OBSERVABLE
panel truncated at t (supplier counts, concentration, and past shortages that
already happened and are therefore public record). The label comes from the
sealed Vault and describes a window strictly after t. Features never read the
Vault, which is what the leakage test verifies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .firewall import Vault

CATEGORY_LIST = list(config.CATEGORIES.keys())

NUMERIC_FEATURES = [
    "suppliers",
    "single_supplier",
    "hhi",
    "log_past_onsets",
    "months_since_last_onset",
    "supply_shock_6m",
    "suppliers_lost_12m",
    "supplier_trend_6m",
    "category_recent_rate",
]
CATEGORY_FEATURES = [f"cat_{c}" for c in CATEGORY_LIST]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORY_FEATURES


def _observed_onsets(in_shortage: np.ndarray) -> np.ndarray:
    """Onset months are transitions from not in shortage to in shortage.

    Derived from the observable in_shortage series, so this is public history,
    not a read of the sealed answer key.
    """
    onset = np.zeros_like(in_shortage)
    onset[0] = in_shortage[0]
    onset[1:] = ((in_shortage[1:] == 1) & (in_shortage[:-1] == 0)).astype(in_shortage.dtype)
    return onset


def build_features(panel, vault: Vault) -> pd.DataFrame:
    """Return one row per (drug, anchor_month) that is at risk with a fully
    observed label window."""
    frame = panel.frame
    t = config.PERIOD_MONTHS
    cats = np.array(CATEGORY_LIST)

    # Reshape observable series into (n_drugs, months) arrays, ordered by the
    # zero padded drug id so row index equals the vault drug index.
    frame = frame.sort_values(["drug_id", "month"], kind="mergesort")
    drug_ids = list(dict.fromkeys(frame["drug_id"].tolist()))
    idx_of = {d: int(d[1:]) for d in drug_ids}
    n = len(drug_ids)

    suppliers = frame["suppliers"].to_numpy().reshape(n, t)
    hhi = frame["hhi"].to_numpy().reshape(n, t)
    shock = frame["supply_shock"].to_numpy().reshape(n, t)
    in_short = frame["in_shortage"].to_numpy().reshape(n, t)
    cat_per_drug = frame.groupby("drug_id", sort=True)["category"].first().to_numpy()

    onset_obs = np.vstack([_observed_onsets(in_short[i]) for i in range(n)])

    # Cumulative past onset counts strictly before each month.
    past_cum = np.cumsum(onset_obs, axis=1)
    past_before = np.zeros_like(past_cum)
    past_before[:, 1:] = past_cum[:, :-1]

    # Months since last onset, strictly before the anchor.
    months_since = np.full((n, t), config.PERIOD_MONTHS, dtype=np.int32)
    for i in range(n):
        last = -1
        for m in range(t):
            months_since[i, m] = (m - last) if last >= 0 else config.PERIOD_MONTHS
            if onset_obs[i, m] == 1:
                last = m

    # Trailing supplier shock in the last 6 months, inclusive of the anchor.
    shock6 = np.zeros((n, t), dtype=np.int32)
    lost12 = np.zeros((n, t), dtype=np.int32)
    for m in range(t):
        lo = max(0, m - 5)
        shock6[:, m] = (shock[:, lo:m + 1].sum(axis=1) > 0).astype(np.int32)
        lo12 = max(0, m - 11)
        lost12[:, m] = shock[:, lo12:m + 1].sum(axis=1).astype(np.int32)

    # Net change in supplier count over the trailing 6 months (negative means
    # the drug is losing suppliers, which is the deterioration signal).
    trend6 = np.zeros((n, t), dtype=np.int32)
    for m in range(t):
        ref = max(0, m - 6)
        trend6[:, m] = suppliers[:, m] - suppliers[:, ref]

    # Category recent onset rate, trailing 12 months strictly before anchor.
    cat_rate = _category_recent_rate(onset_obs, in_short, cat_per_drug, cats)

    rows = []
    horizon = config.LEAD_HORIZON
    for i in range(n):
        di = idx_of[drug_ids[i]]
        cat = cat_per_drug[i]
        cat_onehot = {f"cat_{c}": (1 if c == cat else 0) for c in CATEGORY_LIST}
        for m in range(config.MIN_HISTORY, t):
            if m + horizon >= t:
                continue                      # label window not fully observed
            if in_short[i, m] == 1:
                continue                      # not at risk while already short
            y = vault.label_window(di, m, horizon)
            row = {
                "drug_idx": di,
                "anchor_month": m,
                "suppliers": int(suppliers[i, m]),
                "single_supplier": int(suppliers[i, m] == 1),
                "hhi": round(float(hhi[i, m]), 6),
                "log_past_onsets": round(float(np.log1p(past_before[i, m])), 6),
                "months_since_last_onset": int(min(months_since[i, m], config.PERIOD_MONTHS)),
                "supply_shock_6m": int(shock6[i, m]),
                "suppliers_lost_12m": int(lost12[i, m]),
                "supplier_trend_6m": int(trend6[i, m]),
                "category_recent_rate": round(float(cat_rate[i, m]), 6),
                "y": int(y),
            }
            row.update(cat_onehot)
            rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["anchor_month", "drug_idx"], kind="mergesort").reset_index(drop=True)
    ordered = ["drug_idx", "anchor_month"] + FEATURE_COLUMNS + ["y"]
    return out[ordered]


def _category_recent_rate(onset_obs, in_short, cat_per_drug, cats) -> np.ndarray:
    n, t = onset_obs.shape
    rate = np.zeros((n, t), dtype=np.float64)
    # Per category, per month sums of onsets and at risk drugs.
    for c in cats:
        mask = (cat_per_drug == c)
        if not mask.any():
            continue
        onsets_m = onset_obs[mask].sum(axis=0).astype(np.float64)      # length t
        atrisk_m = (in_short[mask] == 0).sum(axis=0).astype(np.float64)
        con = np.cumsum(onsets_m)
        car = np.cumsum(atrisk_m)
        for m in range(t):
            lo = m - 12
            num = con[m - 1] - (con[lo - 1] if lo - 1 >= 0 else 0.0) if m >= 1 else 0.0
            den = car[m - 1] - (car[lo - 1] if lo - 1 >= 0 else 0.0) if m >= 1 else 0.0
            val = (num / den) if den > 0 else 0.0
            rate[np.flatnonzero(mask), m] = val
    return rate
