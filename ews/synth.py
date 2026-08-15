"""
Synthetic supply history generator.

This builds a monthly panel of drugs with realistic supply chain structure
(supplier counts, manufacturer concentration, therapeutic category) and draws
shortage onsets from a hazard that depends on that structure plus noise.

IMPORTANT HONESTY NOTE
The relationship between features and shortages here is planted by us. Any
score the model earns on this data is signal recovery, not a claim about real
world shortages. Every artifact and UI surface labels this data SYNTHETIC.
The real source adapters (Drug Shortages Canada, openFDA) live in sources.py
and produce the same panel schema when run with internet access.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Panel:
    """The observable panel plus the sealed onset labels, kept separate."""
    frame: pd.DataFrame          # observable columns only
    onset: np.ndarray            # shape (n_drugs, n_months), 1 at shortage start
    drug_ids: list
    categories: list


def generate(seed: int = config.SEED) -> Panel:
    """Generate a deterministic synthetic panel. Same seed, same bytes."""
    rng = np.random.RandomState(seed)
    n = config.N_DRUGS
    t = config.PERIOD_MONTHS
    cat_names = list(config.CATEGORIES.keys())
    cat_weights = np.array([config.CATEGORIES[c] for c in cat_names])

    # Assign each drug a category (skewed so common categories dominate).
    cat_probs = np.array([0.10, 0.14, 0.18, 0.14, 0.12, 0.12, 0.11, 0.09])
    cat_probs = cat_probs / cat_probs.sum()
    drug_cat_idx = rng.choice(len(cat_names), size=n, p=cat_probs)

    drug_ids = ["D%04d" % i for i in range(n)]

    # Static starting supplier structure per drug.
    # Number of suppliers is small and right skewed; single supplier drugs exist.
    start_suppliers = rng.choice(
        [1, 2, 3, 4, 5, 6], size=n, p=[0.16, 0.24, 0.22, 0.18, 0.12, 0.08]
    )

    # Time varying state arrays
    suppliers = np.zeros((n, t), dtype=np.int32)
    hhi = np.zeros((n, t), dtype=np.float64)
    in_shortage = np.zeros((n, t), dtype=np.int32)
    onset = np.zeros((n, t), dtype=np.int32)
    supply_shock = np.zeros((n, t), dtype=np.int32)     # lost a supplier this month
    months_since_recovery = np.full((n, t), 999, dtype=np.int32)

    # Per drug running state
    cur_suppliers = start_suppliers.copy()
    shortage_left = np.zeros(n, dtype=np.int32)          # months remaining in shortage
    past_onsets = np.zeros(n, dtype=np.int32)
    last_recovery_month = np.full(n, -999, dtype=np.int32)

    cat_w = cat_weights[drug_cat_idx]

    def draw_shares(k: int) -> np.ndarray:
        # Dirichlet market shares; lower alpha means more concentrated market.
        alpha = np.full(k, 0.9)
        return rng.dirichlet(alpha)

    win = config.ATTRITION["spiral_window"]
    for month in range(t):
        # Supplier attrition or entry. Attrition is auto correlated: a recent
        # loss raises the odds of another, producing a deterioration spiral.
        recent_loss = np.zeros(n, dtype=bool)
        if month > 0:
            lo = max(0, month - win)
            recent_loss = supply_shock[:, lo:month].sum(axis=1) > 0
        for i in range(n):
            shock = 0
            loss_p = config.ATTRITION["spiral_loss"] if recent_loss[i] \
                else config.ATTRITION["base_loss"]
            if cur_suppliers[i] > 1 and rng.rand() < loss_p:
                cur_suppliers[i] -= 1
                shock = 1
            elif cur_suppliers[i] < 6 and rng.rand() < config.ATTRITION["gain"]:
                cur_suppliers[i] += 1
            supply_shock[i, month] = shock

        # Concentration for this month (fewer suppliers means higher hhi).
        for i in range(n):
            shares = draw_shares(int(cur_suppliers[i]))
            hhi[i, month] = float(np.sum(shares ** 2))     # in (0,1], 1 = monopoly
        suppliers[:, month] = cur_suppliers

        # Trailing 12 month supplier attrition count (the observable ramp).
        lo12 = max(0, month - 11)
        lost_12m = supply_shock[:, lo12:month + 1].sum(axis=1).astype(np.float64)

        single = (cur_suppliers == 1).astype(np.float64)
        recency_flag = ((month - last_recovery_month) >= 0) & \
                       ((month - last_recovery_month) <= 6)
        recency_flag = recency_flag.astype(np.float64)

        # Hazard logit for onset THIS month, for drugs currently at risk.
        logit = (
            config.HAZARD["intercept"]
            + config.HAZARD["single_supplier"] * single
            + config.HAZARD["hhi"] * hhi[:, month]
            + config.HAZARD["log_past"] * np.log1p(past_onsets)
            + config.HAZARD["category"] * cat_w
            + config.HAZARD["supply_shock"] * supply_shock[:, month]
            + config.HAZARD["lost_12m"] * lost_12m
            + config.HAZARD["recency"] * recency_flag
        )
        logit = logit + rng.normal(0.0, config.HAZARD["noise_sd"], size=n)
        prob = _sigmoid(logit)

        for i in range(n):
            if shortage_left[i] > 0:
                # Currently in shortage: cannot onset, tick down.
                in_shortage[i, month] = 1
                shortage_left[i] -= 1
                if shortage_left[i] == 0:
                    last_recovery_month[i] = month
                continue
            # At risk this month
            if rng.rand() < prob[i]:
                onset[i, month] = 1
                in_shortage[i, month] = 1
                past_onsets[i] += 1
                dur = _draw_duration(rng)
                shortage_left[i] = dur - 1   # this month counts as one

        months_since_recovery[:, month] = np.where(
            last_recovery_month >= 0, month - last_recovery_month, 999
        )

    # Build the observable long frame. Note: onset is NOT included here; it is
    # sealed and returned separately so the feature path cannot read it.
    records = []
    year0 = config.FIRST_YEAR
    for i in range(n):
        for m in range(t):
            cal_year = year0 + m // 12
            cal_month = m % 12 + 1
            records.append((
                drug_ids[i],
                cat_names[drug_cat_idx[i]],
                m,
                f"{cal_year:04d}-{cal_month:02d}",
                int(suppliers[i, m]),
                round(float(hhi[i, m]), 6),
                int(in_shortage[i, m]),
                int(supply_shock[i, m]),
            ))
    frame = pd.DataFrame.from_records(
        records,
        columns=[
            "drug_id", "category", "month", "period",
            "suppliers", "hhi", "in_shortage", "supply_shock",
        ],
    ).sort_values(["drug_id", "month"], kind="mergesort").reset_index(drop=True)

    return Panel(
        frame=frame,
        onset=onset,
        drug_ids=drug_ids,
        categories=[cat_names[j] for j in drug_cat_idx],
    )


def _draw_duration(rng: np.random.RandomState) -> int:
    lo = config.SHORTAGE_DURATION["min"]
    hi = config.SHORTAGE_DURATION["max"]
    mode = config.SHORTAGE_DURATION["mode"]
    return int(round(rng.triangular(lo, mode, hi)))
