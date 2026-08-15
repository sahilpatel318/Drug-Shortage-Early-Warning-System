"""
synthetic.py
Deterministic synthetic data-generating process (DGP) used ONLY as a labeled
fallback when the real APIs are unreachable (e.g. no network, or the sandbox this
was built in). It emits raw "reports" in the SAME canonical schema as the real
ingestors so that everything downstream is source-agnostic.

Honesty contract:
  * Every record produced here carries source="synthetic". It is never presented
    as observed real-world data. DATA_LINEAGE.md and the dashboard say so loudly.
  * The DGP exposes only OBSERVABLE reports (shortage/discontinuation events with
    dates and a manufacturer). Latent fragility / hazard variables are NOT written
    out, so the feature builder has to recover signal from report history alone,
    exactly as it would from the real openFDA / DSC feeds.
  * Good metrics on synthetic data demonstrate that the PIPELINE recovers planted
    signal. They say NOTHING about real-world performance. This is stated
    everywhere it could be misread.

The DGP plants a genuine leading signal: prior shortages, recency, single-supplier
dependence and (crucially) discontinuation notices raise a drug's forward hazard,
so risk is elevated in the months BEFORE an onset. That is what makes a non-trivial
lead time recoverable, honestly, from point-in-time features.

A disjoint set of "holdout" drugs is written to data/holdout/ as sealed
ground truth. Working drugs go to data/raw/.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np

from .util import log, write_jsonl, utc_now_iso

CATEGORIES = [
    ("Oncology", 0.9),
    ("Anesthetics", 0.7),
    ("Cardiovascular", 0.2),
    ("Anti-infectives", 0.4),
    ("CNS", 0.1),
    ("Electrolytes/Nutrition", 0.6),
    ("Endocrine", 0.0),
    ("Respiratory", -0.2),
]

# DGP weights (logit space). Documented, not hidden. Tuned so the forward-90d
# onset base rate lands in a realistic, genuinely IMBALANCED single-digit range.
W_INTERCEPT = -4.85
W_CATEGORY = 0.6         # scales the per-category logits below
W_SINGLE_SUPPLIER = 0.80
W_PRIOR = 0.35          # per prior shortage, capped at 3
W_RECENT_DISC = 1.05    # discontinuation notice in last 120d
W_FRAGILITY = 0.60      # latent per-drug fragility (std normal)
DURATION_LOG_MEAN = 4.6  # ln(days) ~ exp(4.6) ~ 100 days median shortage
DURATION_LOG_STD = 0.6


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _month_ends(start: date, end: date) -> list[date]:
    """First-of-month grid stepping monthly (we sample onsets within each month)."""
    outs = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        outs.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return outs


def _make_drugs(rng: np.random.RandomState, n: int, id_prefix: str) -> list[dict]:
    drugs = []
    for i in range(n):
        cat, cat_logit = CATEGORIES[rng.randint(len(CATEGORIES))]
        # ~32% single-supplier
        n_mfr = int(rng.choice([1, 2, 3, 4, 5], p=[0.32, 0.30, 0.20, 0.12, 0.06]))
        mfrs = [f"MFR{rng.randint(1, 60):03d}" for _ in range(n_mfr)]
        mfrs = sorted(set(mfrs)) or ["MFR001"]
        drugs.append({
            "drug_id": f"{id_prefix}{i:04d}",
            "drug_name": f"{cat.split('/')[0].lower()}-agent-{i:04d}",
            "therapeutic_category": cat,
            "_cat_logit": cat_logit,
            "_manufacturers": mfrs,
            "_single_supplier": len(mfrs) == 1,
            "_fragility": float(rng.normal(0.0, 1.0)),
        })
    return sorted(drugs, key=lambda d: d["drug_id"])


def _simulate(rng: np.random.RandomState, drugs: list[dict],
              start: date, end: date) -> list[dict]:
    """Walk the monthly grid; emit shortage + discontinuation reports."""
    reports: list[dict] = []
    grid = _month_ends(start, end)
    for drug in drugs:
        did = drug["drug_id"]
        mfrs = drug["_manufacturers"]
        prior_count = 0
        last_disc_date: date | None = None
        in_shortage_until: date | None = None

        for month_start in grid:
            # skip months while a shortage is still active
            if in_shortage_until is not None and month_start <= in_shortage_until:
                continue

            recent_disc = (
                last_disc_date is not None
                and (month_start - last_disc_date).days <= 120
            )

            # Discontinuation notices arrive stochastically, more often for
            # single-supplier / fragile drugs. They RAISE future hazard, which is
            # what a point-in-time feature can pick up on.
            disc_p = _logistic(
                -4.2
                + 0.9 * drug["_single_supplier"]
                + 0.7 * drug["_fragility"]
            )
            if rng.random_sample() < disc_p:
                dday = month_start + timedelta(days=int(rng.randint(0, 27)))
                mfr = mfrs[rng.randint(len(mfrs))]
                reports.append({
                    "source": "synthetic",
                    "drug_id": did,
                    "drug_name": drug["drug_name"],
                    "therapeutic_category": drug["therapeutic_category"],
                    "manufacturer": mfr,
                    "report_type": "discontinuation",
                    "status": "Discontinued",
                    "start_date": dday.isoformat(),
                    "end_date": None,
                })
                last_disc_date = dday
                recent_disc = True

            # Shortage onset hazard (per month)
            logit = (
                W_INTERCEPT
                + W_CATEGORY * drug["_cat_logit"]
                + W_SINGLE_SUPPLIER * drug["_single_supplier"]
                + W_PRIOR * min(prior_count, 3)
                + W_RECENT_DISC * (1.0 if recent_disc else 0.0)
                + W_FRAGILITY * drug["_fragility"]
            )
            p = _logistic(logit)
            if rng.random_sample() < p:
                sday = month_start + timedelta(days=int(rng.randint(0, 27)))
                dur = int(np.clip(np.exp(rng.normal(DURATION_LOG_MEAN,
                                                    DURATION_LOG_STD)), 7, 900))
                eday = sday + timedelta(days=dur)
                mfr = mfrs[rng.randint(len(mfrs))]
                reports.append({
                    "source": "synthetic",
                    "drug_id": did,
                    "drug_name": drug["drug_name"],
                    "therapeutic_category": drug["therapeutic_category"],
                    "manufacturer": mfr,
                    "report_type": "shortage",
                    "status": "Resolved" if eday <= end else "Current",
                    "start_date": sday.isoformat(),
                    "end_date": eday.isoformat() if eday <= end else None,
                })
                prior_count += 1
                in_shortage_until = eday

    # Stable ordering: by drug then date then type.
    reports.sort(key=lambda r: (r["drug_id"], r["start_date"], r["report_type"]))
    return reports


def generate(cfg) -> dict:
    """Generate working + sealed-holdout synthetic reports; write to disk.
    Returns a small manifest for the lineage doc."""
    rng = np.random.RandomState(cfg.seed)
    start = date.fromisoformat(cfg.synthetic_start)
    end = date.fromisoformat(cfg.synthetic_end)

    working_drugs = _make_drugs(rng, cfg.synthetic_n_working_drugs, "WRK")
    holdout_drugs = _make_drugs(rng, cfg.synthetic_n_holdout_drugs, "HLD")

    working_reports = _simulate(rng, working_drugs, start, end)
    holdout_reports = _simulate(rng, holdout_drugs, start, end)

    os.makedirs(cfg.raw_dir, exist_ok=True)
    os.makedirs(cfg.holdout_dir, exist_ok=True)
    raw_path = os.path.join(cfg.raw_dir, "reports_synthetic.jsonl")
    hold_path = os.path.join(cfg.holdout_dir, "holdout_reports.jsonl")
    write_jsonl(raw_path, working_reports)
    write_jsonl(hold_path, holdout_reports)

    log(f"synthetic DGP: {len(working_reports)} working reports "
        f"({cfg.synthetic_n_working_drugs} drugs), "
        f"{len(holdout_reports)} sealed-holdout reports "
        f"({cfg.synthetic_n_holdout_drugs} drugs)")

    return {
        "source": "synthetic",
        "generated_at": utc_now_iso(),
        "seed": cfg.seed,
        "working_reports": len(working_reports),
        "holdout_reports": len(holdout_reports),
        "working_drugs": cfg.synthetic_n_working_drugs,
        "holdout_drugs": cfg.synthetic_n_holdout_drugs,
        "raw_path": raw_path,
        "holdout_path": hold_path,
        "date_range": [cfg.synthetic_start, cfg.synthetic_end],
    }
