"""
Central configuration for the Drug Shortage Early Warning System.

Every tunable that affects output lives here so a single fixed seed makes the
whole pipeline byte reproducible. Nothing here reads the network.
"""
from __future__ import annotations

from pathlib import Path

# Repository layout
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# Master seed. Change this and the entire synthetic history changes, but any
# given seed reproduces byte for byte.
SEED = 20260813

# Synthetic panel shape
N_DRUGS = 240
PERIOD_MONTHS = 96          # eight years of monthly observations
FIRST_YEAR = 2018           # calendar anchor for readable dates

# Temporal split: train on the earlier window, test on the later window.
# This is a hard out of time split so no future information leaks backward.
SPLIT_MONTH = 66            # rows anchored before this month train, at or after test

# Early warning task
LEAD_HORIZON = 6            # predict a shortage onset within the next N months
MIN_HISTORY = 12           # require this many months of history before scoring
WARN_LOOKBACK = 12         # one year operational watch window for lead-time

# Therapeutic categories and their baseline shortage pressure. These weights
# drive the synthetic hazard, they are not observed by the model as truth.
CATEGORIES = {
    "Oncology":        0.85,
    "Anti-infective":  0.70,
    "Cardiovascular":  0.45,
    "CNS":             0.40,
    "Endocrine":       0.35,
    "Analgesic":       0.55,
    "Respiratory":     0.30,
    "Immunology":      0.25,
}

# Hazard model coefficients used ONLY to synthesize the ground truth.
# The learner never sees these. They exist so the planted signal is real and
# recoverable, which is the honest claim: signal recovery on synthetic data.
HAZARD = {
    "intercept":        -6.20,
    "single_supplier":   1.10,   # single source dependence, the headline risk
    "hhi":               1.30,   # manufacturer concentration in [0,1]
    "log_past":          0.50,   # applied to log1p(past onset count)
    "category":          0.85,   # applied to the category weight above
    "supply_shock":      0.60,   # a supplier exit this month
    "lost_12m":          0.45,   # trailing 12 month supplier attrition
    "recency":           0.40,   # elevated risk shortly after a prior recovery
    "noise_sd":          0.60,   # gaussian logit noise, keeps the signal imperfect
}

# Supplier attrition dynamics. Mild auto correlation: a recent loss modestly
# raises the odds of another, a realistic supply chain drift.
ATTRITION = {
    "base_loss":       0.015,
    "spiral_loss":     0.030,   # loss prob if a loss occurred in the last 6 months
    "gain":            0.010,
    "spiral_window":   6,
}

# Shortage duration in months (drawn per onset, inclusive of the onset month)
SHORTAGE_DURATION = {"min": 2, "max": 9, "mode": 4}

# Operating threshold selection: pick the score cutoff on the TRAIN set that
# maximizes F1, then apply it unchanged to the test set. Reported, never tuned
# on test.
THRESHOLD_OBJECTIVE = "f1"

# Rounding for the serialized report so two runs are byte identical.
FLOAT_DP = 6

# Real source adapters. In sandbox these are unreachable, so the pipeline falls
# back to synthetic and labels it loudly. On an internet connected machine,
# run.py --source real will attempt these.
REAL_SOURCES = {
    "drug_shortages_canada": "https://www.drugshortagescanada.ca/api/v1/search",
    "openfda_shortages":     "https://api.fda.gov/drug/shortages.json",
}
