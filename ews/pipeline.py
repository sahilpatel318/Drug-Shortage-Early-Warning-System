"""
Pipeline orchestrator.

Runs the whole synthetic early warning study end to end, deterministically, and
writes artifacts/report.json. Two runs with the same seed produce byte
identical output. Every number written carries its evaluation N.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn

from . import config, evaluate, model
from .features import build_features, FEATURE_COLUMNS, NUMERIC_FEATURES
from .firewall import Vault, leakage_test
from .synth import generate


def _feature_builder(panel, vault):
    return build_features(panel, vault)


def run(seed: int = config.SEED) -> dict:
    panel = generate(seed)
    vault = Vault(panel.onset)

    # Prove the firewall before trusting any metric.
    leak = leakage_test(_feature_builder, panel, vault, seed=seed)

    feats = build_features(panel, vault)
    train = feats[feats["anchor_month"] < config.SPLIT_MONTH].reset_index(drop=True)
    test = feats[feats["anchor_month"] >= config.SPLIT_MONTH].reset_index(drop=True)

    # Model A: early warning
    ews = model.EarlyWarningModel(seed=seed).fit(train)
    train_scores = ews.score(train)
    test_scores = ews.score(test)
    test = test.copy()
    test["ews"] = test_scores

    y_train = train["y"].to_numpy()
    y_test = test["y"].to_numpy()

    thr = model.choose_threshold(train_scores, y_train)

    # Baselines on the SAME test rows
    prev = model.prevalence_baseline(train, test)
    rand = model.random_baseline(test, seed=seed)
    single_score, single_name = model.best_single_feature(train, test)

    pr = {
        "logistic_regression": round(evaluate.pr_auc(y_test, test_scores), 6),
        "best_single_feature": round(evaluate.pr_auc(y_test, single_score), 6),
        "prevalence": round(evaluate.pr_auc(y_test, prev), 6),
        "random": round(evaluate.pr_auc(y_test, rand), 6),
    }
    roc = {
        "logistic_regression": round(evaluate.roc_auc(y_test, test_scores), 6),
        "best_single_feature": round(evaluate.roc_auc(y_test, single_score), 6),
    }

    clf_metrics = evaluate.classification_at_threshold(y_test, test_scores, thr)
    lead = evaluate.lead_time(test, vault, thr)
    survival = evaluate.km_survival_by_risk(test, vault)
    watchlist = evaluate.top_at_risk(test, panel, thr)
    coefs = ews.coefficients()

    report = {
        "meta": {
            "project": "Drug Shortage Early Warning System",
            "provenance": "SYNTHETIC",
            "provenance_detail": (
                "Data is synthetically generated with a planted supply chain "
                "hazard. Metrics are signal recovery on synthetic data, not a "
                "claim about real world shortages. Real source adapters for "
                "openFDA and Drug Shortages Canada are included for a networked "
                "run."
            ),
            "seed": seed,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "config": {
            "n_drugs": config.N_DRUGS,
            "period_months": config.PERIOD_MONTHS,
            "split_month": config.SPLIT_MONTH,
            "lead_horizon_months": config.LEAD_HORIZON,
            "min_history_months": config.MIN_HISTORY,
            "warn_lookback_months": config.WARN_LOOKBACK,
            "categories": list(config.CATEGORIES.keys()),
            "features": FEATURE_COLUMNS,
        },
        "firewall": leak,
        "dataset": {
            "train_rows": int(len(train)),
            "train_positives": int(y_train.sum()),
            "train_prevalence": round(float(y_train.mean()), 6),
            "test_rows": int(len(test)),
            "test_positives": int(y_test.sum()),
            "test_prevalence": round(float(y_test.mean()), 6),
        },
        "pr_auc": pr,
        "pr_auc_lift_over_prevalence": round(
            pr["logistic_regression"] - pr["prevalence"], 6
        ),
        "roc_auc": roc,
        "best_single_feature_name": single_name,
        "classification_at_threshold": clf_metrics,
        "lead_time": lead,
        "survival": survival,
        "coefficients": coefs,
        "watchlist": watchlist,
        "pr_curves": _pr_curves(y_test, {
            "logistic_regression": test_scores,
            "best_single_feature": single_score,
            "prevalence": prev,
        }),
    }
    report["determinism_hash"] = _hash(report)
    return report


def _pr_curves(y: np.ndarray, score_map: dict, points: int = 40) -> dict:
    """Sampled precision recall curves for plotting, deterministic length."""
    from sklearn.metrics import precision_recall_curve
    out = {}
    for name, s in score_map.items():
        precision, recall, _ = precision_recall_curve(y, s)
        # Sample to a fixed number of points along recall for a stable artifact.
        idx = np.linspace(0, len(recall) - 1, min(points, len(recall))).astype(int)
        out[name] = [
            {"recall": round(float(recall[i]), 6), "precision": round(float(precision[i]), 6)}
            for i in idx
        ]
    return out


def _hash(report: dict) -> str:
    clone = {k: v for k, v in report.items() if k != "determinism_hash"}
    # Exclude volatile meta timestamp from the reproducibility hash.
    clone = json.loads(json.dumps(clone, sort_keys=True))
    clone.get("meta", {}).pop("generated_at_utc", None)
    blob = json.dumps(clone, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_report(report: dict, path=None) -> str:
    path = path or (config.ARTIFACT_DIR / "report.json")
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, indent=2)
        fh.write("\n")
    return str(path)
