"""
backtest.py
Temporal backtest. Train on the earlier period, evaluate on the later period,
always report the model against the two naive baselines, and compute the headline
lead-time distribution.

Lead time definition (documented):
  For each real shortage onset O of drug d in the TEST period, we look back only
  over OUT-OF-SAMPLE (test-period) as_of cutoffs. The lead time is
  O - t*, where t* is the EARLIEST test as_of with score >= flag threshold such
  that there is no other onset of d between t* and O and (O - t*) <= max_lead_days.
  Onsets with no qualifying flag are counted as "not flagged in advance" (they do
  not inflate the lead-time distribution). Lead time can exceed the 90-day label
  window when the model flags risk earlier than the labelling horizon.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import baselines, metrics
from .model import RiskModel
from .util import log

MAX_LEAD_DAYS = 730


def _onsets_by_drug(reports: list[dict]) -> dict[str, list[date]]:
    out: dict[str, list[date]] = {}
    for r in reports:
        if r.get("report_type") == "shortage" and r.get("start_date"):
            out.setdefault(r["drug_id"], []).append(date.fromisoformat(r["start_date"]))
    for k in out:
        out[k].sort()
    return out


def compute_lead_times(reports: list[dict], scored_test: pd.DataFrame,
                       threshold: float, cfg) -> dict:
    split = date.fromisoformat(cfg.split_date)
    onsets = _onsets_by_drug(reports)

    # per-drug test-period scored rows: {drug: [(as_of_date, score)]}
    scored: dict[str, list[tuple[date, float]]] = {}
    for _, row in scored_test.iterrows():
        scored.setdefault(row["drug_id"], []).append(
            (date.fromisoformat(row["as_of"]), float(row["score"])))
    for k in scored:
        scored[k].sort()

    lead_days: list[int] = []
    total_onsets = 0
    flagged = 0
    for drug, dates in onsets.items():
        test_onsets = [o for o in dates if o > split]
        if not test_onsets:
            continue
        drug_scores = scored.get(drug, [])
        for O in test_onsets:
            total_onsets += 1
            # earliest test as_of before O with elevated score, attributable to O
            candidates = []
            for (t, s) in drug_scores:
                if t >= O:
                    continue
                if (O - t).days > MAX_LEAD_DAYS:
                    continue
                if s < threshold:
                    continue
                # no other onset strictly between t and O
                if any(t < other < O for other in dates):
                    continue
                candidates.append(t)
            if candidates:
                t_star = min(candidates)
                lead_days.append((O - t_star).days)
                flagged += 1

    lead = sorted(lead_days)
    if lead:
        arr = np.array(lead)
        hist_edges = [0, 15, 30, 45, 60, 90, 120, 180, 270, 365, MAX_LEAD_DAYS]
        hist = []
        for i in range(len(hist_edges) - 1):
            lo, hi = hist_edges[i], hist_edges[i + 1]
            c = int(((arr >= lo) & (arr < hi)).sum())
            hist.append({"lo": lo, "hi": hi, "count": c})
        summary = {
            "n_onsets_test": total_onsets,
            "n_flagged_in_advance": flagged,
            "detection_rate": round(flagged / total_onsets, 6) if total_onsets else 0.0,
            "median_days": int(np.median(arr)),
            "iqr_days": [int(np.percentile(arr, 25)), int(np.percentile(arr, 75))],
            "min_days": int(arr.min()),
            "max_days": int(arr.max()),
            "histogram": hist,
        }
    else:
        summary = {
            "n_onsets_test": total_onsets,
            "n_flagged_in_advance": 0,
            "detection_rate": 0.0,
            "median_days": None,
            "iqr_days": [None, None],
            "min_days": None,
            "max_days": None,
            "histogram": [],
            "note": "No onsets flagged in advance at this threshold.",
        }
    log(f"lead-time: {flagged}/{total_onsets} test onsets flagged in advance; "
        f"median={summary['median_days']} days")
    return summary


def temporal_backtest(panel: pd.DataFrame, reports: list[dict], cfg) -> dict:
    train_df = panel[panel.split == "train"].reset_index(drop=True)
    test_df = panel[panel.split == "test"].reset_index(drop=True)
    if train_df["label"].sum() == 0 or test_df["label"].sum() == 0:
        raise ValueError("Train or test set has no positive labels; adjust "
                         "split_date or data range.")

    model = RiskModel(cfg.seed)
    model.fit(train_df)

    train_scores = model.predict_proba(train_df)
    test_scores = model.predict_proba(test_df)

    threshold = metrics.choose_threshold(
        train_df["label"].to_numpy(int), train_scores,
        cfg.threshold_policy, cfg.target_recall)

    # attach scores
    scored_test = test_df.copy()
    scored_test["score"] = test_scores

    # model vs baselines on TEST
    y_test = test_df["label"].to_numpy(int)
    base_scores = baselines.base_rate_scores(train_df, test_df)
    pers_scores = baselines.persistence_scores(test_df)

    model_block = metrics.score_block(y_test, test_scores, threshold)
    base_block = metrics.score_block(y_test, base_scores, threshold)
    pers_thr = metrics.choose_threshold(
        train_df["label"].to_numpy(int),
        baselines.persistence_scores(train_df),
        cfg.threshold_policy, cfg.target_recall)
    pers_block = metrics.score_block(y_test, pers_scores, pers_thr)

    beats_base = model_block["pr_auc"] > base_block["pr_auc"]
    beats_pers = model_block["pr_auc"] > pers_block["pr_auc"]

    lead = compute_lead_times(reports, scored_test, threshold, cfg)

    results = {
        "threshold": round(float(threshold), 6),
        "threshold_policy": cfg.threshold_policy,
        "forward_window_days": cfg.forward_window_days,
        "split_date": cfg.split_date,
        "test": {
            "model": model_block,
            "baseline_base_rate": base_block,
            "baseline_persistence": pers_block,
            "model_beats_base_rate": bool(beats_base),
            "model_beats_persistence": bool(beats_pers),
            "pr_curve": metrics.pr_curve_points(y_test, test_scores),
            "pr_curve_persistence": metrics.pr_curve_points(y_test, pers_scores),
            "calibration": metrics.calibration_points(y_test, test_scores),
        },
        "lead_time": lead,
        "coefficients": model.coefficients(),
        "gbt_importances": model.gbt_importances(),
        "gbt_test_pr_auc": round(float(
            metrics.average_precision_score(y_test, model.predict_proba_gbt(test_df))
        ), 6),
    }
    return {"results": results, "model": model, "scored_test": scored_test,
            "train_df": train_df, "test_df": test_df}
