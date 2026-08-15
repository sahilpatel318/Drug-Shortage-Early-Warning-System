"""
Evaluation.

Reports are honest and sized: every metric carries its evaluation N and, where
relevant, its positive count. Headline is PR-AUC (with named baselines), not
accuracy, because the task is rare event ranking. Lead time is measured only
for shortages the model actually flagged before the official onset month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .firewall import Vault

DAYS_PER_MONTH = 30.44


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def classification_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "threshold": round(float(thr), 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": int(len(y)), "positives": int(y.sum()),
    }


def lead_time(test: pd.DataFrame, vault: Vault, thr: float) -> dict:
    """For each test period onset, find the earliest month within the lookback
    window where the model flagged the drug, and record the warning in days."""
    scored = test[["drug_idx", "anchor_month", "ews"]].copy()
    by_drug = {d: g.set_index("anchor_month")["ews"].to_dict()
               for d, g in scored.groupby("drug_idx")}

    leads_days = []
    flagged = 0
    missed = 0
    total = 0
    for drug_idx, months_map in by_drug.items():
        onsets = vault.onset_months(drug_idx)
        test_onsets = [int(o) for o in onsets if o >= config.SPLIT_MONTH]
        for o in test_onsets:
            total += 1
            lo = max(config.SPLIT_MONTH, o - config.WARN_LOOKBACK)
            first_flag = None
            for m in range(lo, o):
                sc = months_map.get(m)
                if sc is not None and sc >= thr:
                    first_flag = m
                    break
            if first_flag is None:
                missed += 1
            else:
                flagged += 1
                leads_days.append((o - first_flag) * DAYS_PER_MONTH)

    leads = np.array(leads_days, dtype=np.float64)
    summary = {
        "test_onsets_total": total,
        "flagged_ahead": flagged,
        "missed": missed,
        "flag_rate": round(flagged / total, 6) if total else 0.0,
        "median_lead_days": round(float(np.median(leads)), 2) if leads.size else None,
        "p25_lead_days": round(float(np.percentile(leads, 25)), 2) if leads.size else None,
        "p75_lead_days": round(float(np.percentile(leads, 75)), 2) if leads.size else None,
        "max_lead_days": round(float(leads.max()), 2) if leads.size else None,
        "lead_days_hist": _histogram(leads),
    }
    return summary


def _histogram(vals: np.ndarray, n_bins: int = 8) -> list:
    if vals.size == 0:
        return []
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    return [
        {"lo": round(float(edges[i]), 1),
         "hi": round(float(edges[i + 1]), 1),
         "count": int(counts[i])}
        for i in range(n_bins)
    ]


def km_survival_by_risk(test: pd.DataFrame, vault: Vault) -> dict:
    """Empirical time to next shortage by model risk group.

    For each at risk test anchor, time to event is the gap in months to the
    next onset for that drug; rows with no later onset are right censored at
    the end of the observed panel. Groups are terciles of the model score.
    """
    end = config.PERIOD_MONTHS - 1
    df = test[["drug_idx", "anchor_month", "ews"]].copy()

    times, events, scores = [], [], []
    for drug_idx, g in df.groupby("drug_idx"):
        onsets = vault.onset_months(drug_idx)
        for _, r in g.iterrows():
            m = int(r["anchor_month"])
            later = onsets[onsets > m]
            if later.size:
                times.append(int(later.min() - m))
                events.append(1)
            else:
                times.append(int(end - m))
                events.append(0)
            scores.append(float(r["ews"]))

    times = np.array(times)
    events = np.array(events)
    scores = np.array(scores)

    q1, q2 = np.quantile(scores, [1 / 3, 2 / 3])
    group = np.where(scores >= q2, "high", np.where(scores >= q1, "medium", "low"))

    out = {"groups": {}, "n": int(len(times))}
    for name in ("high", "medium", "low"):
        mask = group == name
        curve, median = _km(times[mask], events[mask])
        out["groups"][name] = {
            "n": int(mask.sum()),
            "events": int(events[mask].sum()),
            "median_months": median,
            "curve": curve,
        }
    return out


def _km(times: np.ndarray, events: np.ndarray):
    """Kaplan Meier estimator. Returns (curve points, median survival months)."""
    if times.size == 0:
        return [], None
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]
    unique_t = np.unique(times[events == 1])
    n = times.size
    surv = 1.0
    curve = [{"t": 0, "s": 1.0}]
    median = None
    for ut in unique_t:
        at_risk = int((times >= ut).sum())
        d = int(((times == ut) & (events == 1)).sum())
        if at_risk == 0:
            continue
        surv *= (1.0 - d / at_risk)
        curve.append({"t": int(ut), "s": round(float(surv), 6)})
        if median is None and surv <= 0.5:
            median = int(ut)
    return curve, median


def top_at_risk(test: pd.DataFrame, panel, thr: float, top_k: int = 12) -> list:
    """Latest scored month per drug, ranked by risk, for the watchlist table."""
    latest_month = test["anchor_month"].max()
    latest = test[test["anchor_month"] == latest_month].copy()
    latest = latest.sort_values("ews", ascending=False, kind="mergesort").head(top_k)

    id_by_idx = {int(d[1:]): d for d in panel.frame["drug_id"].unique()}
    cat_by_id = panel.frame.groupby("drug_id")["category"].first().to_dict()

    rows = []
    for _, r in latest.iterrows():
        did = id_by_idx[int(r["drug_idx"])]
        rows.append({
            "drug_id": did,
            "category": cat_by_id[did],
            "score": round(float(r["ews"]), 4),
            "flag": bool(r["ews"] >= thr),
            "suppliers": int(r["suppliers"]),
            "single_supplier": bool(r["single_supplier"]),
            "hhi": round(float(r["hhi"]), 3),
            "past_onsets": int(round(np.expm1(r["log_past_onsets"]))),
            "anchor_period": panel.frame[
                (panel.frame["drug_id"] == did) &
                (panel.frame["month"] == int(r["anchor_month"]))
            ]["period"].iloc[0],
        })
    return rows
