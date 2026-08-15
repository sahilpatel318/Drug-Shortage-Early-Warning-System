"""
features.py
Point-in-time feature construction.

For each (drug, as_of) cell on a month-end grid we compute features using ONLY
reports whose start_date <= as_of, and a forward label using shortage onsets whose
start_date is in (as_of, as_of + window]. Every row records `max_used_date`, the
latest source date consulted, so leakage.py can independently prove
max_used_date <= as_of for every row.

Design decisions (documented in README):
  * Prediction unit: drug (generic name). Strengths/forms collapse to one drug.
  * Label: 1 iff a NEW shortage onset falls in the forward window AND the drug is
    not already in an active shortage at as_of. Cells where the drug is already in
    an active shortage are EXCLUDED (you cannot "enter" a shortage you are in).
  * Cells whose forward window extends past the data end are dropped (right-
    censoring); otherwise absence of a label would be indistinguishable from
    "not yet observed".
  * No resolution-to-onset leakage: end_date of the labelled future shortage is
    never read; the label only checks for an onset start_date in the window.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .util import log

CAP_DAYS = 2000  # sentinel cap for "days since ..." when the event never happened

FEATURE_COLUMNS = [
    "prior_shortage_count",
    "has_prior_shortage",
    "days_since_last_shortage",
    "recent_shortage_count_365",
    "distinct_manufacturers",
    "single_supplier_flag",
    "manufacturer_hhi",
    "disc_count",
    "days_since_last_disc",
    "recent_disc_flag_120",
    "category_base_rate_pit",
]


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def _grid(start: date, end: date, freq: str) -> list[date]:
    idx = pd.date_range(start=start, end=end, freq=freq)
    return [ts.date() for ts in idx]


def _hhi(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(sum((c / total) ** 2 for c in counts.values()))


def build_panel(reports: list[dict], cfg) -> pd.DataFrame:
    """Return a point-in-time panel DataFrame with FEATURE_COLUMNS + meta."""
    window = timedelta(days=cfg.forward_window_days)

    # Index reports by drug -------------------------------------------------- #
    by_drug: dict[str, dict] = {}
    all_dates: list[date] = []
    for r in reports:
        sd = _d(r.get("start_date"))
        if sd is None:
            continue
        all_dates.append(sd)
        d = by_drug.setdefault(r["drug_id"], {
            "drug_id": r["drug_id"],
            "drug_name": r.get("drug_name", r["drug_id"]),
            "therapeutic_category": r.get("therapeutic_category", "Unclassified"),
            "shortages": [],        # list of (start_date, end_date)
            "discontinuations": [],  # list of start_date
            "mfr_events": [],       # list of (date, manufacturer)
        })
        mfr = r.get("manufacturer") or "unknown"
        d["mfr_events"].append((sd, mfr))
        if r["report_type"] == "shortage":
            d["shortages"].append((sd, _d(r.get("end_date"))))
        else:
            d["discontinuations"].append(sd)

    if not all_dates:
        raise ValueError("No dated reports available to build features.")

    for d in by_drug.values():
        d["shortages"].sort(key=lambda t: t[0])
        d["discontinuations"].sort()
        d["mfr_events"].sort(key=lambda t: t[0])

    data_start = min(all_dates)
    data_end = max(all_dates)
    grid = _grid(date(data_start.year, data_start.month, 1), data_end,
                 cfg.panel_freq)
    # Only as_of dates whose forward window stays inside observed data.
    grid = [g for g in grid if g + window <= data_end]
    if not grid:
        raise ValueError("Grid empty after censoring; widen the data range.")

    # Point-in-time category base rate --------------------------------------- #
    # onsets_by_cat_sorted[cat] = sorted onset dates ; drugs_per_cat = count
    onsets_by_cat: dict[str, list[date]] = {}
    drugs_per_cat: dict[str, int] = {}
    for d in by_drug.values():
        cat = d["therapeutic_category"]
        drugs_per_cat[cat] = drugs_per_cat.get(cat, 0) + 1
        onsets_by_cat.setdefault(cat, [])
        for (sd, _e) in d["shortages"]:
            onsets_by_cat[cat].append(sd)
    for cat in onsets_by_cat:
        onsets_by_cat[cat].sort()

    grid_sorted = sorted(grid)
    month_rank = {g: i + 1 for i, g in enumerate(grid_sorted)}

    def category_base_rate_pit(cat: str, as_of: date) -> float:
        onsets = onsets_by_cat.get(cat, [])
        # count onsets strictly on/before as_of
        lo, hi = 0, len(onsets)
        while lo < hi:
            mid = (lo + hi) // 2
            if onsets[mid] <= as_of:
                lo = mid + 1
            else:
                hi = mid
        onsets_le = lo
        exposure = drugs_per_cat.get(cat, 1) * month_rank.get(as_of, 1)
        return onsets_le / exposure if exposure else 0.0

    # Assemble rows ---------------------------------------------------------- #
    rows: list[dict] = []
    for drug_id in sorted(by_drug.keys()):
        d = by_drug[drug_id]
        shortages = d["shortages"]
        discs = d["discontinuations"]
        mfr_events = d["mfr_events"]

        for as_of in grid_sorted:
            # active shortage at as_of? (start<=as_of and (end is None or end>as_of))
            active = any(
                s <= as_of and (e is None or e > as_of) for (s, e) in shortages
            )
            if active:
                continue  # cannot "enter" a shortage while in one

            past_shortages = [s for (s, e) in shortages if s <= as_of]
            prior_count = len(past_shortages)
            has_prior = 1 if prior_count else 0
            if past_shortages:
                days_since_last = (as_of - past_shortages[-1]).days
            else:
                days_since_last = CAP_DAYS
            recent_365 = sum(1 for s in past_shortages
                             if (as_of - s).days <= 365)

            past_discs = [s for s in discs if s <= as_of]
            disc_count = len(past_discs)
            days_since_disc = ((as_of - past_discs[-1]).days
                               if past_discs else CAP_DAYS)
            recent_disc = 1 if (past_discs and
                                (as_of - past_discs[-1]).days <= 120) else 0

            mfr_counts: dict[str, int] = {}
            max_used = None
            for (ed, mfr) in mfr_events:
                if ed <= as_of:
                    mfr_counts[mfr] = mfr_counts.get(mfr, 0) + 1
                    if max_used is None or ed > max_used:
                        max_used = ed
            distinct_mfr = len(mfr_counts)
            single_supplier = 1 if distinct_mfr == 1 else 0
            hhi = _hhi(mfr_counts)

            # forward label: onset in (as_of, as_of + window]
            label = 1 if any(as_of < s <= as_of + window
                             for (s, e) in shortages) else 0

            # max source date consulted for FEATURES (labels excluded on purpose)
            max_used_date = max_used.isoformat() if max_used else as_of.isoformat()

            rows.append({
                "drug_id": drug_id,
                "drug_name": d["drug_name"],
                "therapeutic_category": d["therapeutic_category"],
                "as_of": as_of.isoformat(),
                "label": label,
                "max_used_date": max_used_date,
                # features
                "prior_shortage_count": prior_count,
                "has_prior_shortage": has_prior,
                "days_since_last_shortage": min(days_since_last, CAP_DAYS),
                "recent_shortage_count_365": recent_365,
                "distinct_manufacturers": distinct_mfr,
                "single_supplier_flag": single_supplier,
                "manufacturer_hhi": round(hhi, 6),
                "disc_count": disc_count,
                "days_since_last_disc": min(days_since_disc, CAP_DAYS),
                "recent_disc_flag_120": recent_disc,
                "category_base_rate_pit": round(
                    category_base_rate_pit(d["therapeutic_category"], as_of), 6),
            })

    df = pd.DataFrame(rows)
    # temporal split flag
    split = date.fromisoformat(cfg.split_date)
    df["split"] = df["as_of"].map(
        lambda s: "train" if date.fromisoformat(s) <= split else "test")
    df = df.sort_values(["as_of", "drug_id"], kind="stable").reset_index(drop=True)
    log(f"panel: {len(df)} rows, {df['drug_id'].nunique()} drugs, "
        f"positives={int(df['label'].sum())} "
        f"({df['label'].mean():.4f} base rate); "
        f"train={int((df.split=='train').sum())} test={int((df.split=='test').sum())}")
    return df
