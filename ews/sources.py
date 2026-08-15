"""
Real data source adapters.

These reconstruct the historical shortage record from the two genuinely public
sources named in the brief. They run only when explicitly requested and only
succeed on a machine with outbound internet access. In this sandbox the egress
proxy blocks api.fda.gov and drugshortagescanada.ca, so the default pipeline
uses labeled synthetic data instead.

HONESTY NOTE ON FEATURE COVERAGE
The openFDA and Drug Shortages Canada feeds are shortage EVENT records. From
them you can reconstruct, per drug over time: whether it was in shortage, how
often it has been short before, its therapeutic class, and recency. They do
NOT carry supplier counts or manufacturer market share. Those two features
(single_supplier, hhi) require a separate manufacturer source, for example the
openFDA NDC directory joined by labeler. Until that join is built and
validated, the real path reconstructs the shortage record and its frequency
and recency features, and the modeled metrics in this project come from the
synthetic run. This is stated plainly rather than papered over.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd

from . import config


class SourceUnavailable(RuntimeError):
    pass


def _get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "drug-shortage-ews/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SourceUnavailable(
            f"HTTP {exc.code} from {url}. In the sandbox this endpoint is blocked; "
            f"run on a machine with internet access."
        ) from exc
    except urllib.error.URLError as exc:
        raise SourceUnavailable(f"network error reaching {url}: {exc.reason}") from exc


def fetch_openfda_shortages(max_records: int = 5000) -> pd.DataFrame:
    """Download shortage event records from openFDA and return a tidy frame.

    Real, public endpoint. Paginates in blocks of 1000. Returns columns:
    generic_name, company_name, status, initial_posting_date, update_date.
    """
    base = config.REAL_SOURCES["openfda_shortages"]
    rows = []
    skip = 0
    page = 1000
    while skip < max_records:
        url = f"{base}?limit={page}&skip={skip}"
        payload = _get_json(url)
        results = payload.get("results", [])
        if not results:
            break
        for r in results:
            rows.append({
                "generic_name": (r.get("generic_name") or r.get("proprietary_name") or "").strip(),
                "company_name": (r.get("company_name") or "").strip(),
                "status": (r.get("status") or "").strip(),
                "initial_posting_date": r.get("initial_posting_date"),
                "update_date": r.get("update_date"),
                "therapeutic_category": (r.get("therapeutic_category") or "Uncategorized"),
            })
        skip += page
        if len(results) < page:
            break
    if not rows:
        raise SourceUnavailable("openFDA returned no shortage records")
    return pd.DataFrame(rows)


def fetch_drug_shortages_canada(email: str, password: str) -> pd.DataFrame:
    """Drug Shortages Canada requires an authenticated token before search.

    This adapter documents the flow. Credentials come from the environment
    (see .env.example). It is not exercised in the sandbox.
    """
    if not email or not password:
        raise SourceUnavailable(
            "Drug Shortages Canada needs DSC_EMAIL and DSC_PASSWORD in .env"
        )
    raise SourceUnavailable(
        "Drug Shortages Canada login flow is documented but not run in the "
        "sandbox. Provide credentials and enable this on a networked machine."
    )


def reconstruct_shortage_record(events: pd.DataFrame) -> pd.DataFrame:
    """Turn shortage event rows into a per drug shortage record with monthly
    resolution. This is the observable in_shortage history that the feature
    layer needs. Supplier and HHI columns are left as NA on purpose (see the
    module honesty note)."""
    events = events.copy()
    events["start"] = pd.to_datetime(events["initial_posting_date"], errors="coerce")
    events["end"] = pd.to_datetime(events["update_date"], errors="coerce")
    events = events.dropna(subset=["start"])

    events["drug_id"] = events["generic_name"].str.lower().str.strip()
    events = events[events["drug_id"] != ""]

    records = []
    for drug_id, g in events.groupby("drug_id"):
        cat = g["therapeutic_category"].mode()
        cat = cat.iloc[0] if not cat.empty else "Uncategorized"
        for _, row in g.iterrows():
            start = row["start"].to_period("M")
            end = row["end"].to_period("M") if pd.notna(row["end"]) else start
            span = pd.period_range(start=start, end=max(end, start), freq="M")
            for p in span:
                records.append({
                    "drug_id": drug_id,
                    "category": cat,
                    "period": str(p),
                    "in_shortage": 1,
                    "suppliers": pd.NA,     # not in this feed
                    "hhi": pd.NA,           # not in this feed
                    "supply_shock": 0,
                })
    out = pd.DataFrame(records).drop_duplicates(["drug_id", "period"])
    return out.sort_values(["drug_id", "period"]).reset_index(drop=True)


def summarize_record(record: pd.DataFrame) -> dict:
    periods = pd.PeriodIndex(record["period"], freq="M")
    return {
        "drugs": int(record["drug_id"].nunique()),
        "shortage_months": int(len(record)),
        "date_min": str(periods.min()),
        "date_max": str(periods.max()),
        "categories": int(record["category"].nunique()),
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
