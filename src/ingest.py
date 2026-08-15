"""
ingest.py
Data ingestion with graceful degradation and honest lineage.

Order of operations (data_mode="auto"):
  1. If a cached raw snapshot exists and use_cache is on -> reuse it (this is what
     makes two runs byte-identical without re-hitting the network).
  2. Else try the REAL sources:
       - openFDA drug shortages   : https://api.fda.gov/drug/shortages.json
       - Drug Shortages Canada API: https://www.drugshortagescanada.ca/api/v1/
     Both are normalised into one canonical report schema.
  3. If the real sources are unreachable, fall back to the synthetic DGP and label
     everything source="synthetic".

Canonical report schema (one dict per report):
    source, drug_id, drug_name, therapeutic_category, manufacturer,
    report_type ("shortage"|"discontinuation"), status,
    start_date (ISO, official/declaration/initial-posting date),
    end_date  (ISO or None, resolution date), pull_timestamp

IMPORTANT (documented assumptions):
  * drug_id is the normalised generic name. Strengths / dosage forms collapse to
    one drug. This is a deliberate aggregation choice; see README.
  * The DSC public API requires an authenticated token (set EWS_DSC_EMAIL and
    EWS_DSC_PASSWORD). Without credentials DSC is skipped and recorded as such.
  * Field names below reflect the sources' documented schemas. On the FIRST real
    run, verify against the live response and adapt the small mapping functions
    if a field has been renamed. The code degrades gracefully if a field is
    missing rather than inventing a value.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from . import synthetic
from .util import (log, read_json, read_jsonl, utc_now_iso, write_jsonl,
                   write_json)

try:
    import requests
except Exception:  # pragma: no cover - requests is pinned in requirements
    requests = None


OPENFDA_URL = "https://api.fda.gov/drug/shortages.json"
DSC_BASE = "https://www.drugshortagescanada.ca/api/v1"


class SourceUnreachable(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def _norm_drug_id(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "unknown"


def _iso_or_none(value: Optional[str]) -> Optional[str]:
    """Accept 'YYYYMMDD', 'YYYY-MM-DD', or ISO datetime; return ISO date or None."""
    if not value:
        return None
    v = str(value).strip()
    m = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", v)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# --------------------------------------------------------------------------- #
# openFDA
# --------------------------------------------------------------------------- #
def fetch_openfda(pull_ts: str, max_records: int = 20000) -> list[dict]:
    if requests is None:
        raise SourceUnreachable("requests not available")
    out: list[dict] = []
    limit = 1000
    skip = 0
    try:
        while skip < max_records:
            resp = requests.get(
                OPENFDA_URL,
                params={"limit": limit, "skip": skip},
                timeout=30,
            )
            if resp.status_code == 404 and skip == 0:
                raise SourceUnreachable("openFDA returned 404 (no results)")
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                break
            for r in results:
                report_type = ("discontinuation"
                               if "discontin" in str(r.get("status", "")).lower()
                               else "shortage")
                out.append({
                    "source": "openfda",
                    "drug_id": _norm_drug_id(r.get("generic_name")
                                             or r.get("proprietary_name")),
                    "drug_name": (r.get("generic_name")
                                  or r.get("proprietary_name") or "unknown"),
                    "therapeutic_category": (r.get("therapeutic_category")
                                             or "Unclassified"),
                    "manufacturer": r.get("company_name") or "unknown",
                    "report_type": report_type,
                    "status": r.get("status") or "unknown",
                    "start_date": _iso_or_none(r.get("initial_posting_date")),
                    "end_date": _iso_or_none(r.get("resolved_note_date")
                                             or r.get("update_date")
                                             if str(r.get("status", "")).lower()
                                             == "resolved" else None),
                    "pull_timestamp": pull_ts,
                })
            skip += limit
    except SourceUnreachable:
        raise
    except Exception as exc:  # network / HTTP / JSON errors -> unreachable
        raise SourceUnreachable(f"openFDA error: {type(exc).__name__}: {exc}")
    # Drop rows with no usable start date (cannot be placed point-in-time).
    out = [r for r in out if r["start_date"]]
    log(f"openFDA: {len(out)} usable reports")
    return out


# --------------------------------------------------------------------------- #
# Drug Shortages Canada (requires auth token)
# --------------------------------------------------------------------------- #
def fetch_dsc(pull_ts: str, max_pages: int = 60) -> list[dict]:
    if requests is None:
        raise SourceUnreachable("requests not available")
    email = os.environ.get("EWS_DSC_EMAIL")
    password = os.environ.get("EWS_DSC_PASSWORD")
    if not (email and password):
        raise SourceUnreachable(
            "DSC skipped: set EWS_DSC_EMAIL and EWS_DSC_PASSWORD to enable "
            "(the DSC API requires an authenticated token)."
        )
    try:
        sess = requests.Session()
        login = sess.post(f"{DSC_BASE}/login",
                          json={"email": email, "password": password},
                          timeout=30)
        login.raise_for_status()
        token = login.headers.get("auth-token") or login.json().get("auth_token")
        if not token:
            raise SourceUnreachable("DSC login returned no token")
        headers = {"auth-token": token}

        out: list[dict] = []
        page = 1
        while page <= max_pages:
            resp = sess.get(f"{DSC_BASE}/search",
                            params={"term": "", "page": page},
                            headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", payload.get("results", []))
            if not data:
                break
            for r in data:
                rtype = str(r.get("type", "shortage")).lower()
                report_type = ("discontinuation" if "discont" in rtype
                               else "shortage")
                out.append({
                    "source": "dsc",
                    "drug_id": _norm_drug_id(r.get("en_ingredients")
                                             or r.get("brand_name")
                                             or r.get("company_name")),
                    "drug_name": (r.get("brand_name")
                                  or r.get("en_ingredients") or "unknown"),
                    "therapeutic_category": (r.get("atc_description")
                                             or "Unclassified"),
                    "manufacturer": r.get("company_name") or "unknown",
                    "report_type": report_type,
                    "status": r.get("status") or "unknown",
                    "start_date": _iso_or_none(r.get("actual_start_date")
                                               or r.get("anticipated_start_date")
                                               or r.get("created_date")),
                    "end_date": _iso_or_none(r.get("actual_end_date")
                                             or r.get("estimated_end_date")),
                    "pull_timestamp": pull_ts,
                })
            page += 1
        out = [r for r in out if r["start_date"]]
        log(f"DSC: {len(out)} usable reports")
        return out
    except SourceUnreachable:
        raise
    except Exception as exc:
        raise SourceUnreachable(f"DSC error: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def ingest(cfg) -> dict:
    """Return {'reports': [...], 'lineage': {...}}. Writes the working snapshot
    to data/raw/ and (in synthetic mode) the sealed holdout to data/holdout/."""
    os.makedirs(cfg.raw_dir, exist_ok=True)
    cache_path = os.path.join(cfg.raw_dir, "reports_cache.jsonl")
    cache_meta_path = os.path.join(cfg.raw_dir, "cache_meta.json")
    pull_ts = utc_now_iso()

    # 1) cache reuse (determinism) -- but never serve a cache whose provenance is
    #    incompatible with the requested mode (e.g. real mode must NOT reuse a
    #    synthetic cache and mislabel it as real).
    if cfg.use_cache and os.path.exists(cache_path):
        cached_src = "unknown"
        if os.path.exists(cache_meta_path):
            try:
                cached_src = read_json(cache_meta_path).get("cached_source",
                                                            "unknown")
            except Exception:
                cached_src = "unknown"
        compatible = (
            cfg.data_mode == "auto"
            or (cfg.data_mode == "real" and cached_src == "real")
            or (cfg.data_mode == "synthetic" and cached_src == "synthetic")
        )
        if compatible:
            reports = read_jsonl(cache_path)
            lineage = {
                "resolved_source": "cache",
                "cached_source": cached_src,
                "cache_path": cache_path,
                "records": len(reports),
                "note": ("Reused cached raw snapshot for reproducibility "
                         f"(underlying source: {cached_src})."),
            }
            log(f"using cached raw snapshot: {len(reports)} reports "
                f"(underlying source: {cached_src})")
            return {"reports": reports, "lineage": lineage}
        else:
            log(f"cache present but its source ({cached_src}) is incompatible "
                f"with data_mode={cfg.data_mode}; ignoring cache.")

    real_attempts = []
    reports: list[dict] = []
    resolved = None

    if cfg.data_mode in ("auto", "real"):
        # openFDA
        try:
            fda = fetch_openfda(pull_ts)
            reports.extend(fda)
            real_attempts.append({"source": "openfda", "status": "ok",
                                  "records": len(fda), "pull_timestamp": pull_ts})
        except SourceUnreachable as exc:
            real_attempts.append({"source": "openfda", "status": "unreachable",
                                  "detail": str(exc), "pull_timestamp": pull_ts})
        # DSC
        try:
            dsc = fetch_dsc(pull_ts)
            reports.extend(dsc)
            real_attempts.append({"source": "dsc", "status": "ok",
                                  "records": len(dsc), "pull_timestamp": pull_ts})
        except SourceUnreachable as exc:
            real_attempts.append({"source": "dsc", "status": "unreachable",
                                  "detail": str(exc), "pull_timestamp": pull_ts})

        if reports:
            resolved = "real"

    if resolved is None:
        if cfg.data_mode == "real":
            raise SourceUnreachable(
                "data_mode='real' but no real source returned data. Attempts: "
                + str(real_attempts)
            )
        # 3) synthetic fallback (auto or synthetic)
        manifest = synthetic.generate(cfg)
        reports = read_jsonl(manifest["raw_path"])
        for r in reports:
            r.setdefault("pull_timestamp", pull_ts)
        resolved = "synthetic"
        lineage = {
            "resolved_source": "synthetic",
            "reason": ("real sources unreachable"
                       if cfg.data_mode == "auto" else "data_mode=synthetic"),
            "real_attempts": real_attempts,
            "synthetic_manifest": manifest,
            "records": len(reports),
            "pull_timestamp": pull_ts,
            "WARNING": ("SYNTHETIC DATA. Not observed real-world data. Metrics "
                        "demonstrate pipeline signal-recovery only."),
        }
    else:
        # real data: also generate a sealed holdout carved from the tail time slice
        reports.sort(key=lambda r: (r["drug_id"], r["start_date"],
                                    r["report_type"]))
        lineage = {
            "resolved_source": "real",
            "real_attempts": real_attempts,
            "records": len(reports),
            "pull_timestamp": pull_ts,
            "note": ("Real open-data pull. Point-in-time reconstruction is only "
                     "as good as the sources' posting dates; see LIMITATIONS.md."),
        }
        # In real mode the sealed holdout is a disjoint DRUG sample (25%) so the
        # final check measures generalisation to unseen drugs, mirroring synthetic.
        _carve_real_holdout(cfg, reports)

    # cache the working snapshot, tagged with its true provenance so a later
    # real-mode run can refuse to reuse a synthetic cache.
    write_jsonl(cache_path, reports)
    write_json(cache_meta_path, {"cached_source": resolved,
                                 "pull_timestamp": pull_ts,
                                 "records": len(reports)})
    write_json(os.path.join(cfg.raw_dir, "ingest_lineage.json"), lineage)
    return {"reports": reports, "lineage": lineage}


def _carve_real_holdout(cfg, reports: list[dict]) -> None:
    """Deterministically move ~25% of drugs (by hashed id) into the sealed
    holdout file, and REMOVE them from the working set. Structural separation."""
    import hashlib

    def in_holdout(drug_id: str) -> bool:
        h = int(hashlib.sha256(drug_id.encode("utf-8")).hexdigest(), 16)
        return (h % 4) == 0  # ~25%

    holdout = [r for r in reports if in_holdout(r["drug_id"])]
    working = [r for r in reports if not in_holdout(r["drug_id"])]
    reports[:] = working
    os.makedirs(cfg.holdout_dir, exist_ok=True)
    write_jsonl(os.path.join(cfg.holdout_dir, "holdout_reports.jsonl"), holdout)
    log(f"real mode: sealed {len(holdout)} holdout reports "
        f"({len({r['drug_id'] for r in holdout})} drugs)")
