# DATA_LINEAGE

This document records where the data comes from, how each source maps into the
canonical schema, what is and is not reconstructable point-in-time, and the record
counts of the reference build. The pipeline also writes a machine-readable lineage
to `data/raw/ingest_lineage.json` and `output/run_lineage.json` on every run.

## Canonical report schema

Every source is normalized into one record shape:

```
source                (openfda | dsc | synthetic)
drug_id               normalized generic name (aggregation key)
drug_name
therapeutic_category
manufacturer
report_type           (shortage | discontinuation)
status
start_date            ISO date: the official / declaration / initial-posting date
end_date              ISO date or null: resolution date (never used to predict onset)
pull_timestamp        UTC time the record was ingested
```

## Sources

### 1. openFDA drug shortages
- Endpoint: `https://api.fda.gov/drug/shortages.json`, paginated by `limit` and
  `skip`.
- Field mapping (verify on first live run; the code degrades gracefully if a field
  is renamed rather than inventing a value):
  - `drug_id` from `generic_name` (fallback `proprietary_name`), normalized.
  - `therapeutic_category` from `therapeutic_category` (fallback "Unclassified").
  - `manufacturer` from `company_name`.
  - `start_date` from `initial_posting_date`.
  - `report_type` is discontinuation if the status text contains "discontin",
    else shortage.
- Records with no usable start date are dropped (they cannot be placed
  point-in-time).

### 2. Drug Shortages Canada (DSC)
- Base: `https://www.drugshortagescanada.ca/api/v1/`. The search endpoint requires
  an authenticated token obtained from `/login`.
- Enable by setting `EWS_DSC_EMAIL` and `EWS_DSC_PASSWORD`. Without credentials the
  source is skipped and the skip is recorded in the lineage.
- Field mapping (verify on first live run):
  - `drug_id` from `en_ingredients` (fallback `brand_name`, `company_name`).
  - `manufacturer` from `company_name`.
  - `start_date` from `actual_start_date` (fallback `anticipated_start_date`,
    `created_date`).
  - `end_date` from `actual_end_date` (fallback `estimated_end_date`).
  - `report_type` is discontinuation if the type text contains "discont", else
    shortage.

## Point-in-time reconstruction: what is and is not available

- **Available**: the posting/declaration date of each shortage and discontinuation
  report, the reporting manufacturer, and (for openFDA) a therapeutic category.
  These support point-in-time features: prior-shortage counts and recency,
  discontinuation signals, distinct-manufacturer and concentration measures, and an
  expanding category base rate.
- **Not reliably available point-in-time**: any revision history of a record. We
  observe a report as of its posting date; if a source later backfills or revises,
  a true real-time observer's view could have differed. We do not attempt to
  reconstruct revision history and we do not impute it.
- **Never used**: the resolution (end) date of a shortage is never used to predict
  that shortage's onset. It is retained only to determine whether a drug was in an
  active shortage at a cutoff (active cells are then excluded).

## Pull behavior and caching

1. If a cached raw snapshot exists and is compatible with the requested mode, it is
   reused for reproducibility. A real-mode run will **refuse** to reuse a synthetic
   cache (and vice versa); the cache is tagged with its true provenance in
   `data/raw/cache_meta.json`.
2. Otherwise the real sources are attempted and normalized.
3. If the real sources are unreachable and the mode allows it, the deterministic
   synthetic generator is used and everything is labeled SYNTHETIC.

## Reference build record counts

The environment used to build this reference had **no access to the real APIs**
(both returned HTTP 403 from an egress allowlist), so the pipeline fell back to the
synthetic generator. Recorded in `ingest_lineage.json`:

- Resolved source: **synthetic** (reason: real sources unreachable).
- Real attempts:
  - openFDA: unreachable (HTTP 403 Forbidden).
  - DSC: skipped (no credentials provided).
- Working reports: **784** across **220** drugs.
- Sealed holdout reports: **122** across **40** disjoint drugs.
- Synthetic date range: 2018-01-01 to 2024-12-31.
- Seed: 20260810.

To produce a real-data lineage, run `python run.py --data-mode real` on a machine
with internet access (and DSC credentials if desired). The lineage file will then
record the live pull timestamps and per-source record counts.
