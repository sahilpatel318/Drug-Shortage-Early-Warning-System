# Drug-Shortage Early-Warning System (retrospective signal-recovery PoC)

A proof-of-concept that scores each drug's risk of **entering a shortage within a
configurable forward window (default 90 days)**, ranks drugs by that risk, explains
every score by its top contributing features, and backtests the flags against
historical shortage onsets by **lead time** (how many days ahead of the official
shortage date the score would have flagged the drug).

The model proposes and ranks. A human disposes. This tool never issues a
procurement or clinical verdict.

## Honest framing (read this first)

This is a **retrospective signal-recovery** PoC on historical open data. It is
**not** a deployment-ready predictor.

What it **does**:
- Reconstructs strictly point-in-time features (only information available on or
  before each prediction cutoff).
- Uses a time-based train/test split (train earlier, test later), never a random
  split on time-ordered data.
- Runs a leakage self-check that **fails the run** if any feature carries a source
  date later than its prediction cutoff.
- Reports the model only in comparison to two naive baselines (base rate,
  prior-shortage persistence), and states plainly when it does not beat them.
- Reports imbalance-aware metrics (PR-AUC, precision, recall, confusion matrix,
  Brier, calibration). Accuracy is deliberately **not** a headline metric.
- Reports **lead time** as the headline result, with a full distribution.
- Keeps a planted ground-truth holdout that the training/scoring pipeline is
  structurally forbidden to read until final evaluation.

What it does **not** do:
- It makes no dollar-savings claims and no "prevents X shortages" claims.
- It implies no deployment readiness.
- It issues no verdict. It flags and ranks for human review.

## Data sources and what is honestly reconstructable

Two real open sources are targeted by the ingestion layer:

1. **openFDA drug shortages** (`https://api.fda.gov/drug/shortages.json`): historical
   shortage and status records.
2. **Drug Shortages Canada** (`https://www.drugshortagescanada.ca/api/v1/`): shortage
   and discontinuation reports. The DSC API requires an authenticated token; set
   `EWS_DSC_EMAIL` and `EWS_DSC_PASSWORD` to enable it. Without credentials, DSC is
   skipped and the omission is recorded.

**Point-in-time honesty.** These feeds expose a report's posting/declaration date,
which is what we treat as the observation date. True point-in-time state is only as
good as those posting dates. If a source backfills or revises records, the
reconstructable history may differ from what a real-time observer would have seen.
This is documented in `DATA_LINEAGE.md` and `LIMITATIONS.md`. No field is ever
imputed and presented as observed. If a source is unreachable, the pipeline
degrades gracefully (see below) and records exactly what happened.

**Synthetic fallback (fully labeled).** If the real sources are unreachable (for
example, no network access, or a restricted build environment), the pipeline falls
back to a deterministic **synthetic** generator and labels every record, the data
lineage, and the dashboard banner as SYNTHETIC. Synthetic figures demonstrate that
the pipeline recovers planted signal. They say **nothing** about real-world
performance. Synthetic data is never presented as observed.

> The reference results in this README were produced on the **synthetic fallback**
> (the build environment had no access to the real APIs). They are labeled
> accordingly. To regenerate on real data, run on a machine with internet access
> and (optionally) DSC credentials.

## How to run

```bash
pip install -r requirements.txt

# End-to-end. Tries real sources; falls back to synthetic if unreachable.
python run.py

# Force a mode:
python run.py --data-mode real        # real only; fails loudly if unreachable
python run.py --data-mode synthetic   # never touches the network

# Prove two runs are byte-identical on metrics:
python run.py --verify-determinism

# Independent leakage / holdout-seal tests:
python tests/test_leakage.py
```

Then open `output/index.html` in a browser. Outputs are written to `output/`
(`metrics.json`, `run_lineage.json`, `index.html`) and raw snapshots to `data/`.

## Results (reference synthetic run, seed 20260810)

**These are synthetic-data results. They measure pipeline signal-recovery only.**

Temporal test set (as-of cutoffs after 2023-01-31), forward window 90 days, test
base rate 0.107:

| Scorer                              | PR-AUC | Precision | Recall | F1    | Brier  |
|-------------------------------------|--------|-----------|--------|-------|--------|
| Logistic regression (headline)      | 0.314  | 0.233     | 0.499  | 0.317 | 0.088  |
| Baseline: base rate (constant)      | 0.107  | 0.000     | 0.000  | 0.000 | 0.097  |
| Baseline: prior-shortage persistence| 0.268  | 0.214     | 0.519  | 0.303 | 0.559  |
| Gradient-boosted trees (reference)  | 0.263  | -         | -      | -     | -      |

The logistic regression beats both baselines on PR-AUC on this synthetic run. It is
kept as the headline over the gradient-boosted trees for interpretability: every
score decomposes into per-feature contributions. Operating threshold 0.135 (max-F1
on train). ROC-AUC 0.696 is reported only as a secondary reference.

**Lead time (headline result).** Among 142 shortage onsets in the test period, 81
(detection rate 0.57) were flagged in advance at the operating threshold. For those,
the lead time before the official shortage date was: median 82 days, IQR 41 to 207
days, range 1 to 508 days. Lead time can exceed the 90-day label window when the
model flags elevated risk earlier than the labeling horizon.

**Sealed holdout (planted ground truth, 37 unseen drugs).** PR-AUC 0.125 against a
base rate of 0.058 (roughly 2x base rate on entities never seen in training), median
lead time 94 days. Modest and honestly reported.

**Reproducibility.** `output/metrics.json` is byte-identical across runs. Reference
SHA-256:
`2d12146d24a569e0aec48791e460816eab27d3903e9d688be6dfb4ed7fa539d6`
(this hash is specific to the synthetic reference run and its seed; a real-data run
produces a different but internally reproducible hash).

## Temporal integrity and the leakage self-check

Every `(drug, as_of)` panel row records `max_used_date`, the latest source date any
feature consulted. `src/leakage.py` fails the run if `max_used_date > as_of` for any
row, if the temporal split is violated, or if the split boundary leaks. The check is
deliberately blunt: leakage crashes, it does not warn. An injected leak is caught
(see the note in `tests/`). There is no resolution-to-onset leakage: the end
(resolution) date of the labeled future shortage is never read; the label only
checks for an onset start date inside the forward window.

## The sealed holdout

A disjoint set of drugs (synthetic mode) or a deterministically hashed 25% of drugs
(real mode) is written to `data/holdout/` and quarantined. `src/holdout_guard.py`
holds a process-global latch that starts locked; only `src/final_eval.py` may open
it, for the duration of the final evaluation. Any attempt to read the holdout while
locked raises `HoldoutViolation` and fails the run. A test enforces that no
modeling or scoring module reads the holdout and that `load_holdout()` is called
only from `final_eval.py`.

## Features (all point-in-time)

manufacturer concentration (HHI) and single-supplier dependence, prior shortage
count and recency, recent-365-day shortage count, distinct manufacturers, prior
discontinuation notices and their recency, a recent-discontinuation flag, and a
point-in-time therapeutic-category base rate (computed only from data on or before
the cutoff). Every feature is computable point-in-time. Cells where a drug is
already in an active shortage are excluded, because a drug cannot "enter" a
shortage it is already in.

## Modeling

Regularized (L2) logistic regression on standardized features is the headline model,
chosen for interpretability. A gradient-boosted-trees model is trained alongside for
a comparison PR-AUC and feature importances, but it never overrides the logistic
explanation and is not the headline. Probabilities are the raw logistic outputs (no
class reweighting) so they remain interpretable as probabilities; imbalance is
handled at the decision threshold and through PR-based metrics. Calibration is shown
honestly via a reliability curve on the validation page.

## Optional narration layer

Per-drug plain-language rationale, driven only by a drug's top feature
contributions. Narration explains; it never decides and never introduces a fact not
already in the contributions. Fallback chain: Anthropic (if `ANTHROPIC_API_KEY`),
then OpenAI (if `OPENAI_API_KEY`), then a deterministic template stub. The stub is
the default so the pipeline runs fully offline. Narration output is excluded from the
reproducibility hash because language-model outputs are not deterministic;
`metrics.json` (the reproducible artifact) contains no narration.

## Design system

Aesthetic anchors Bloomberg Terminal and Linear: dense, table-first, information
rich. IBM Plex Sans and IBM Plex Mono (with offline-safe fallback stacks). Slate on
warm paper. Status shown by glyph plus label (never color alone): triangle HIGH,
diamond ELEVATED, filled circle WATCH, open circle LOW. The dashboard uses no hero
banners, no gradients, no large rounded cards, no drop shadows, no emoji headings, no
three-KPI-tile rows, and no chatbot widget. Charts (PR curve, calibration curve,
lead-time histogram) are hand-drawn inline SVG with no chart library, so the report
is dense, dependency-free, and offline-capable.

## Assumptions (every one recorded)

1. Prediction unit is the generic drug. Strengths and dosage forms collapse to one
   drug (`drug_id` is the normalized generic name).
2. The observation date for a report is its posting/declaration date (openFDA
   `initial_posting_date`; DSC actual/anticipated start or created date).
3. A "shortage event" is a shortage-type report with a start date. The label is 1 if
   a new onset falls in `(as_of, as_of + window]`.
4. Cells where the drug is already in an active shortage at `as_of` are excluded.
5. Panel cutoffs are month-ends; cutoffs whose forward window would exceed the data
   end are dropped (right-censoring) so absence of a label is never confused with
   "not yet observed".
6. Lead time attributes the earliest elevated out-of-sample flag to an onset only if
   no other onset of that drug falls between the flag and the onset, within a 730-day
   look-back cap.
7. The therapeutic-category base rate is an expanding point-in-time estimate.
8. In synthetic mode, the generator's latent variables are never written out; the
   feature builder recovers signal from observable report history alone.
9. Real-mode determinism is guaranteed against the cached raw snapshot; a fresh
   network pull can differ if the sources change between pulls.

## Interview honesty

True scope: retrospective signal recovery on open drug-shortage data. This project
demonstrates temporal integrity (point-in-time features, a time-based split, and a
leakage self-check that fails the run), honest baselines, imbalance-aware metrics,
calibrated and explainable scores, and a lead-time backtest with a sealed
ground-truth holdout. It is explicitly not a production predictor. On the synthetic
fallback, the numbers show only that the pipeline recovers planted signal, not that
the approach works in the real world. The right way to describe the headline result
is "on this synthetic run the score would have flagged a median of 82 days ahead of
the official shortage date for the onsets it caught", not "saves X" or "prevents Y".

## File tree

```
drug-shortage-ews/
  run.py                 end-to-end entry point
  config.json            window, thresholds, seeds, split date
  requirements.txt       pinned dependencies
  README.md
  DATA_LINEAGE.md        sources, pull behavior, record counts, limitations
  LIMITATIONS.md
  src/
    __init__.py
    util.py              UTF-8 I/O, seeding, deterministic hashing
    config.py            typed config loader
    ingest.py            real pulls + cache + synthetic fallback + lineage
    synthetic.py         labeled synthetic data-generating process
    features.py          point-in-time feature build (records max_used_date)
    leakage.py           leakage self-check (fails the run)
    baselines.py         base-rate and persistence baselines
    model.py             logistic regression (headline) + GBT + explanations
    metrics.py           PR-AUC, confusion, PR curve, calibration, thresholds
    backtest.py          temporal backtest + lead-time distribution
    holdout_guard.py     structural seal on the ground-truth holdout
    final_eval.py        the only reader of the sealed holdout
    narrate.py           narration (Anthropic -> OpenAI -> deterministic stub)
    report.py            HTML dashboard (three views, inline SVG charts)
  data/                  raw snapshots + sealed holdout (created at run time)
  output/                metrics.json, run_lineage.json, index.html
  tests/
    test_leakage.py      leakage + holdout-seal enforcement
```
