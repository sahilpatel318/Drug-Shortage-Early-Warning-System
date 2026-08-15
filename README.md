# Drug Shortage Early Warning System

A proof of concept that scores which medicines are drifting toward a supply
shortage before the official declaration. It reconstructs a monthly supply
history, engineers supply chain risk features, trains an early warning model,
and measures how far ahead the score would have flagged shortages that later
occurred.

## Honesty first

This build runs on **synthetic data** by default. The relationship between the
features and the shortages is planted by the generator, so the reported numbers
are **signal recovery on synthetic data**, not a claim about real medicines. The
provenance is stated on the dashboard, in the report artifact, and in the run
output. The interview framing is exactly that sentence.

The two genuinely public sources named in the brief, Drug Shortages Canada and
openFDA, are wired as real adapters (`ews/sources.py`). On a machine with
internet access, `python run.py --fetch-real` downloads and reconstructs the
real historical shortage record. Those feeds carry shortage events, not supplier
counts or manufacturer market share, so two of the strongest features here need
a manufacturer source join before a full real data model. That gap is documented
rather than hidden.

## What it does, in one pass

1. Generate a deterministic monthly panel of drugs with supplier structure,
   manufacturer concentration, therapeutic category, and shortage onsets drawn
   from a hazard that depends on those factors plus noise (`ews/synth.py`).
2. Seal the onset labels in a firewalled vault and prove the feature path cannot
   read the future (`ews/firewall.py`).
3. Engineer leakage safe temporal features from the observable panel only
   (`ews/features.py`).
4. Split out of time, train a logistic early warning model, and compare it to
   honest baselines (`ews/model.py`).
5. Measure PR-AUC against baselines, precision and recall at a training chosen
   threshold, lead time before onset, and Kaplan Meier time to next shortage by
   risk group (`ews/evaluate.py`).
6. Write `artifacts/report.json` and render the dashboard (`app/`).

## Quickstart

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python run.py                 # runs the deterministic study, writes artifacts/report.json
uvicorn app.main:app --reload # serve the dashboard at http://127.0.0.1:8000
```

Optional real data fetch (needs internet):

```bash
python run.py --fetch-real    # reconstructs the real shortage record from openFDA
```

## Metrics truth bank

These come from the default seeded run (`SEED = 20260813`). They regenerate
byte for byte, so any figure below can be reproduced by running the code. Sizes
are printed beside every number because the task is a rare event.

| Measure | Value | Notes |
| --- | --- | --- |
| Test prevalence (base rate) | 0.144 | 724 positives in 5028 at risk drug months |
| PR-AUC, logistic model | 0.385 | headline, rare event ranking metric |
| PR-AUC, best single feature | 0.338 | feature: hhi |
| PR-AUC, prevalence baseline | 0.144 | no skill |
| PR-AUC, random baseline | 0.145 | sanity floor |
| Lift over prevalence | +0.241 | logistic minus no skill |
| Precision at F1 threshold | 0.402 | threshold 0.159, chosen on train |
| Recall at F1 threshold | 0.598 | tp 433, fp 643, fn 291 |
| Shortages flagged before onset | 64.8% | 127 of 196 test shortages |
| Missed shortages | 69 | counted openly |
| Median lead time | 304 days | IQR 183 to 365, one year watch window |
| High risk time to shortage | 14 months | top tercile; lower groups stay above 50% |
| Top model driver | single_supplier | standardized coefficient about +0.52 |

Headline metric is PR-AUC, not accuracy, because a rare event classifier can
look excellent on accuracy while being useless. The logistic model clearly beats
the strongest single feature and the base rate, which is the honest claim.

## The firewall

The onset labels are the answer key. They live in a sealed `Vault`. The feature
builder never receives them; it reads only the observable panel (supplier
counts, concentration, and shortages that already happened and are therefore
public record). The label for a row anchored at month t comes from the vault and
describes a window strictly after t.

The runtime leakage test builds the feature matrix twice, scrambling every onset
strictly after each row's anchor the second time. If any feature had peeked into
the future, its value would move. The test asserts the feature columns are byte
identical across both builds and that the labels do move. This runs on every
pipeline execution and in the test suite.

## Reproducibility

Everything is seeded from `ews/config.py`. Two runs with the same seed produce an
identical `determinism_hash`. Change the seed and the whole synthetic history
changes, but any given seed reproduces exactly.

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers determinism, the leakage firewall (including a deliberately leaky builder
that must be caught), feature integrity (at risk anchors, no feature equal to the
label, fully observed label windows), and model quality (logistic beats every
baseline, valid operating point).

## Design system

The dashboard uses a single typeface, Lora, with contrast from weight and
italic, and an aubergine and chartreuse palette where chartreuse is reserved for
one focal point per view. Charts are server side inline SVG with no CDN
dependency, colored from the palette.

## Limitations

Stated on the dashboard and worth repeating: the run is synthetic; the score is a
persistent risk register rather than a countdown timer, so lead time skews toward
the top of the watch window; the real feeds lack supplier and concentration data;
first ever shortages with no supply deterioration are undetectable by design and
are counted as misses; and precision is modest by choice at the F1 optimal
threshold, with the full precision recall curve available for stricter operating
points.

## Project layout

```
drug-shortage-ews/
  run.py                 single command entry point
  ews/
    config.py            seeds, horizons, hazard, all tunables
    synth.py             synthetic panel and sealed onsets
    firewall.py          vault and the runtime leakage test
    features.py          leakage safe temporal features
    model.py             logistic model and honest baselines
    evaluate.py          PR-AUC, lead time, Kaplan Meier survival
    sources.py           openFDA and Drug Shortages Canada adapters
    pipeline.py          orchestration and report writing
  app/
    main.py              FastAPI dashboard server
    charts.py            inline SVG chart builders
    templates/           dashboard and fallback pages
    static/styles.css    locked design tokens
  tests/                 determinism, leakage, features, model
  artifacts/report.json  sample output from the default run
  requirements.txt
  .env.example
```
