# LIMITATIONS

Limitations are treated as features of an honest PoC, not as things to hide. They are
also printed on the dashboard's validation page.

## Scope

- This is a **retrospective signal-recovery** PoC on open data. It is **not** a
  deployed or deployment-ready predictor.
- It makes **no** dollar-savings claims and **no** "shortages prevented" claims. The
  tool flags and ranks for human review; it issues no verdict.
- On the synthetic fallback, every metric demonstrates that the pipeline recovers
  planted signal from a labeled generator. Synthetic metrics say **nothing** about
  real-world performance.

## Data

- Point-in-time fidelity is bounded by the sources' posting dates. If a source
  backfills or revises records, the reconstructable point-in-time state can differ
  from what a real-time observer would have seen. Revision history is not
  reconstructed and is never imputed.
- The prediction unit is the generic drug. Strengths and dosage forms collapse into
  one drug, so manufacturer-level and presentation-level dynamics are only partially
  captured.
- Drug Shortages Canada requires authenticated access; without credentials that
  source is omitted, which narrows coverage. openFDA coverage and field semantics
  can change; the ingestion mapping should be verified on the first live run.

## Labeling and population

- The label is a new shortage onset within the forward window. Cells where a drug is
  already in an active shortage at the cutoff are excluded, which narrows the modeled
  population and means the tool speaks to onset risk, not to ongoing-shortage
  severity or duration.
- Cutoffs whose forward window would extend past the observed data end are dropped
  (right-censoring). This avoids confusing "no onset" with "not yet observable" but
  reduces the number of evaluable cutoffs near the data boundary.

## Modeling

- The logistic model uses correlated features (for example `single_supplier_flag`
  and `manufacturer_hhi`). Under collinearity, individual coefficients can split an
  effect and even flip sign, so per-feature contributions should be read as the
  model's decomposition, not as isolated causal effects. The direction of the
  overall ranking remains interpretable.
- Probabilities are raw logistic outputs. Calibration is reported honestly and is
  noisier in sparsely populated high-probability bins. Treat high-confidence scores
  with appropriate caution.
- The operating threshold trades precision for recall on an imbalanced problem.
  Precision at the default threshold is modest by design; the intended use is
  triage and ranking for human review, not automated action.

## Reproducibility

- `metrics.json` is byte-identical across runs against a fixed data snapshot.
  Determinism is guaranteed against the cached raw snapshot; a fresh network pull can
  legitimately differ if the upstream sources change between pulls.
- Narration text is excluded from the reproducibility guarantee because
  language-model outputs are not deterministic. The reproducible artifact
  (`metrics.json`) contains no narration.

## Honest interpretation of the headline

The headline is lead time on caught onsets: on the synthetic reference run, a median
of 82 days of warning for the onsets the model flagged, at a detection rate of 0.57.
This is a statement about signal recovery on synthetic data at a chosen threshold. It
is not a claim that the system would achieve this on real data, nor that acting on it
would prevent any shortage.
