"""
final_eval.py
The ONLY module permitted to open the sealed holdout. It unseals, builds
point-in-time features for the held-out drugs, scores them with the ALREADY-fitted
model, measures PR-AUC and lead-time on truly unseen entities, then re-seals.

This is the planted ground-truth check: the training/scoring pipeline never saw
these drugs. If the model recovers signal here, it is not merely overfitting the
working set.
"""
from __future__ import annotations

import pandas as pd

from . import backtest, holdout_guard, metrics
from .features import build_panel
from .util import log


def unseal_and_evaluate(model, cfg) -> dict:
    holdout_guard._unseal()
    try:
        reports = holdout_guard.load_holdout(cfg.holdout_dir)
        if not reports:
            return {"status": "empty", "note": "No holdout reports present."}
        panel = build_panel(reports, cfg)
        # Score the WHOLE holdout panel point-in-time (all rows are out-of-sample:
        # these drugs were never in training).
        scores = model.predict_proba(panel)
        panel = panel.copy()
        panel["score"] = scores
        y = panel["label"].to_numpy(int)

        threshold = cfg_threshold(model, cfg, panel)
        block = metrics.score_block(y, scores, threshold)
        # lead time on holdout uses the same definition; treat every as_of as test.
        lead = backtest.compute_lead_times(
            reports, panel.assign(as_of=panel["as_of"]), threshold, _AllTestCfg(cfg))
        log(f"holdout eval: PR-AUC={block['pr_auc']} "
            f"lead median={lead.get('median_days')}")
        return {
            "status": "ok",
            "n_drugs": int(panel["drug_id"].nunique()),
            "n_rows": int(len(panel)),
            "threshold": round(float(threshold), 6),
            "metrics": block,
            "lead_time": lead,
        }
    finally:
        holdout_guard._seal()


class _AllTestCfg:
    """Wrapper so lead-time treats every holdout as_of as evaluable (split in the
    far past). Holdout drugs are entirely out-of-sample by construction."""
    def __init__(self, cfg):
        self._cfg = cfg
        self.split_date = "1900-01-01"

    def __getattr__(self, name):
        return getattr(self._cfg, name)


def cfg_threshold(model, cfg, panel: pd.DataFrame) -> float:
    # Reuse the operating threshold policy on the holdout's own scores would be
    # circular; instead use a fixed, documented operating point: the ELEVATED tier
    # cut. This keeps the holdout check independent of holdout labels.
    return float(cfg.tier_cuts["ELEVATED"])
