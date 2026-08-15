#!/usr/bin/env python3
"""
run.py
End-to-end entry point for the Drug-Shortage Early-Warning PoC.

Pipeline:
  ingest -> point-in-time features -> LEAKAGE SELF-CHECK (hard fail) ->
  temporal backtest (model vs baselines, lead-time) ->
  sealed-holdout final evaluation -> deterministic metrics.json -> HTML dashboard.

Usage:
  python run.py                      # run once, write output/
  python run.py --config config.json
  python run.py --verify-determinism # run twice, assert byte-identical metrics
  python run.py --data-mode synthetic|auto|real
"""
from __future__ import annotations

import argparse
import os
import sys

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (backtest, features, final_eval, ingest, leakage, report)  # noqa: E402
from src.config import load_config, Config  # noqa: E402
from src.util import (log, round_floats, seed_everything, sha256_of_obj,  # noqa: E402
                      utc_now_iso, write_json)

LIMITATIONS = [
    "Retrospective signal-recovery PoC on open data. NOT a deployed predictor and "
    "NOT deployment-ready.",
    "No dollar-savings or 'shortages prevented' claims are made anywhere. The tool "
    "flags and ranks for human review; it issues no verdict.",
    "Point-in-time fidelity is bounded by the sources' posting dates. If a source "
    "backfills or revises records, true point-in-time state may differ from what is "
    "reconstructable here.",
    "Prediction unit is the generic drug; strengths and dosage forms collapse to "
    "one drug. Manufacturer-level dynamics are only partially captured.",
    "Cells where a drug is already in an active shortage are excluded (you cannot "
    "'enter' a shortage you are in); this narrows the population modelled.",
    "When run on the synthetic fallback, all metrics demonstrate pipeline signal-"
    "recovery on a labeled generator ONLY. They say nothing about real-world "
    "performance.",
]

INTERVIEW_HONESTY = (
    "True scope: retrospective signal recovery on open drug-shortage data. It "
    "demonstrates temporal integrity (point-in-time features, time-based split, a "
    "leakage self-check that fails the run), honest baselines, imbalance-aware "
    "metrics, calibrated and explainable scores, and a lead-time backtest. It is "
    "explicitly NOT a production predictor, and on the synthetic fallback the "
    "numbers show only that the pipeline recovers planted signal."
)


def run_pipeline(cfg: Config, build_html: bool = True) -> dict:
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    ing = ingest.ingest(cfg)
    reports, lineage = ing["reports"], ing["lineage"]

    panel = features.build_panel(reports, cfg)
    leak_report = leakage.assert_no_leakage(panel, cfg)

    bt = backtest.temporal_backtest(panel, reports, cfg)
    results, model, scored_test = bt["results"], bt["model"], bt["scored_test"]

    holdout = final_eval.unseal_and_evaluate(model, cfg)

    # ---- deterministic metrics artifact (this is what must be byte-identical) ---
    # NOTE: the data-feed provenance label (synthetic / real / cache) is
    # deliberately NOT in this object. It legitimately differs between a fresh
    # generate run and a cache-reuse run, and it is not a modelling metric. It is
    # recorded in run_lineage.json and shown on the dashboard banner instead, so
    # metrics.json stays byte-identical across runs. If the underlying DATA ever
    # changes (e.g. synthetic -> real), the results below change and the hash
    # catches it.
    metrics_obj = round_floats({
        "config": {
            "forward_window_days": cfg.forward_window_days,
            "seed": cfg.seed,
            "split_date": cfg.split_date,
            "threshold_policy": cfg.threshold_policy,
        },
        "leakage_self_check": leak_report,
        "results": results,
        "holdout": {k: v for k, v in holdout.items()},
    })
    metrics_path = os.path.join(cfg.out_dir, "metrics.json")
    write_json(metrics_path, metrics_obj)
    metrics_hash = sha256_of_obj(metrics_obj)

    _src = lineage.get("resolved_source")
    _under = (lineage.get("cached_source") if _src == "cache" else _src)
    run_meta = {
        "build_stamp": f"build {utc_now_iso()} | seed {cfg.seed} | "
                       f"source {_under}"
                       f"{' (cached)' if _src == 'cache' else ''} | "
                       f"window {cfg.forward_window_days}d",
        "limitations": LIMITATIONS,
        "interview_honesty": INTERVIEW_HONESTY,
        "holdout": holdout,
        "metrics_sha256": metrics_hash,
    }
    write_json(os.path.join(cfg.out_dir, "run_lineage.json"), {
        "generated_at": utc_now_iso(),
        "data_source": lineage.get("resolved_source"),
        "ingest_lineage": lineage,
        "metrics_sha256": metrics_hash,
    })

    if build_html:
        out_html = report.build_report(cfg, lineage, results, model,
                                       scored_test, reports, run_meta)
        log(f"dashboard: {out_html}")

    log(f"metrics.json sha256 = {metrics_hash}")
    return {"metrics_hash": metrics_hash, "metrics": metrics_obj,
            "lineage": lineage}


def verify_determinism(cfg: Config) -> None:
    log("determinism check: run 1/2")
    a = run_pipeline(cfg, build_html=False)
    log("determinism check: run 2/2")
    b = run_pipeline(cfg, build_html=False)
    if a["metrics_hash"] == b["metrics_hash"]:
        log(f"DETERMINISM OK: both runs -> {a['metrics_hash']}")
        print(f"DETERMINISM OK {a['metrics_hash']}")
    else:
        log(f"DETERMINISM FAIL: {a['metrics_hash']} != {b['metrics_hash']}")
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Drug-Shortage Early-Warning PoC")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data-mode", choices=["auto", "real", "synthetic"],
                    default=None)
    ap.add_argument("--verify-determinism", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_mode:
        cfg = Config(**{**cfg.__dict__, "data_mode": args.data_mode})

    if args.verify_determinism:
        verify_determinism(cfg)
        return

    out = run_pipeline(cfg)
    src = out["lineage"].get("resolved_source")
    print(f"OK source={src} metrics_sha256={out['metrics_hash']}")
    print(f"Open {os.path.join(cfg.out_dir, 'index.html')} in a browser.")


if __name__ == "__main__":
    main()
