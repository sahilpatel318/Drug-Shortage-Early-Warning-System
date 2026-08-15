"""
Single command entry point.

    python run.py                 run the deterministic synthetic study
    python run.py --fetch-real    download the real shortage record (needs net)

The synthetic run writes artifacts/report.json, which the dashboard reads.
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from ews import config, pipeline


def _print_summary(report: dict) -> None:
    d = report["dataset"]
    pr = report["pr_auc"]
    lead = report["lead_time"]
    clf = report["classification_at_threshold"]
    fw = report["firewall"]

    print("=" * 66)
    print("DRUG SHORTAGE EARLY WARNING SYSTEM")
    print("PROVENANCE: SYNTHETIC (signal recovery on planted data)")
    print("=" * 66)
    print(f"firewall  features independent of future : {fw['features_independent_of_future']}")
    print(f"firewall  labels react to future scramble: {fw['labels_depend_on_future']}")
    print(f"          rows checked                    : {fw['n_rows']}")
    print("-" * 66)
    print(f"train rows {d['train_rows']:>6}  positives {d['train_positives']:>5}"
          f"  prevalence {d['train_prevalence']:.4f}")
    print(f"test  rows {d['test_rows']:>6}  positives {d['test_positives']:>5}"
          f"  prevalence {d['test_prevalence']:.4f}")
    print("-" * 66)
    print(f"PR-AUC  logistic regression : {pr['logistic_regression']:.4f}"
          f"   (n={d['test_rows']}, pos={d['test_positives']})")
    print(f"PR-AUC  best single feature : {pr['best_single_feature']:.4f}"
          f"   ({report['best_single_feature_name']})")
    print(f"PR-AUC  prevalence baseline : {pr['prevalence']:.4f}")
    print(f"PR-AUC  random baseline     : {pr['random']:.4f}")
    print(f"lift over prevalence        : {report['pr_auc_lift_over_prevalence']:+.4f}")
    print("-" * 66)
    print(f"threshold (max F1 on train) : {clf['threshold']:.4f}")
    print(f"precision {clf['precision']:.3f}  recall {clf['recall']:.3f}"
          f"  f1 {clf['f1']:.3f}  (tp={clf['tp']} fp={clf['fp']} fn={clf['fn']})")
    print("-" * 66)
    print(f"test period shortages       : {lead['test_onsets_total']}")
    print(f"flagged before onset        : {lead['flagged_ahead']}"
          f"  ({lead['flag_rate']*100:.1f} percent)")
    print(f"missed                      : {lead['missed']}")
    if lead["median_lead_days"] is not None:
        print(f"median lead time            : {lead['median_lead_days']:.1f} days"
              f"  (IQR {lead['p25_lead_days']:.0f} to {lead['p75_lead_days']:.0f})")
    print("-" * 66)
    print(f"determinism hash            : {report['determinism_hash'][:16]}...")
    print(f"report written              : artifacts/report.json")
    print("=" * 66)


def _fetch_real() -> int:
    from ews import sources
    print("Attempting to reconstruct the real shortage record from openFDA ...")
    try:
        events = sources.fetch_openfda_shortages()
    except sources.SourceUnavailable as exc:
        print(f"\nCould not reach openFDA: {exc}")
        print("The default synthetic run works offline: python run.py")
        return 2
    record = sources.reconstruct_shortage_record(events)
    summary = sources.summarize_record(record)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DATA_DIR / "real_shortage_record.csv"
    record.to_csv(out, index=False)
    print("Reconstructed real shortage record:")
    for k, v in summary.items():
        print(f"  {k:16}: {v}")
    print(f"  saved to        : {out}")
    print("\nNote: supplier count and manufacturer concentration are not in this")
    print("feed. Full real data modeling needs a manufacturer source join, which")
    print("is documented in the README. Reported model metrics remain synthetic.")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Drug Shortage Early Warning System")
    parser.add_argument("--fetch-real", action="store_true",
                        help="download the real shortage record from openFDA")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    if args.fetch_real:
        return _fetch_real()

    report = pipeline.run(seed=args.seed)
    pipeline.write_report(report)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
