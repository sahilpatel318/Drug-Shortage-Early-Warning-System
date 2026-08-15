"""
test_leakage.py
Independent checks that back up the in-pipeline self-check.

Run: python -m pytest -q   (or)   python tests/test_leakage.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import holdout_guard
from src.config import load_config
from src.holdout_guard import HoldoutViolation


def test_holdout_sealed_by_default():
    assert holdout_guard.is_sealed() is True
    raised = False
    try:
        holdout_guard.load_holdout("data/holdout")
    except HoldoutViolation:
        raised = True
    assert raised, "Reading the holdout while sealed must raise HoldoutViolation."


def test_modeling_modules_never_read_holdout():
    """Structural separation. Writing the sealed holdout during ingestion is fine
    (it happens before any training). What must never happen is a MODELLING /
    SCORING module READING it. We enforce two things:
      1) The modelling modules do not reference the holdout at all.
      2) load_holdout() is INVOKED only in final_eval.py (defined in holdout_guard).
    """
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "src")
    modelling = {"features.py", "model.py", "backtest.py", "metrics.py",
                 "baselines.py", "report.py", "leakage.py"}
    offenders = []
    callers = []
    for fn in os.listdir(src_dir):
        if not fn.endswith(".py"):
            continue
        text = open(os.path.join(src_dir, fn), encoding="utf-8").read()
        # data-ACCESS patterns only. Rendering an already-computed holdout metrics
        # dict passed in from final_eval is fine; READING the holdout data is not.
        if fn in modelling and re.search(
                r"load_holdout\(|holdout_dir|data/holdout|holdout_reports", text):
            offenders.append(fn)
        # a call site looks like "load_holdout(" ; the definition lives in
        # holdout_guard.py and is allowed there.
        if fn != "holdout_guard.py" and "load_holdout(" in text:
            callers.append(fn)
    assert not offenders, f"Modelling modules reference the holdout: {offenders}"
    assert callers == ["final_eval.py"], (
        f"load_holdout() must be called only in final_eval.py, found: {callers}")


def test_point_in_time_after_build():
    """If a panel exists, every row's max_used_date must be <= its as_of."""
    import pandas as pd
    from src import features, ingest, leakage
    cfg = load_config("config.json")
    reports = ingest.ingest(cfg)["reports"]
    panel = features.build_panel(reports, cfg)
    used = pd.to_datetime(panel["max_used_date"])
    asof = pd.to_datetime(panel["as_of"])
    assert (used <= asof).all(), "Point-in-time violation detected."
    leakage.assert_no_leakage(panel, cfg)


if __name__ == "__main__":
    test_holdout_sealed_by_default()
    test_modeling_modules_never_read_holdout()
    test_point_in_time_after_build()
    print("ALL LEAKAGE TESTS PASSED")
