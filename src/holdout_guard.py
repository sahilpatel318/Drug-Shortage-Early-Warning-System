"""
holdout_guard.py
Structural enforcement that the sealed ground-truth holdout is NOT read by the
training or scoring pipeline. The guard is a process-global latch that starts
locked. Only final_eval.unseal_and_evaluate() is permitted to open it, and it
does so for the duration of the final evaluation only.

Any attempt to load the holdout while the latch is locked raises HoldoutViolation
and hard-fails the run. This is deliberately blunt: leakage should crash, not warn.

Two independent barriers:
  1) This runtime latch (below).
  2) Filesystem separation: the holdout lives in data/holdout/ and NO module
     other than final_eval.py references that directory. (Enforced by a grep-based
     test in tests/test_leakage.py.)
"""
from __future__ import annotations

import os

from .util import log, read_jsonl


class HoldoutViolation(RuntimeError):
    pass


_UNSEALED = False


def _seal() -> None:
    global _UNSEALED
    _UNSEALED = False


def _unseal() -> None:
    global _UNSEALED
    _UNSEALED = True


def is_sealed() -> bool:
    return not _UNSEALED


def load_holdout(holdout_dir: str) -> list[dict]:
    """Load sealed holdout reports. Raises unless the latch has been opened by
    final_eval. Never call this from feature/model/backtest code."""
    if not _UNSEALED:
        raise HoldoutViolation(
            "Attempt to read the sealed holdout before final evaluation. "
            "The training/scoring pipeline is structurally forbidden from "
            "touching data/holdout/. This run has been failed on purpose."
        )
    path = os.path.join(holdout_dir, "holdout_reports.jsonl")
    log(f"holdout unsealed for final evaluation: {path}")
    return read_jsonl(path)
