"""
config.py
Loads config.json into a typed object. Central place for the forward window,
thresholds, seeds, split date, and paths. No magic numbers scattered in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .util import read_json


@dataclass(frozen=True)
class Config:
    # Core modelling knobs -------------------------------------------------- #
    forward_window_days: int = 90        # predict shortage onset within this horizon
    seed: int = 20260810                 # single global seed
    panel_freq: str = "ME"               # month-end grid of as-of dates (pandas offset)
    # Temporal split: train on as-of dates strictly before this date; test after.
    split_date: str = "2023-01-31"
    # Threshold policy: choose the flag threshold on TRAIN, apply to TEST.
    threshold_policy: str = "max_f1"     # "max_f1" | "target_recall"
    target_recall: float = 0.60          # used only when policy == target_recall
    # Risk tier cut points on predicted probability (glyph + label in the UI).
    tier_cuts: dict = field(default_factory=lambda: {
        "HIGH": 0.50, "ELEVATED": 0.25, "WATCH": 0.10  # below WATCH => LOW
    })

    # Data mode ------------------------------------------------------------- #
    # "auto"  -> try real APIs, fall back to synthetic if unreachable
    # "real"  -> real APIs only (fail loudly if unreachable)
    # "synthetic" -> never touch the network
    data_mode: str = "auto"
    use_cache: bool = True               # reuse cached raw snapshot for determinism
    synthetic_n_working_drugs: int = 220
    synthetic_n_holdout_drugs: int = 40
    synthetic_start: str = "2018-01-01"
    synthetic_end: str = "2024-12-31"

    # Narration ------------------------------------------------------------- #
    narration: str = "auto"              # "auto" | "off"  (auto -> anthropic/openai/stub)

    # Paths ----------------------------------------------------------------- #
    out_dir: str = "output"
    data_dir: str = "data"

    # Derived paths --------------------------------------------------------- #
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def working_dir(self) -> str:
        return os.path.join(self.data_dir, "working")

    @property
    def holdout_dir(self) -> str:
        return os.path.join(self.data_dir, "holdout")


def load_config(path: str | None = None) -> Config:
    if path is None or not os.path.exists(path):
        return Config()
    raw: dict[str, Any] = read_json(path)
    # Only accept known keys; ignore unknowns so a stray key never crashes a run.
    known = {f for f in Config.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in known}
    return Config(**filtered)
