"""
util.py
Shared utilities: deterministic seeding, UTF-8-pinned file I/O, stable hashing,
and a tiny logger. Every file read/write in this project routes through here so
that encoding (UTF-8) and newline behaviour are consistent on Windows 11 and Linux.

Honesty note: nothing in this module fabricates or imputes data. It only moves
bytes and seeds randomness.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

UTF8 = "utf-8"


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed all sources of randomness we use. sklearn draws from numpy/random
    via explicit random_state, so seeding these two covers the pipeline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Logging (stderr, so stdout stays clean for any piped output)
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    sys.stderr.write(f"[ews] {msg}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# File I/O (UTF-8 pinned everywhere)
# --------------------------------------------------------------------------- #
def read_json(path: str) -> Any:
    with open(path, "r", encoding=UTF8) as fh:
        return json.load(fh)


def write_json(path: str, obj: Any, *, sort_keys: bool = True) -> None:
    """Deterministic JSON: sorted keys, fixed separators, trailing newline,
    LF newlines regardless of platform (newline='' + explicit \\n)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=sort_keys)
    with open(path, "w", encoding=UTF8, newline="") as fh:
        fh.write(text)
        fh.write("\n")


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding=UTF8) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding=UTF8, newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding=UTF8, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_text(path: str) -> str:
    with open(path, "r", encoding=UTF8) as fh:
        return fh.read()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding=UTF8, newline="") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# Stable hashing (used to prove two runs are byte-identical)
# --------------------------------------------------------------------------- #
def sha256_of_obj(obj: Any) -> str:
    """Hash of a canonical JSON encoding. Order-independent for dict keys."""
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode(UTF8)).hexdigest()


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats so metric files are byte-identical across runs
    despite platform-level float formatting differences."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, (np.floating,)):
        return round(float(obj), ndigits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
