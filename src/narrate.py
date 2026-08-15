"""
narrate.py
Optional narration layer. Turns a drug's top feature contributions into a plain-
language rationale. NARRATION ONLY: it explains why the model ranked a drug where
it did. It never makes or implies a decision, and it never introduces a fact that
is not already in the contributions passed to it.

Fallback chain:
  1) Anthropic  (if ANTHROPIC_API_KEY set)
  2) OpenAI     (if OPENAI_API_KEY set)
  3) Deterministic template stub (always available, no network, no key)

The deterministic stub is the default so the pipeline runs fully offline and two
runs stay byte-identical (LLM outputs are non-deterministic and are therefore
excluded from the metrics that must be reproducible; narration is presentation
only).
"""
from __future__ import annotations

import os

from .util import log

_FEATURE_PHRASES = {
    "prior_shortage_count": "a history of prior shortages",
    "has_prior_shortage": "at least one prior shortage on record",
    "days_since_last_shortage": "recency of its last shortage",
    "recent_shortage_count_365": "shortages within the trailing 12 months",
    "distinct_manufacturers": "the number of distinct suppliers seen",
    "single_supplier_flag": "single-supplier dependence",
    "manufacturer_hhi": "supplier concentration",
    "disc_count": "prior discontinuation notices",
    "days_since_last_disc": "recency of a discontinuation notice",
    "recent_disc_flag_120": "a discontinuation notice in the last 120 days",
    "category_base_rate_pit": "its therapeutic category's historical shortage rate",
}


def _stub(drug_name: str, contribs: list[dict]) -> str:
    raises = [c for c in contribs if c["direction"] == "raises"][:3]
    lowers = [c for c in contribs if c["direction"] == "lowers"][:1]
    up = "; ".join(_FEATURE_PHRASES.get(c["feature"], c["feature"]) for c in raises)
    text = (f"{drug_name} is ranked at elevated risk mainly due to {up}."
            if up else f"{drug_name} shows no strong upward risk drivers.")
    if lowers:
        d = lowers[0]
        text += (f" Partially offsetting: "
                 f"{_FEATURE_PHRASES.get(d['feature'], d['feature'])}.")
    text += (" This is a retrospective, model-proposed ranking for human review, "
             "not a procurement or clinical decision.")
    return text


def _anthropic(drug_name: str, contribs: list[dict]) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        drivers = ", ".join(f"{c['feature']} ({c['direction']} risk)"
                            for c in contribs)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=160,
            messages=[{"role": "user", "content": (
                "Write 2 plain sentences explaining why a drug is ranked at "
                "elevated shortage risk, using ONLY these model drivers. Do not "
                "invent facts. Do not recommend any action. Drivers: "
                f"{drivers}. Drug: {drug_name}.")}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception as exc:  # noqa: BLE001
        log(f"anthropic narration failed, falling back: {exc}")
        return None


def _openai(drug_name: str, contribs: list[dict]) -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        drivers = ", ".join(f"{c['feature']} ({c['direction']} risk)"
                            for c in contribs)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=160,
            messages=[{"role": "user", "content": (
                "Write 2 plain sentences explaining why a drug is ranked at "
                "elevated shortage risk, using ONLY these model drivers. Do not "
                "invent facts. Do not recommend any action. Drivers: "
                f"{drivers}. Drug: {drug_name}.")}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        log(f"openai narration failed, falling back: {exc}")
        return None


def narrate(drug_name: str, contribs: list[dict], mode: str = "auto") -> dict:
    """Return {'text', 'engine'}. engine in {anthropic, openai, stub}."""
    if mode == "off":
        return {"text": _stub(drug_name, contribs), "engine": "stub"}
    for fn, name in ((_anthropic, "anthropic"), (_openai, "openai")):
        text = fn(drug_name, contribs)
        if text:
            return {"text": text, "engine": name}
    return {"text": _stub(drug_name, contribs), "engine": "stub"}
