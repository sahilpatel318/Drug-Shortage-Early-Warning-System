"""
Dashboard server.

Reads artifacts/report.json (produced by run.py) and renders a single page.
If the artifact is missing, it shows a clear instruction rather than crashing.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ews import config
from . import charts

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Drug Shortage Early Warning System")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

REPORT_PATH = config.ARTIFACT_DIR / "report.json"

CATEGORY_LABELS = {
    "cat_Oncology": "Oncology", "cat_Anti-infective": "Anti-infective",
    "cat_Cardiovascular": "Cardiovascular", "cat_CNS": "CNS",
    "cat_Endocrine": "Endocrine", "cat_Analgesic": "Analgesic",
    "cat_Respiratory": "Respiratory", "cat_Immunology": "Immunology",
}


def _load_report():
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok" if REPORT_PATH.exists() else "no-report"


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    report = _load_report()
    if report is None:
        return templates.TemplateResponse(
            "missing.html", {"request": request}, status_code=503
        )

    ds = report["dataset"]
    pr = report["pr_auc"]
    lead = report["lead_time"]
    surv = report["survival"]

    detection_pct = round(lead["flag_rate"] * 100, 1)
    high = surv["groups"].get("high", {})

    svgs = {
        "pr": charts.pr_curve_svg(report["pr_curves"], ds["test_prevalence"], pr),
        "coef": charts.coef_svg(report["coefficients"]),
        "lead": charts.leadtime_svg(lead["lead_days_hist"], lead["median_lead_days"]),
        "survival": charts.survival_svg(surv),
    }

    category_effects = sorted(
        [c for c in report["coefficients"] if c["feature"].startswith("cat_")],
        key=lambda c: c["coef"], reverse=True,
    )
    for c in category_effects:
        c["label"] = CATEGORY_LABELS.get(c["feature"], c["feature"])

    ctx = {
        "request": request,
        "report": report,
        "meta": report["meta"],
        "cfg": report["config"],
        "ds": ds,
        "pr": pr,
        "clf": report["classification_at_threshold"],
        "lead": lead,
        "surv": surv,
        "high": high,
        "detection_pct": detection_pct,
        "svgs": svgs,
        "watchlist": report["watchlist"],
        "category_effects": category_effects,
        "firewall": report["firewall"],
    }
    return templates.TemplateResponse("dashboard.html", ctx)
