"""
report.py
Static HTML dashboard generator. No external chart libraries: PR curve,
calibration curve and lead-time histogram are hand-drawn inline SVG so the report
is dense, dependency-free, offline-capable, and under our own aesthetic control.

Design system (non-negotiable, enforced here):
  * Aesthetic anchors: Bloomberg Terminal + Linear. Dense, table-first.
  * Typography: IBM Plex Sans / IBM Plex Mono (with offline-safe fallback stacks).
  * Palette: slate on warm paper.
  * Status by GLYPH + LABEL, never colour alone.
  * Forbidden and absent: hero banners, gradients, large rounded cards, drop
    shadows, emoji headings, three-KPI-tile rows, chatbot widgets, template look.
  * Three views: (a) ranked risk table, (b) per-drug drill-down, (c) validation.
"""
from __future__ import annotations

import html
import os
from datetime import date

import pandas as pd

from . import narrate
from .util import write_text

TIER_GLYPH = {"HIGH": "\u25B2", "ELEVATED": "\u25C6",
              "WATCH": "\u25CF", "LOW": "\u25CB"}  # ▲ ◆ ● ○


def _tier(score: float, cuts: dict) -> str:
    if score >= cuts["HIGH"]:
        return "HIGH"
    if score >= cuts["ELEVATED"]:
        return "ELEVATED"
    if score >= cuts["WATCH"]:
        return "WATCH"
    return "LOW"


def _esc(s) -> str:
    return html.escape(str(s))


# --------------------------------------------------------------------------- #
# SVG primitives
# --------------------------------------------------------------------------- #
def _svg_frame(w, h, inner):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="var(--mono)">{inner}</svg>')


def _axes(x0, y0, x1, y1):
    return (f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" '
            f'stroke="var(--rule)" stroke-width="1"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" '
            f'stroke="var(--rule)" stroke-width="1"/>')


def pr_curve_svg(model_pts, pers_pts, base_rate):
    W, H = 520, 300
    x0, y0, x1, y1 = 46, 20, 500, 260
    def sx(r): return x0 + r * (x1 - x0)
    def sy(p): return y1 - p * (y1 - y0)
    def path(pts):
        if not pts:
            return ""
        d = "M " + " L ".join(f"{sx(p['recall']):.1f},{sy(p['precision']):.1f}"
                              for p in pts)
        return d
    inner = _axes(x0, y0, x1, y1)
    # base-rate reference line
    inner += (f'<line x1="{x0}" y1="{sy(base_rate):.1f}" x2="{x1}" '
              f'y2="{sy(base_rate):.1f}" stroke="var(--rule)" '
              f'stroke-dasharray="3 3" stroke-width="1"/>')
    inner += (f'<path d="{path(pers_pts)}" fill="none" '
              f'stroke="var(--ink-soft)" stroke-width="1.2" '
              f'stroke-dasharray="5 3"/>')
    inner += (f'<path d="{path(model_pts)}" fill="none" '
              f'stroke="var(--ink)" stroke-width="1.8"/>')
    # ticks
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        inner += (f'<text x="{sx(t):.1f}" y="{y1+14}" font-size="9" '
                  f'fill="var(--ink-soft)" text-anchor="middle">{t:.2f}</text>')
        inner += (f'<text x="{x0-6}" y="{sy(t)+3:.1f}" font-size="9" '
                  f'fill="var(--ink-soft)" text-anchor="end">{t:.2f}</text>')
    inner += (f'<text x="{(x0+x1)/2}" y="{H-2}" font-size="10" '
              f'fill="var(--ink-soft)" text-anchor="middle">recall</text>')
    inner += (f'<text x="12" y="{(y0+y1)/2}" font-size="10" '
              f'fill="var(--ink-soft)" text-anchor="middle" '
              f'transform="rotate(-90 12 {(y0+y1)/2})">precision</text>')
    return _svg_frame(W, H, inner)


def calibration_svg(points):
    W, H = 520, 300
    x0, y0, x1, y1 = 46, 20, 500, 260
    def sx(v): return x0 + v * (x1 - x0)
    def sy(v): return y1 - v * (y1 - y0)
    inner = _axes(x0, y0, x1, y1)
    inner += (f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y0}" '
              f'stroke="var(--rule)" stroke-dasharray="3 3" stroke-width="1"/>')
    if points:
        d = "M " + " L ".join(f"{sx(p['mean_pred']):.1f},{sy(p['obs_freq']):.1f}"
                              for p in points)
        inner += f'<path d="{d}" fill="none" stroke="var(--ink)" stroke-width="1.6"/>'
        for p in points:
            inner += (f'<circle cx="{sx(p["mean_pred"]):.1f}" '
                      f'cy="{sy(p["obs_freq"]):.1f}" r="2.4" fill="var(--ink)"/>')
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        inner += (f'<text x="{sx(t):.1f}" y="{y1+14}" font-size="9" '
                  f'fill="var(--ink-soft)" text-anchor="middle">{t:.2f}</text>')
        inner += (f'<text x="{x0-6}" y="{sy(t)+3:.1f}" font-size="9" '
                  f'fill="var(--ink-soft)" text-anchor="end">{t:.2f}</text>')
    inner += (f'<text x="{(x0+x1)/2}" y="{H-2}" font-size="10" '
              f'fill="var(--ink-soft)" text-anchor="middle">mean predicted</text>')
    inner += (f'<text x="12" y="{(y0+y1)/2}" font-size="10" '
              f'fill="var(--ink-soft)" text-anchor="middle" '
              f'transform="rotate(-90 12 {(y0+y1)/2})">observed freq</text>')
    return _svg_frame(W, H, inner)


def leadtime_svg(hist):
    W, H = 520, 300
    x0, y0, x1, y1 = 46, 20, 500, 260
    if not hist:
        return _svg_frame(W, H, _axes(x0, y0, x1, y1) +
                          f'<text x="{(x0+x1)/2}" y="{(y0+y1)/2}" font-size="11" '
                          f'fill="var(--ink-soft)" text-anchor="middle">'
                          f'no onsets flagged in advance</text>')
    inner = _axes(x0, y0, x1, y1)
    maxc = max(h["count"] for h in hist) or 1
    n = len(hist)
    bw = (x1 - x0) / n
    for i, h in enumerate(hist):
        bx = x0 + i * bw
        bh = (h["count"] / maxc) * (y1 - y0)
        inner += (f'<rect x="{bx+3:.1f}" y="{y1-bh:.1f}" width="{bw-6:.1f}" '
                  f'height="{bh:.1f}" fill="none" stroke="var(--ink)" '
                  f'stroke-width="1.3"/>')
        inner += (f'<text x="{bx+bw/2:.1f}" y="{y1-bh-4:.1f}" font-size="9" '
                  f'fill="var(--ink)" text-anchor="middle">{h["count"]}</text>')
        inner += (f'<text x="{bx+bw/2:.1f}" y="{y1+14}" font-size="8" '
                  f'fill="var(--ink-soft)" text-anchor="middle">'
                  f'{h["lo"]}-{h["hi"]}</text>')
    inner += (f'<text x="{(x0+x1)/2}" y="{H-2}" font-size="10" '
              f'fill="var(--ink-soft)" text-anchor="middle">'
              f'lead time (days) before official shortage date</text>')
    return _svg_frame(W, H, inner)


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
CSS = """
:root{
  --paper:#efeae0; --panel:#f6f3ec; --ink:#23262b; --ink-soft:#5b616b;
  --rule:#c9c2b4; --rule-strong:#b0a894;
  --sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:13px;line-height:1.45}
a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--rule)}
a:hover{border-bottom-color:var(--ink)}
.wrap{max-width:1180px;margin:0 auto;padding:0 18px 60px}
.mast{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:2px solid var(--ink);padding:14px 0 8px;margin-bottom:0}
.mast h1{font-size:15px;font-weight:600;letter-spacing:.02em;margin:0;
  text-transform:uppercase;font-family:var(--mono)}
.mast .stamp{font-family:var(--mono);font-size:11px;color:var(--ink-soft)}
.banner{font-family:var(--mono);font-size:11.5px;padding:6px 0;
  border-bottom:1px solid var(--rule);color:var(--ink)}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--rule-strong);margin:0 0 16px}
.tab{font-family:var(--mono);font-size:11.5px;text-transform:uppercase;
  letter-spacing:.03em;padding:8px 14px;cursor:pointer;border:none;background:none;
  color:var(--ink-soft);border-bottom:2px solid transparent}
.tab.active{color:var(--ink);border-bottom:2px solid var(--ink)}
.view{display:none}.view.active{display:block}
h2{font-family:var(--mono);font-size:12px;text-transform:uppercase;
  letter-spacing:.04em;font-weight:600;margin:22px 0 8px;color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule);
  vertical-align:top}
th{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.03em;color:var(--ink-soft);font-weight:500;
  border-bottom:1px solid var(--rule-strong)}
td.num,th.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.glyph{font-family:var(--mono);font-weight:600}
.tier-HIGH .glyph{color:#7a2f2f}.tier-ELEVATED .glyph{color:#8a5a1a}
.tier-WATCH .glyph{color:#3f5566}.tier-LOW .glyph{color:#6b7280}
tr.drug-row:hover{background:var(--panel)}
.drivers{color:var(--ink-soft);font-size:11px}
.panel{border:1px solid var(--rule-strong);background:var(--panel);padding:12px 14px;
  margin:10px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.kv{font-family:var(--mono);font-size:11.5px}
.kv td:first-child{color:var(--ink-soft)}
.note{font-size:11.5px;color:var(--ink-soft);margin:6px 0}
.limbox{border:1px solid var(--rule-strong);padding:12px 14px;background:var(--panel);
  margin:14px 0}
.limbox h3{font-family:var(--mono);font-size:11px;text-transform:uppercase;
  margin:0 0 6px;letter-spacing:.04em}
.limbox ul{margin:0;padding-left:18px}.limbox li{margin:3px 0;font-size:12px}
.pass{font-family:var(--mono);color:#2f5d3a}.fail{font-family:var(--mono);color:#7a2f2f}
.small{font-size:11px;color:var(--ink-soft)}
.drill{border-top:1px solid var(--rule);padding:14px 0}
.drill h3{margin:0 0 4px;font-size:13px;font-family:var(--mono)}
.contrib-up{color:#7a2f2f}.contrib-down{color:#2f5d3a}
"""

JS = """
function showTab(id,btn){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  window.scrollTo(0,0);
}
function gotoDrug(anchor){
  showTab('view-drill',document.getElementById('tab-drill'));
  var el=document.getElementById(anchor);
  if(el){el.scrollIntoView({behavior:'smooth',block:'start'});}
}
"""


def _banner(lineage) -> str:
    src = lineage.get("resolved_source", "unknown")
    # When serving from cache, the honest label is the UNDERLYING source, never
    # a generic "open-data pull" (that would mislabel synthetic data).
    underlying = (lineage.get("cached_source", "unknown")
                  if src == "cache" else src)
    from_cache = " (cached)" if src == "cache" else ""

    if underlying == "synthetic":
        glyph = "\u25C6"
        label = f"DATA: SYNTHETIC{from_cache or ' (fallback)'}"
        detail = ("Real sources were unreachable at build time, so a labeled "
                  "synthetic generator was used. Figures below are signal-recovery "
                  "on synthetic data: NOT observed real-world data and NOT real-"
                  "world performance.")
    elif underlying == "real":
        glyph = "\u25CF"
        label = f"DATA: REAL{from_cache}"
        detail = ("Open-data pull from openFDA / Drug Shortages Canada. Point-in-"
                  "time reconstruction is bounded by the sources' posting dates; "
                  "see limitations.")
    else:
        glyph = "\u25CB"
        label = f"DATA: {str(underlying).upper()}{from_cache}"
        detail = ("Provenance unclear; treat figures as illustrative only.")
    recs = lineage.get("records", "?")
    return (f'<div class="banner"><span class="glyph">{glyph}</span> '
            f'{_esc(label)} &nbsp;|&nbsp; records={_esc(recs)} &nbsp;|&nbsp; '
            f'{_esc(detail)}</div>')


def _ranked_view(ranked, cfg) -> str:
    rows = ""
    for i, r in enumerate(ranked, 1):
        tier = r["tier"]
        g = TIER_GLYPH[tier]
        drivers = ", ".join([d["feature"] for d in r["contribs"]
                             if d["direction"] == "raises"][:3])
        rows += (
            f'<tr class="drug-row tier-{tier}">'
            f'<td class="num">{i}</td>'
            f'<td><span class="glyph">{g}</span> {tier}</td>'
            f'<td class="num">{r["score"]:.3f}</td>'
            f'<td><a href="javascript:void(0)" onclick="gotoDrug(\'d-{_esc(r["drug_id"])}\')">'
            f'{_esc(r["drug_name"])}</a></td>'
            f'<td>{_esc(r["category"])}</td>'
            f'<td>{"yes" if r["single_supplier"] else "no"}</td>'
            f'<td class="num">{r["prior"]}</td>'
            f'<td class="num">{r["recent_disc"]}</td>'
            f'<td class="drivers">{_esc(drivers)}</td>'
            f'</tr>')
    return (
        '<div class="view active" id="view-rank">'
        '<h2>Ranked shortage-risk table &middot; latest point-in-time cutoff</h2>'
        '<p class="note">The tool flags and ranks; it never issues a verdict. '
        'Scores are model-proposed probabilities of a new shortage onset within '
        f'{cfg.forward_window_days} days, for human review only.</p>'
        '<table><thead><tr>'
        '<th class="num">#</th><th>Risk tier</th><th class="num">Score</th>'
        '<th>Drug (generic)</th><th>Category</th><th>Single-supplier</th>'
        '<th class="num">Prior</th><th class="num">Rec.disc</th>'
        '<th>Top upward drivers</th>'
        '</tr></thead><tbody>' + rows + '</tbody></table>'
        '<p class="small">Tier cuts (on predicted probability): '
        f'HIGH&ge;{cfg.tier_cuts["HIGH"]}, ELEVATED&ge;{cfg.tier_cuts["ELEVATED"]}, '
        f'WATCH&ge;{cfg.tier_cuts["WATCH"]}, else LOW. Glyph+label, never colour '
        'alone.</p></div>')


def _drill_view(drills) -> str:
    body = ('<div class="view" id="view-drill">'
            '<h2>Per-drug drill-down &middot; point-in-time features, drivers, '
            'lead-time backtest</h2>')
    for d in drills:
        feat_rows = "".join(
            f'<tr><td>{_esc(k)}</td><td class="num">{_esc(v)}</td></tr>'
            for k, v in d["features"].items())
        contrib_rows = "".join(
            f'<tr><td>{_esc(c["feature"])}</td>'
            f'<td class="num">{_esc(c["value"])}</td>'
            f'<td class="{"contrib-up" if c["direction"]=="raises" else "contrib-down"}">'
            f'{"+" if c["contribution"]>=0 else ""}{c["contribution"]:.3f} '
            f'({c["direction"]})</td></tr>'
            for c in d["contribs"])
        lt = d["lead"]
        body += (
            f'<div class="drill" id="d-{_esc(d["drug_id"])}">'
            f'<h3><span class="glyph">{TIER_GLYPH[d["tier"]]}</span> '
            f'{_esc(d["drug_name"])} &middot; {d["tier"]} '
            f'(score {d["score"]:.3f})</h3>'
            f'<p class="note">{_esc(d["narration"])} '
            f'<span class="small">[narration engine: {_esc(d["engine"])}]</span></p>'
            '<div class="grid2">'
            '<div><table class="kv"><thead><tr><th>Point-in-time feature</th>'
            '<th class="num">Value</th></tr></thead><tbody>'
            + feat_rows + '</tbody></table></div>'
            '<div><table class="kv"><thead><tr><th>Top contribution</th>'
            '<th class="num">Value</th><th>Effect (coef&times;z)</th></tr></thead>'
            '<tbody>' + contrib_rows + '</tbody></table>'
            f'<p class="small">Lead-time backtest: {_esc(lt)}</p>'
            '</div></div></div>')
    body += '</div>'
    return body


def _validation_view(results, lineage, run_meta, cfg) -> str:
    t = results["test"]
    m, b, p = t["model"], t["baseline_base_rate"], t["baseline_persistence"]

    def mrow(name, blk, extra=""):
        c = blk["confusion"]
        return (f'<tr><td>{name}</td>'
                f'<td class="num">{blk["pr_auc"]:.4f}</td>'
                f'<td class="num">{c["precision"]:.3f}</td>'
                f'<td class="num">{c["recall"]:.3f}</td>'
                f'<td class="num">{c["f1"]:.3f}</td>'
                f'<td class="num">{blk["brier"]:.4f}</td>'
                f'<td class="small">{extra}</td></tr>')

    beats = ('<span class="pass">YES</span>' if t["model_beats_persistence"]
             else '<span class="fail">NO</span>')
    beats_base = ('<span class="pass">YES</span>' if t["model_beats_base_rate"]
                  else '<span class="fail">NO</span>')

    lt = results["lead_time"]
    lead_summary = (
        f'median <b>{lt["median_days"]}</b> d, IQR {lt["iqr_days"][0]}-'
        f'{lt["iqr_days"][1]} d, range {lt["min_days"]}-{lt["max_days"]} d; '
        f'{lt["n_flagged_in_advance"]}/{lt["n_onsets_test"]} test onsets flagged '
        f'in advance (detection {lt["detection_rate"]:.2f})'
        if lt.get("median_days") is not None else
        'no onsets flagged in advance at the operating threshold')

    conf = m["confusion"]
    hold = run_meta.get("holdout", {})
    hold_html = ""
    if hold.get("status") == "ok":
        hm = hold["metrics"]
        hl = hold["lead_time"]
        hold_html = (
            '<h2>Sealed holdout &middot; planted ground truth, unseen by training</h2>'
            f'<p class="note">{hold["n_drugs"]} drugs / {hold["n_rows"]} rows, '
            'structurally quarantined until this final step.</p>'
            '<table><thead><tr><th>PR-AUC</th><th class="num">Precision</th>'
            '<th class="num">Recall</th><th class="num">F1</th>'
            '<th class="num">Base rate</th><th>Lead (median)</th></tr></thead><tbody>'
            f'<tr><td class="num">{hm["pr_auc"]:.4f}</td>'
            f'<td class="num">{hm["confusion"]["precision"]:.3f}</td>'
            f'<td class="num">{hm["confusion"]["recall"]:.3f}</td>'
            f'<td class="num">{hm["confusion"]["f1"]:.3f}</td>'
            f'<td class="num">{hm["base_rate"]:.4f}</td>'
            f'<td class="num">{hl.get("median_days")}</td></tr>'
            '</tbody></table>')

    lim = "".join(f"<li>{_esc(x)}</li>" for x in run_meta.get("limitations", []))

    return (
        '<div class="view" id="view-valid">'
        '<h2>Model vs baselines &middot; temporal test set</h2>'
        '<p class="note">Headline metric is PR-AUC (imbalanced problem). Accuracy '
        'is deliberately not reported as a headline. Model reported only in '
        'comparison to naive baselines.</p>'
        '<table><thead><tr><th>Scorer</th><th class="num">PR-AUC</th>'
        '<th class="num">Precision</th><th class="num">Recall</th>'
        '<th class="num">F1</th><th class="num">Brier</th><th></th></tr></thead>'
        '<tbody>'
        + mrow('Logistic regression (headline)', m,
               f'GBT ref PR-AUC {results["gbt_test_pr_auc"]:.4f}')
        + mrow('Baseline: base rate', b, 'constant score')
        + mrow('Baseline: prior-shortage persistence', p, 'rank by prior count')
        + '</tbody></table>'
        f'<p class="note">Beats base-rate PR-AUC: {beats_base} &nbsp;&middot;&nbsp; '
        f'Beats persistence PR-AUC: {beats}. '
        'If NO, that is stated here plainly rather than hidden.</p>'

        '<h2>Lead time &middot; headline result</h2>'
        f'<p class="note">{lead_summary}</p>'
        '<div class="grid2">'
        f'<div><h2>Lead-time distribution</h2>{leadtime_svg(lt.get("histogram", []))}</div>'
        f'<div><h2>Precision-recall (solid=model, dashed=persistence, '
        f'dotted=base rate)</h2>{pr_curve_svg(t["pr_curve"], t["pr_curve_persistence"], m["base_rate"])}</div>'
        '</div>'
        '<div class="grid2">'
        f'<div><h2>Calibration (reliability)</h2>{calibration_svg(t["calibration"])}'
        '<p class="small">Diagonal = perfect calibration. Probabilities are raw '
        'logistic outputs (no reweighting), so they stay interpretable as '
        'probabilities.</p></div>'
        '<div><h2>Confusion matrix @ operating threshold '
        f'{results["threshold"]:.3f}</h2>'
        '<table class="kv"><tbody>'
        f'<tr><td>True positives</td><td class="num">{conf["tp"]}</td></tr>'
        f'<tr><td>False positives</td><td class="num">{conf["fp"]}</td></tr>'
        f'<tr><td>False negatives</td><td class="num">{conf["fn"]}</td></tr>'
        f'<tr><td>True negatives</td><td class="num">{conf["tn"]}</td></tr>'
        f'<tr><td>Precision</td><td class="num">{conf["precision"]:.3f}</td></tr>'
        f'<tr><td>Recall</td><td class="num">{conf["recall"]:.3f}</td></tr>'
        '</tbody></table></div>'
        '</div>'
        + hold_html +
        f'<div class="limbox"><h3>Limitations stated on this page</h3><ul>{lim}</ul>'
        '</div></div>')


def build_report(cfg, lineage, results, model, scored_test: pd.DataFrame,
                 reports, run_meta) -> str:
    # latest as_of per drug -> ranking
    scored_test = scored_test.sort_values(["drug_id", "as_of"], kind="stable")
    latest = scored_test.groupby("drug_id", as_index=False).tail(1)
    latest = latest.sort_values("score", ascending=False, kind="stable")

    # onset lookup for drilldown lead-time text
    onset_map: dict[str, list[str]] = {}
    for r in reports:
        if r.get("report_type") == "shortage" and r.get("start_date"):
            onset_map.setdefault(r["drug_id"], []).append(r["start_date"])
    split = date.fromisoformat(cfg.split_date)

    ranked = []
    drills = []
    for _, row in latest.iterrows():
        contribs = model.contributions(row, top_k=5)
        tier = _tier(float(row["score"]), cfg.tier_cuts)
        ranked.append({
            "drug_id": row["drug_id"], "drug_name": row["drug_name"],
            "category": row["therapeutic_category"], "score": float(row["score"]),
            "tier": tier, "contribs": contribs,
            "single_supplier": int(row["single_supplier_flag"]),
            "prior": int(row["prior_shortage_count"]),
            "recent_disc": int(row["recent_disc_flag_120"]),
        })

    # drilldowns for the top 30 ranked drugs
    for r in ranked[:30]:
        row = latest[latest.drug_id == r["drug_id"]].iloc[0]
        feats = {k: (round(float(row[k]), 4) if isinstance(row[k], float) else row[k])
                 for k in [
                     "as_of", "prior_shortage_count", "has_prior_shortage",
                     "days_since_last_shortage", "recent_shortage_count_365",
                     "distinct_manufacturers", "single_supplier_flag",
                     "manufacturer_hhi", "disc_count", "days_since_last_disc",
                     "recent_disc_flag_120", "category_base_rate_pit"]}
        nar = narrate.narrate(r["drug_name"], r["contribs"], cfg.narration)
        test_onsets = [o for o in onset_map.get(r["drug_id"], [])
                       if date.fromisoformat(o) > split]
        lead_txt = (f'{len(test_onsets)} onset(s) in test period '
                    f'(first {test_onsets[0]})' if test_onsets
                    else 'no shortage onset in the test period for this drug')
        drills.append({**r, "features": feats, "narration": nar["text"],
                       "engine": nar["engine"], "lead": lead_txt})

    stamp = run_meta.get("build_stamp", "")
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Drug-Shortage Early-Warning &middot; retrospective signal recovery</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        "<div class='mast'><h1>Drug-Shortage Early-Warning &middot; "
        "retrospective signal recovery</h1>"
        f"<span class='stamp'>{_esc(stamp)}</span></div>"
        + _banner(lineage) +
        "<div class='tabs'>"
        "<button class='tab active' onclick=\"showTab('view-rank',this)\">Ranked risk</button>"
        "<button class='tab' id='tab-drill' onclick=\"showTab('view-drill',this)\">Drill-down</button>"
        "<button class='tab' onclick=\"showTab('view-valid',this)\">Validation</button>"
        "</div>"
        + _ranked_view(ranked, cfg)
        + _drill_view(drills)
        + _validation_view(results, lineage, run_meta, cfg)
        + f"<script>{JS}</script>"
        "</div></body></html>")

    out_path = os.path.join(cfg.out_dir, "index.html")
    write_text(out_path, doc)
    return out_path
