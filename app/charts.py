"""
Inline SVG chart builders.

Charts are drawn on the deep well background so the palette data colors read
clearly. Chartreuse is reserved for the single most important series or bar in
a chart; everything else uses the plum to mauve ramp.
"""
from __future__ import annotations

from html import escape

BG = "#1F1322"
GRID = "#5E4A62"
AXIS = "#9A8592"
TEXT = "#E8E2D0"
DIM = "#BFB2A0"
ACCENT = "#B4E33D"
RAMP = ["#8C6E8E", "#B3A0B2", "#E8E2D0"]   # plum-mauve, light-mauve, oat
BAR = "#8C6E8E"


def _fmt(v: float, dp: int = 2) -> str:
    return f"{v:.{dp}f}"


def pr_curve_svg(pr_curves: dict, test_prevalence: float, pr_auc: dict) -> str:
    W, H = 520, 300
    ml, mr, mt, mb = 48, 16, 16, 40
    pw, ph = W - ml - mr, H - mt - mb

    def X(recall):  return ml + recall * pw
    def Y(prec):    return mt + (1 - prec) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Precision recall curves">']

    # gridlines and axis ticks at 0, 0.25, 0.5, 0.75, 1.0
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = Y(g); x = X(g)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-6}" y="{y+3:.1f}" fill="{AXIS}" font-size="10" '
                     f'text-anchor="end">{_fmt(g,2)}</text>')
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+16}" fill="{AXIS}" font-size="10" '
                     f'text-anchor="middle">{_fmt(g,2)}</text>')

    # no-skill baseline at precision = prevalence
    yb = Y(test_prevalence)
    parts.append(f'<line x1="{ml}" y1="{yb:.1f}" x2="{ml+pw}" y2="{yb:.1f}" '
                 f'stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="4 4"/>')

    # secondary series first (best single feature), then LR on top in chartreuse
    def poly(points, color, width, dash=""):
        d = " ".join(f"{X(p['recall']):.1f},{Y(p['precision']):.1f}" for p in points)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                     f'stroke-width="{width}"{da}/>')

    if "best_single_feature" in pr_curves:
        poly(pr_curves["best_single_feature"], RAMP[0], 2)
    if "logistic_regression" in pr_curves:
        poly(pr_curves["logistic_regression"], ACCENT, 2.6)

    # axis titles
    parts.append(f'<text x="{ml+pw/2:.0f}" y="{H-4}" fill="{AXIS}" font-size="11" '
                 f'text-anchor="middle">Recall</text>')
    parts.append(f'<text x="14" y="{mt+ph/2:.0f}" fill="{AXIS}" font-size="11" '
                 f'text-anchor="middle" transform="rotate(-90 14 {mt+ph/2:.0f})">Precision</text>')
    parts.append("</svg>")
    return "".join(parts)


def coef_svg(coefficients: list) -> str:
    numeric = [c for c in coefficients if not c["feature"].startswith("cat_")]
    numeric = sorted(numeric, key=lambda c: abs(c["coef"]), reverse=True)
    top = numeric  # all numeric drivers
    n = len(top)
    row_h = 26
    W = 520
    ml, mr, mt, mb = 150, 16, 10, 24
    ph = n * row_h
    H = mt + ph + mb
    max_abs = max(abs(c["coef"]) for c in top) or 1.0
    zero_x = ml + (W - ml - mr) * 0.5
    half = (W - ml - mr) * 0.5

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Standardized model coefficients">']
    parts.append(f'<line x1="{zero_x:.1f}" y1="{mt}" x2="{zero_x:.1f}" y2="{mt+ph}" '
                 f'stroke="{GRID}" stroke-width="1"/>')

    labels = {
        "single_supplier": "Single supplier",
        "hhi": "Concentration (HHI)",
        "suppliers": "Supplier count",
        "log_past_onsets": "Past shortages (log)",
        "supply_shock_6m": "Recent supplier exit",
        "suppliers_lost_12m": "Suppliers lost, 1y",
        "supplier_trend_6m": "Supplier trend, 6m",
        "months_since_last_onset": "Months since last",
        "category_recent_rate": "Category recent rate",
    }
    for i, c in enumerate(top):
        y = mt + i * row_h + row_h / 2
        val = c["coef"]
        w = abs(val) / max_abs * half
        color = ACCENT if i == 0 else BAR
        if val >= 0:
            x = zero_x
        else:
            x = zero_x - w
        parts.append(f'<rect x="{x:.1f}" y="{y-8:.1f}" width="{w:.1f}" height="16" '
                     f'rx="2" fill="{color}"/>')
        name = escape(labels.get(c["feature"], c["feature"]))
        parts.append(f'<text x="{ml-10}" y="{y+4:.1f}" fill="{TEXT}" font-size="12" '
                     f'text-anchor="end">{name}</text>')
        vx = (x + w + 6) if val >= 0 else (x - 6)
        anchor = "start" if val >= 0 else "end"
        parts.append(f'<text x="{vx:.1f}" y="{y+4:.1f}" fill="{DIM}" font-size="11" '
                     f'text-anchor="{anchor}" font-variant-numeric="tabular-nums">'
                     f'{val:+.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def leadtime_svg(hist: list, median_days) -> str:
    W, H = 520, 300
    ml, mr, mt, mb = 40, 16, 16, 46
    pw, ph = W - ml - mr, H - mt - mb
    if not hist:
        return f'<svg viewBox="0 0 {W} {H}" class="chart"></svg>'
    n = len(hist)
    gap = 6
    bw = (pw - gap * (n - 1)) / n
    max_c = max(b["count"] for b in hist) or 1

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Lead time distribution">']
    # y gridlines
    for frac in (0, 0.5, 1.0):
        y = mt + (1 - frac) * ph
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-6}" y="{y+3:.1f}" fill="{AXIS}" font-size="10" '
                     f'text-anchor="end">{int(round(max_c*frac))}</text>')

    for i, b in enumerate(hist):
        x = ml + i * (bw + gap)
        h = b["count"] / max_c * ph
        y = mt + ph - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                     f'rx="2" fill="{BAR}"/>')
        mid = int((b["lo"] + b["hi"]) / 2)
        parts.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+14}" fill="{AXIS}" font-size="9" '
                     f'text-anchor="middle">{mid}</text>')

    if median_days is not None:
        # map median (days) onto the bin axis by proportion of range
        lo = hist[0]["lo"]; hi = hist[-1]["hi"]
        frac = (median_days - lo) / (hi - lo) if hi > lo else 0.5
        frac = min(max(frac, 0.0), 1.0)
        mx = ml + frac * pw
        parts.append(f'<line x1="{mx:.1f}" y1="{mt}" x2="{mx:.1f}" y2="{mt+ph}" '
                     f'stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{mx:.1f}" y="{mt-2}" fill="{DIM}" font-size="10" '
                     f'text-anchor="middle">median {int(median_days)}d</text>')

    parts.append(f'<text x="{ml+pw/2:.0f}" y="{H-6}" fill="{AXIS}" font-size="11" '
                 f'text-anchor="middle">Days of warning before official onset</text>')
    parts.append("</svg>")
    return "".join(parts)


def survival_svg(survival: dict) -> str:
    W, H = 520, 300
    ml, mr, mt, mb = 44, 16, 16, 44
    pw, ph = W - ml - mr, H - mt - mb
    groups = survival.get("groups", {})
    # x axis max is the largest t across curves
    max_t = 1
    for g in groups.values():
        for pnt in g["curve"]:
            max_t = max(max_t, pnt["t"])

    def X(t):  return ml + (t / max_t) * pw
    def Y(s):  return mt + (1 - s) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Time to next shortage by risk group">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = Y(frac)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ml-6}" y="{y+3:.1f}" fill="{AXIS}" font-size="10" '
                     f'text-anchor="end">{_fmt(frac,2)}</text>')
    # 50 percent reference
    parts.append(f'<line x1="{ml}" y1="{Y(0.5):.1f}" x2="{ml+pw}" y2="{Y(0.5):.1f}" '
                 f'stroke="{AXIS}" stroke-width="1" stroke-dasharray="3 3"/>')

    order = [("high", RAMP[2]), ("medium", RAMP[1]), ("low", RAMP[0])]
    for name, color in order:
        g = groups.get(name)
        if not g:
            continue
        pts = g["curve"]
        d = f'M {X(0):.1f} {Y(1.0):.1f}'
        prev_s = 1.0
        for pnt in pts:
            d += f' L {X(pnt["t"]):.1f} {Y(prev_s):.1f}'
            d += f' L {X(pnt["t"]):.1f} {Y(pnt["s"]):.1f}'
            prev_s = pnt["s"]
        d += f' L {X(max_t):.1f} {Y(prev_s):.1f}'
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"/>')

    parts.append(f'<text x="{ml+pw/2:.0f}" y="{H-6}" fill="{AXIS}" font-size="11" '
                 f'text-anchor="middle">Months from observation</text>')
    parts.append(f'<text x="14" y="{mt+ph/2:.0f}" fill="{AXIS}" font-size="11" '
                 f'text-anchor="middle" transform="rotate(-90 14 {mt+ph/2:.0f})">'
                 f'Share not yet short</text>')
    parts.append("</svg>")
    return "".join(parts)
