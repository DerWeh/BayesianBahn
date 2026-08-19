"""Render the DB comparison as a self-contained HTML report.

The numbers in this evaluation only mean something alongside their definitions,
their caveats and the commands that produced them, so the report carries all
three. It is generated, never hand-edited: rerun it after another day of
collection and the figures move on their own.

Input layout, which tools/run_evaluation.sh produces:

    <scored-dir>/<day>/arrivals-live.jsonl
    <scored-dir>/<day>/arrivals-blind.jsonl
    <scored-dir>/<day>/connections-live.jsonl
    <scored-dir>/<day>/connections-blind.jsonl

Usage:
    python tools/report.py --scored-dir tools/.scored --days 2026-08-17 \
        --out tools/.scored/report.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from score_events import (  # noqa: E402
    BUCKET_LABELS, SURPRISE_MINUTES, TRANSFER_MINUTES, anchor_minutes, bucket_of,
    wall_to_epoch,
)

# Categorical slots 1-3 of the reference palette, which validate on all pairs in
# both modes. Three series is also the cap here: DB, and the model with and
# without the live signal.
SERIES = {
    "db": ("DB", "var(--series-1)"),
    "blind": ("BayesianBahn, history only", "var(--series-2)"),
    "live": ("BayesianBahn, as shipped", "var(--series-3)"),
}

ANCHOR = "planned_departure"


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def arrivals_table(live: list[dict], blind: list[dict]) -> list[dict]:
    """Per lead bucket: DB's error, and both variants of ours."""
    def binned(rows):
        out: dict[str, list[dict]] = {}
        for r in rows:
            at = anchor_minutes(r, ANCHOR)
            if at is None:
                continue
            label = bucket_of((wall_to_epoch(at) - r["read_at"]) / 60)
            if label:
                out.setdefault(label, []).append(r)
        return out

    by_live, by_blind = binned(live), binned(blind)
    rows = []
    for label in BUCKET_LABELS:
        g_live, g_blind = by_live.get(label), by_blind.get(label)
        if not g_live or not g_blind:
            continue
        rows.append({
            "bucket": label,
            "n": len(g_blind),
            # A point forecast's CRPS is its absolute error, so DB's MAE is
            # directly comparable with our CRPS.
            "db": st.mean(abs(r["db"] - r["truth"]) for r in g_blind),
            "db_bias": st.mean(r["db"] - r["truth"] for r in g_blind),
            "db_surprise": sum(1 for r in g_blind
                               if r["truth"] > r["db"] + SURPRISE_MINUTES) / len(g_blind),
            "blind": st.mean(r["crps"] for r in g_blind),
            "blind_cover": sum(1 for r in g_blind
                               if r["q10"] <= r["truth"] <= r["q90"]) / len(g_blind),
            "live": st.mean(r["crps"] for r in g_live),
            "live_cover": sum(1 for r in g_live
                              if r["q10"] <= r["truth"] <= r["q90"]) / len(g_live),
        })
    return rows


BANDS = ((2, 6, "2-5 min"), (6, 11, "6-10 min"), (11, 21, "11-20 min"),
         (21, 31, "21-30 min"))


def brier(rows: list[dict], key: str) -> float:
    return st.mean((r[key] - (1.0 if r["caught"] else 0.0)) ** 2 for r in rows)


def connections_table(live: list[dict], blind: list[dict]) -> list[dict]:
    out = []
    for lo, hi, label in BANDS:
        g_live = [r for r in live if lo <= r["slack"] < hi]
        g_blind = [r for r in blind if lo <= r["slack"] < hi]
        if not g_blind:
            continue
        out.append({
            "band": label,
            "n": len(g_blind),
            "caught": sum(1 for r in g_blind if r["caught"]) / len(g_blind),
            "db": brier(g_blind, "db_catch_p"),
            "blind": brier(g_blind, "p_catch"),
            "live": brier(g_live, "p_catch") if g_live else float("nan"),
        })
    return out


def cluster_ci(rows: list[dict], value, *, draws: int = 2000,
               seed: int = 20260819) -> tuple[float, float, float]:
    """Mean of `value`, with a 95% interval from a bootstrap over *trains*.

    Resampling events would be wrong and flattering: one late ICE contributes a
    dozen correlated predictions, so events are nowhere near independent and an
    event-level interval comes out far too narrow. Resampling whole trains keeps
    each cluster intact. Seeded, so the published interval is reproducible.
    """
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r["cat"], r["num"]), []).append(value(r))
    keys = list(groups)
    point = st.mean(v for g in groups.values() for v in g)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pick = [groups[keys[rng.randrange(len(keys))]] for _ in range(len(keys))]
        means.append(st.mean(v for g in pick for v in g))
    means.sort()
    return point, means[int(0.025 * draws)], means[int(0.975 * draws)]


def crps_gap(rows: list[dict]) -> tuple[float, float, float]:
    """Our CRPS minus DB's, per prediction. Negative means we are better."""
    return cluster_ci(rows, lambda r: r["crps"] - abs(r["db"] - r["truth"]))


def brier_gap(rows: list[dict]) -> tuple[float, float, float]:
    def gap(r):
        outcome = 1.0 if r["caught"] else 0.0
        return (r["p_catch"] - outcome) ** 2 - (r["db_catch_p"] - outcome) ** 2
    return cluster_ci(rows, gap)


def headline(live, blind, conn_live, conn_blind) -> list[dict]:
    """The comparisons against DB, each with the uncertainty that decides it.

    Without an interval a difference in the third decimal reads as a finding. It
    was exactly this that made a single evening's 189 missed connections look
    like a result.
    """
    missed_live = [r for r in conn_live if r["caught"] is False]
    missed_blind = [r for r in conn_blind if r["caught"] is False]
    rows = [
        ("Arrival time, as shipped", "CRPS, minutes", len(live), crps_gap(live)),
        ("Arrival time, history only", "CRPS, minutes", len(blind), crps_gap(blind)),
        ("Every connection, as shipped", "Brier", len(conn_live), brier_gap(conn_live)),
        ("Every connection, history only", "Brier", len(conn_blind), brier_gap(conn_blind)),
        ("Missed connections, as shipped", "Brier", len(missed_live), brier_gap(missed_live)),
        ("Missed connections, history only", "Brier", len(missed_blind),
         brier_gap(missed_blind)),
    ]
    out = []
    for what, unit, n, (point, lo, hi) in rows:
        out.append({
            "what": what, "unit": unit, "n": n,
            "gap": point, "lo": lo, "hi": hi,
            # The interval, not the point, decides what may be claimed.
            "verdict": "we are better" if hi < 0 else
                       ("DB is better" if lo > 0 else "not separated"),
        })
    return out


def per_day(days: list[str], scored_dir: Path) -> list[dict]:
    """One row per collected day: the same headline numbers, unpooled.

    Pooling hides whether a result is a property of the model or of one day's
    weather. A claim that moves when a day is added was never a claim.
    """
    out = []
    for day in days:
        base = scored_dir / day
        live, blind = load(base / "arrivals-live.jsonl"), load(base / "arrivals-blind.jsonl")
        conn_blind = load(base / "connections-blind.jsonl")
        conn_live = load(base / "connections-live.jsonl")
        if not blind:
            continue
        missed_l = [r for r in conn_live if r["caught"] is False]
        missed_b = [r for r in conn_blind if r["caught"] is False]
        out.append({
            "day": day,
            "n": len(blind),
            "db": st.mean(abs(r["db"] - r["truth"]) for r in blind),
            "blind": st.mean(r["crps"] for r in blind),
            "live": st.mean(r["crps"] for r in live),
            "live_cover": sum(1 for r in live if r["q10"] <= r["truth"] <= r["q90"]) / len(live),
            "blind_cover": sum(1 for r in blind
                               if r["q10"] <= r["truth"] <= r["q90"]) / len(blind),
            "missed": len(missed_b),
            "db_missed": brier(missed_b, "db_catch_p") if missed_b else float("nan"),
            "blind_missed": brier(missed_b, "p_catch") if missed_b else float("nan"),
            "live_missed": brier(missed_l, "p_catch") if missed_l else float("nan"),
        })
    return out


def outcome_split(live: list[dict], blind: list[dict]) -> list[dict]:
    out = []
    for label, want in (("Connection was caught", True), ("Connection was missed", False)):
        g_blind = [r for r in blind if r["caught"] is want]
        g_live = [r for r in live if r["caught"] is want]
        if not g_blind:
            continue
        out.append({
            "outcome": label,
            "n": len(g_blind),
            "db_right": sum(1 for r in g_blind
                            if (r["db_catch_p"] == 1) == r["caught"]) / len(g_blind),
            "db": brier(g_blind, "db_catch_p"),
            "blind": brier(g_blind, "p_catch"),
            "blind_p": st.mean(r["p_catch"] for r in g_blind),
            "live": brier(g_live, "p_catch") if g_live else float("nan"),
            "live_p": st.mean(r["p_catch"] for r in g_live) if g_live else float("nan"),
        })
    return out


# --- charts ------------------------------------------------------------------


def line_chart(rows, keys, *, y_label, height=240, y_max=None, rule=None,
               rule_label="") -> str:
    """Ordered categories on x, one line per series, markers at every point."""
    width, pad_l, pad_r, pad_t, pad_b = 720, 54, 16, 16, 44
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    top = y_max or max(max(r[k] for k in keys) for r in rows) * 1.15
    step = plot_w / max(1, len(rows) - 1) if len(rows) > 1 else plot_w

    def x(i):
        return pad_l + i * step

    def y(v):
        return pad_t + plot_h - (v / top) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(y_label)} by lead time">']
    # Recessive grid, four steps.
    for g in range(5):
        v = top * g / 4
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" '
                     f'y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 8}" y="{y(v) + 4:.1f}" '
                     f'text-anchor="end">{v:.1f}</text>')
    if rule is not None:
        parts.append(f'<line class="rule-line" x1="{pad_l}" x2="{width - pad_r}" '
                     f'y1="{y(rule):.1f}" y2="{y(rule):.1f}"/>')
        parts.append(f'<text class="rule-label" x="{width - pad_r}" '
                     f'y="{y(rule) - 6:.1f}" text-anchor="end">'
                     f'{html.escape(rule_label)}</text>')
    for i, r in enumerate(rows):
        parts.append(f'<text class="tick" x="{x(i):.1f}" y="{height - pad_b + 20}" '
                     f'text-anchor="middle">{html.escape(r["bucket"])}</text>')
    for key in keys:
        label, colour = SERIES[key]
        points = " ".join(f"{x(i):.1f},{y(r[key]):.1f}" for i, r in enumerate(rows))
        parts.append(f'<polyline class="series" points="{points}" stroke="{colour}"/>')
        for i, r in enumerate(rows):
            parts.append(
                f'<circle cx="{x(i):.1f}" cy="{y(r[key]):.1f}" r="5" fill="{colour}" '
                f'class="marker"><title>{html.escape(label)} — '
                f'{html.escape(r["bucket"])}: {r[key]:.2f}</title></circle>')
    parts.append(f'<text class="axis-label" x="{pad_l}" y="{height - 6}">'
                 f'{html.escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(rows, keys, *, y_label, label_key, height=240, y_max=None,
              percent=False, rule=None, rule_label="") -> str:
    width, pad_l, pad_r, pad_t, pad_b = 720, 54, 16, 16, 44
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    top = y_max or max(max(r[k] for k in keys) for r in rows) * 1.2
    group_w = plot_w / len(rows)
    bar_w = min(34, (group_w - 14) / len(keys))

    def y(v):
        return pad_t + plot_h - (v / top) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(y_label)}">']
    for g in range(5):
        v = top * g / 4
        shown = f"{v * 100:.0f}%" if percent else f"{v:.3f}"
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" '
                     f'y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 8}" y="{y(v) + 4:.1f}" '
                     f'text-anchor="end">{shown}</text>')
    if rule is not None:
        parts.append(f'<line class="rule-line" x1="{pad_l}" x2="{width - pad_r}" '
                     f'y1="{y(rule):.1f}" y2="{y(rule):.1f}"/>')
        parts.append(f'<text class="rule-label" x="{width - pad_r}" '
                     f'y="{y(rule) - 6:.1f}" text-anchor="end">'
                     f'{html.escape(rule_label)}</text>')
    for i, r in enumerate(rows):
        centre = pad_l + group_w * (i + 0.5)
        # 2px gap between adjacent bars, per the mark spec.
        span = bar_w * len(keys) + 2 * (len(keys) - 1)
        for j, key in enumerate(keys):
            value = r[key]
            if value != value:      # NaN: the variant produced nothing here
                continue
            label, colour = SERIES[key]
            bx = centre - span / 2 + j * (bar_w + 2)
            bh = max(1.0, plot_h - (y(value) - pad_t))
            parts.append(
                f'<rect x="{bx:.1f}" y="{y(value):.1f}" width="{bar_w:.1f}" '
                f'height="{bh:.1f}" rx="4" fill="{colour}" class="marker">'
                f'<title>{html.escape(label)} — {html.escape(str(r[label_key]))}: '
                f'{value:.3f}</title></rect>')
        parts.append(f'<text class="tick" x="{centre:.1f}" y="{height - pad_b + 20}" '
                     f'text-anchor="middle">{html.escape(str(r[label_key]))}</text>')
    parts.append(f'<text class="axis-label" x="{pad_l}" y="{height - 6}">'
                 f'{html.escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def legend(keys) -> str:
    items = "".join(
        f'<li><span class="swatch" style="background:{SERIES[k][1]}"></span>'
        f'{html.escape(SERIES[k][0])}</li>' for k in keys)
    return f'<ul class="legend">{items}</ul>'


def table(rows, columns) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title, _ in columns)
    body = []
    for r in rows:
        cells = "".join(f"<td>{fmt(r)}</td>" for _, _, fmt in columns)
        body.append(f"<tr>{cells}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


# --- the document ------------------------------------------------------------

STYLE = """
:root {
  color-scheme: light;
  --ground:      #f6f7f9;
  --panel:       #ffffff;
  --ink:         #14181d;
  --ink-2:       #545d6b;
  --ink-3:       #79828f;
  --rule:        #dfe3e9;
  --rule-strong: #c3cad4;
  --series-1:    #2a78d6;
  --series-2:    #eb6834;
  --series-3:    #1baf7a;
  --flag:        #b4451f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:      #14161a;
    --panel:       #1b1e24;
    --ink:         #eef1f5;
    --ink-2:       #a7b0bd;
    --ink-3:       #7e8895;
    --rule:        #2a2f37;
    --rule-strong: #3a414b;
    --series-1:    #3987e5;
    --series-2:    #d95926;
    --series-3:    #199e70;
    --flag:        #e08256;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:      #14161a;
  --panel:       #1b1e24;
  --ink:         #eef1f5;
  --ink-2:       #a7b0bd;
  --ink-3:       #7e8895;
  --rule:        #2a2f37;
  --rule-strong: #3a414b;
  --series-1:    #3987e5;
  --series-2:    #d95926;
  --series-3:    #199e70;
  --flag:        #e08256;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
}
.wrap { max-width: 60rem; margin: 0 auto; padding: 3.5rem 1.5rem 6rem; }
header { border-bottom: 2px solid var(--rule-strong); padding-bottom: 1.75rem; }
.eyebrow {
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, monospace;
  font-size: .75rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--ink-3); margin: 0 0 .75rem;
}
h1 { font-size: 2.1rem; line-height: 1.15; letter-spacing: -.022em; margin: 0 0 .6rem;
     text-wrap: balance; font-weight: 640; }
h2 { font-size: 1.3rem; letter-spacing: -.014em; margin: 0 0 .3rem; font-weight: 620;
     text-wrap: balance; }
h3 { font-size: .95rem; letter-spacing: -.006em; margin: 0 0 .4rem; font-weight: 620; }
p { margin: 0 0 1rem; max-width: 68ch; color: var(--ink-2); }
p.lede { color: var(--ink); font-size: 1.06rem; }
section { padding-top: 2.75rem; }
section + section { border-top: 1px solid var(--rule); margin-top: 2.75rem; }
.figure { background: var(--panel); border: 1px solid var(--rule);
          border-radius: 10px; padding: 1.25rem 1.25rem .75rem; margin: 1.25rem 0; }
.figure svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--rule); stroke-width: 1; }
.rule-line { stroke: var(--ink-3); stroke-width: 1.5; stroke-dasharray: 5 4; }
.rule-label, .tick, .axis-label {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-variant-numeric: tabular-nums;
}
.tick { font-size: 11px; fill: var(--ink-3); }
.rule-label { font-size: 11px; fill: var(--ink-2); }
.axis-label { font-size: 11px; fill: var(--ink-3); letter-spacing: .06em;
              text-transform: uppercase; }
.series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.marker { stroke: var(--panel); stroke-width: 2; }
.legend { display: flex; flex-wrap: wrap; gap: 1.1rem; list-style: none;
          padding: 0; margin: .35rem 0 1rem; font-size: .85rem; color: var(--ink-2); }
.legend li { display: flex; align-items: center; gap: .45rem; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .85rem;
        font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .5rem .7rem; border-bottom: 1px solid var(--rule);
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--ink-3); font-weight: 600; font-size: .74rem;
           letter-spacing: .07em; text-transform: uppercase; }
td { font-family: ui-monospace, "SF Mono", Menlo, monospace; color: var(--ink); }
td:first-child { font-family: inherit; color: var(--ink-2); }
tbody tr:last-child td { border-bottom: none; }
.tiles { display: grid; gap: .9rem; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
         margin: 1.5rem 0; }
.tile { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
        padding: 1rem 1.1rem; }
.tile .k { font-family: ui-monospace, Menlo, monospace; font-size: 1.65rem;
           font-variant-numeric: tabular-nums; letter-spacing: -.02em; display: block; }
.tile .v { font-size: .8rem; color: var(--ink-2); display: block; margin-top: .25rem; }
.tile.flag .k { color: var(--flag); }
dl { margin: 0; display: grid; gap: 1rem; }
dt { font-weight: 620; font-size: .92rem; }
dd { margin: .15rem 0 0; color: var(--ink-2); max-width: 68ch; font-size: .92rem; }
code, pre { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .82rem; }
pre { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
      padding: 1rem 1.1rem; overflow-x: auto; color: var(--ink); line-height: 1.65; }
.caveats li { color: var(--ink-2); margin-bottom: .5rem; max-width: 68ch; }
footer { margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--rule);
         color: var(--ink-3); font-size: .8rem; }
a { color: var(--series-1); }
:focus-visible { outline: 2px solid var(--series-1); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def pct(x):
    return "—" if x != x else f"{x * 100:.0f}%"


def num(x, digits=2):
    return "—" if x != x else f"{x:.{digits}f}"


def render(days, arrivals, connections, split, totals, out: Path, *,
           gaps=(), daily=()) -> None:
    span = ", ".join(days)
    cross = next((r["bucket"] for r in arrivals if r["blind"] < r["db"]), None)
    missed = next((r for r in split if "missed" in r["outcome"]), None)

    tiles = [
        ("tile", f"{totals['events']:,}", "arrival predictions scored"),
        ("tile", f"{totals['connections']:,}", "one-change connections scored"),
        ("tile", "history only" if cross else "—",
         f"beats DB from {cross} before departure" if cross else "no crossover found"),
    ]
    if missed:
        tiles.append(("tile flag", pct(missed["db_right"]),
                      "of missed connections DB called correctly"))

    doc = [f"""<title>Do we beat the DB Navigator?</title>
<style>{STYLE}</style>
<div class="wrap">
<header>
  <p class="eyebrow">BayesianBahn · evaluation · {html.escape(span)}</p>
  <h1>Do we beat the DB Navigator?</h1>
  <p class="lede">An app that predicts train delays only earns its place if its
  numbers are better than the ones already on the platform display. This is that
  comparison, measured against what the trains actually did.</p>
</header>

<section>
  <h2>What was measured</h2>
  <p>Every ten minutes, DB’s own forecast was recorded for twenty stations chosen
  in advance across the whole network — six major hubs down to three village
  halts. The next day the archive says when each train really arrived, and both
  forecasts are scored against that.</p>
  <p>The model is the one that ships, run through the app’s own prediction code,
  and it is only ever shown history from before the day it predicts. It appears
  twice: <strong>as shipped</strong>, which adjusts DB’s live number using past
  runs, and <strong>history only</strong>, which ignores the live number
  entirely.</p>
  <div class="tiles">"""]
    for cls, k, v in tiles:
        doc.append(f'<div class="{cls}"><span class="k">{html.escape(k)}</span>'
                   f'<span class="v">{html.escape(v)}</span></div>')
    doc.append("</div>\n</section>")

    if gaps:
        doc.append("""
<section>
  <h2>The answer, with its uncertainty</h2>
  <p>Each row is our score minus DB’s for the same predictions, so a
  <strong>negative number means we are better</strong>. The interval is what
  decides it: delays arrive in clusters — one late train produces a dozen
  correlated predictions — so the range comes from resampling whole trains, not
  individual predictions. Where the interval crosses zero, two days of data are
  not enough to claim anything, however suggestive the middle number looks.</p>""")
        doc.append(table(gaps, [
            ("what", "Comparison", lambda r: html.escape(r["what"])),
            ("n", "Predictions", lambda r: f"{r['n']:,}"),
            ("unit", "Score", lambda r: html.escape(r["unit"])),
            ("gap", "Ours − DB", lambda r: num(r["gap"], 3)),
            ("lo", "95% interval",
             lambda r: f"{num(r['lo'], 3)} to {num(r['hi'], 3)}"),
            ("verdict", "Reading", lambda r: html.escape(r["verdict"])),
        ]))
        doc.append("</section>")

    doc.append("""
<section>
  <h2>How far ahead, and how wrong</h2>
  <p>Lead time is counted back from the train’s <em>scheduled departure</em>,
  because that is when a passenger can still act on the answer. Lower is better;
  the scores are in minutes and are directly comparable — see the definitions at
  the foot of the page for why a point forecast and a distribution can be put on
  one axis.</p>""")
    doc.append('<div class="figure">')
    doc.append(legend(["db", "blind", "live"]))
    doc.append(line_chart(arrivals, ["db", "blind", "live"],
                          y_label="CRPS, minutes (lower is better)"))
    doc.append("</div>")
    doc.append(table(arrivals, [
        ("bucket", "Before departure", lambda r: html.escape(r["bucket"])),
        ("n", "Predictions", lambda r: f"{r['n']:,}"),
        ("db", "DB", lambda r: num(r["db"])),
        ("blind", "History only", lambda r: num(r["blind"])),
        ("live", "As shipped", lambda r: num(r["live"])),
        ("db_bias", "DB bias", lambda r: num(r["db_bias"])),
        ("db_surprise", "DB surprises", lambda r: pct(r["db_surprise"])),
    ]))
    doc.append("""</section>

<section>
  <h2>Is the 80% range honest?</h2>
  <p>The app does not only give a time, it gives a range it claims will contain
  the truth four times in five. That claim is checkable: count how often the real
  arrival landed inside it. A bar near the dashed line is an honest range; well
  below it means the app is more confident than it has earned.</p>""")
    doc.append('<div class="figure">')
    doc.append(legend(["blind", "live"]))
    doc.append(bar_chart(
        [{"bucket": r["bucket"], "blind": r["blind_cover"], "live": r["live_cover"]}
         for r in arrivals],
        ["blind", "live"], y_label="Share of arrivals inside the stated range",
        label_key="bucket", y_max=1.0, percent=True,
        rule=0.8, rule_label="80% — what the app claims"))
    doc.append("</div>")
    doc.append(table(arrivals, [
        ("bucket", "Before departure", lambda r: html.escape(r["bucket"])),
        ("n", "Predictions", lambda r: f"{r['n']:,}"),
        ("blind_cover", "History only", lambda r: pct(r["blind_cover"])),
        ("live_cover", "As shipped", lambda r: pct(r["live_cover"])),
    ]))
    doc.append("""</section>

<section>
  <h2>Will you make your connection?</h2>
  <p>This is what the app is for. A connection here is a train arriving at one of
  the sampled stations and another leaving it a few minutes later, judged from
  before the first train set off — the only moment the answer can still change a
  decision. DB answers yes or no. The app answers with a probability.</p>""")
    doc.append('<div class="figure">')
    doc.append(legend(["db", "blind", "live"]))
    doc.append(bar_chart(connections, ["db", "blind", "live"],
                         y_label="Brier score (lower is better)", label_key="band"))
    doc.append("</div>")
    doc.append(table(connections, [
        ("band", "Time to change", lambda r: html.escape(r["band"])),
        ("n", "Connections", lambda r: f"{r['n']:,}"),
        ("caught", "Actually caught", lambda r: pct(r["caught"])),
        ("db", "DB", lambda r: num(r["db"], 3)),
        ("blind", "History only", lambda r: num(r["blind"], 3)),
        ("live", "As shipped", lambda r: num(r["live"], 3)),
    ]))
    doc.append("""
  <h3>Split by what actually happened</h3>
  <p>The averages above hide the finding. Almost every connection is caught, so
  answering “yes” every time scores well — and that is close to what a yes/no
  answer computed from the timetable does. The row that matters is the one where
  the change failed.</p>""")
    doc.append(table(split, [
        ("outcome", "Outcome", lambda r: html.escape(r["outcome"])),
        ("n", "Connections", lambda r: f"{r['n']:,}"),
        ("db_right", "DB called it right", lambda r: pct(r["db_right"])),
        ("db", "DB Brier", lambda r: num(r["db"], 3)),
        ("blind", "History only", lambda r: num(r["blind"], 3)),
        ("blind_p", "…mean P(catch)", lambda r: num(r["blind_p"])),
        ("live", "As shipped", lambda r: num(r["live"], 3)),
    ]))
    doc.append("</section>")

    if len(daily) > 1:
        doc.append("""
<section>
  <h2>Does it hold from one day to the next?</h2>
  <p>Everything above pools the collected days, which is the right way to get a
  number and the wrong way to find out whether that number is real. Here each day
  stands alone. A result worth acting on is one that points the same way in every
  row; a column that changes its mind between days is telling you about the
  weather, not about the model.</p>
  <p>Read the row lengths too. The first day was collected from the evening
  onwards, so it is both smaller and drawn only from the busiest hours — a day
  with fewer missed connections here is not necessarily a calmer day.</p>""")
        doc.append(table(daily, [
            ("day", "Day", lambda r: html.escape(r["day"])),
            ("n", "Predictions", lambda r: f"{r['n']:,}"),
            ("db", "DB", lambda r: num(r["db"])),
            ("blind", "History only", lambda r: num(r["blind"])),
            ("live", "As shipped", lambda r: num(r["live"])),
            ("blind_cover", "80% range, history", lambda r: pct(r["blind_cover"])),
            ("live_cover", "80% range, shipped", lambda r: pct(r["live_cover"])),
        ]))
        doc.append("""
  <h3>The missed connections, day by day</h3>
  <p>This is the smallest sample on the page and the one most easily
  over-read.</p>""")
        doc.append(table(daily, [
            ("day", "Day", lambda r: html.escape(r["day"])),
            ("missed", "Missed", lambda r: f"{r['missed']:,}"),
            ("db_missed", "DB Brier", lambda r: num(r["db_missed"], 3)),
            ("blind_missed", "History only", lambda r: num(r["blind_missed"], 3)),
            ("live_missed", "As shipped", lambda r: num(r["live_missed"], 3)),
        ]))
        doc.append("</section>")

    doc.append("""
<section>
  <h2>Definitions</h2>
  <dl>
    <dt>CRPS — continuous ranked probability score</dt>
    <dd>The integral of (F(x) − 1{x ≥ y})² over all x, where F is the forecast’s
    cumulative distribution and y is what happened. In minutes; lower is better.
    A point forecast is a distribution with all its weight in one place, and for
    that case CRPS is exactly the absolute error — which is why DB’s column and
    ours sit on the same axis. It is the only score here that credits the whole
    distribution rather than just its middle.</dd>

    <dt>Brier score</dt>
    <dd>Mean squared distance between a stated probability and the 0/1 outcome;
    lower is better. A yes/no answer is a probability of 1 or 0, so its Brier
    score is simply the share it got wrong. This makes “yes” and “72%%”
    comparable.</dd>

    <dt>DB bias</dt>
    <dd>Mean signed error. Negative means DB predicted the train earlier than it
    arrived — optimism, which is the direction that costs a passenger a
    connection.</dd>

    <dt>DB surprises</dt>
    <dd>Share of arrivals more than %d minutes later than DB said. This is the
    error that turns into a missed change.</dd>

    <dt>As shipped vs history only</dt>
    <dd>Both are the same model. <em>As shipped</em> feeds it DB’s live number
    for the station; <em>history only</em> withholds that and uses past runs
    alone. The model was built to take the measured delay at the train’s
    <em>previous</em> stop, and the live path substitutes this station’s forecast
    instead — an approximation documented in the source, and the difference
    between these two columns is what it costs.</dd>

    <dt>Ground truth</dt>
    <dd>The archive’s recorded arrival minus the scheduled arrival, the same
    definition used to build the history the model learns from. Cancelled stops
    are excluded from the delay scores.</dd>

    <dt>Time to change</dt>
    <dd>Scheduled gap between the feeder arriving and the connection leaving,
    less the %d minutes assumed for walking between platforms.</dd>
  </dl>
</section>

<section>
  <h2>What this does not establish</h2>
  <ul class="caveats">
    <li><strong>%s of data.</strong> Delays cluster by line, by weather and by
    incident, so the effective sample is far smaller than the counts suggest. A
    single quiet day is one draw, not a result — which is why every comparison
    above carries an interval and every day is also shown on its own.</li>
    <li><strong>The days do not cover the same hours.</strong> Collection began
    in the evening on the first day and ran round the clock afterwards, so a
    difference between days mixes the date with the time of day. Both matter: DB’s
    yes/no on connections is far more often right during the day than in the
    evening, so the pooled figure for missed connections depends on the mix.</li>
    <li><strong>Every feasible pair counts as a connection</strong>, including
    changes nobody would make. That inflates how often connections are caught and
    flatters both forecasters against real journeys.</li>
    <li><strong>DB’s forecast is read from IRIS</strong>, the feed behind the
    station displays, not from the Navigator app itself. The two normally agree;
    the cross-check that would confirm it was unavailable while this ran.</li>
    <li><strong>Twenty stations</strong>, chosen before any data existed but still
    twenty. Nothing here is weighted to how often people actually travel.</li>
  </ul>
</section>

<section>
  <h2>Reproducing this</h2>
  <p>Every figure above comes from the commands below. The collector must have
  been running on the day in question; the archive publishes the ground truth the
  following morning.</p>
  <pre>%s</pre>
</section>

<footer>
  Generated %s from %s. BayesianBahn is MIT-licensed; delay data is CC BY 4.0 by
  Deutsche Bahn via the piebro/deutsche-bahn-data archive.
</footer>
</div>""" % (
        SURPRISE_MINUTES, TRANSFER_MINUTES,
        "One day" if len(days) == 1 else f"{len(days)} days",
        html.escape(
            "# collect (runs continuously, survives restarts and power cuts)\n"
            "pixi run -e evaluate collect\n"
            "pixi run -e evaluate collect-status      # is it still healthy?\n\n"
            "# once the archive has published the day, score it and rebuild\n"
            "# this page — stages already done are skipped, so adding a day\n"
            "# costs only that day\n"
            "pixi run -e evaluate evaluate " + " ".join(days)),
        dt.date.today().isoformat(),
        html.escape(", ".join(days)),
    ))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(doc), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored-dir", type=Path, required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    live, blind, conn_live, conn_blind = [], [], [], []
    for day in args.days:
        base = args.scored_dir / day
        live += load(base / "arrivals-live.jsonl")
        blind += load(base / "arrivals-blind.jsonl")
        conn_live += load(base / "connections-live.jsonl")
        conn_blind += load(base / "connections-blind.jsonl")
    if not blind:
        raise SystemExit(f"no scored arrivals under {args.scored_dir} for {args.days}")

    render(args.days, arrivals_table(live, blind),
           connections_table(conn_live, conn_blind),
           outcome_split(conn_live, conn_blind),
           {"events": len(blind), "connections": len(conn_blind)}, args.out,
           gaps=headline(live, blind, conn_live, conn_blind),
           daily=per_day(args.days, args.scored_dir))


if __name__ == "__main__":
    main()
