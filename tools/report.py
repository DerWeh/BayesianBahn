"""Render the DB comparison as a self-contained HTML report.

The numbers in this evaluation only mean something alongside their definitions,
their caveats and the commands that produced them, so the report carries all
three. It is generated, never hand-edited: rerun it after another day of
collection and the figures move on their own.

Input layout, which tools/run_evaluation.py produces:

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
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).parent))

from score_events import (  # noqa: E402
    BERLIN, BUCKET_LABELS, SURPRISE_MINUTES, TRANSFER_MINUTES, anchor_minutes,
    bucket_of, wall_to_epoch,
)

# Categorical slots 1-3 of the reference palette, which validate on all pairs in
# both modes. Three series is also the cap here: DB, and the model with and
# without the live signal.
SERIES = {
    "db": ("DB", "var(--series-1)"),
    "blind": ("BayesianBahn, history only", "var(--series-2)"),
    "live": ("BayesianBahn, as shipped", "var(--series-3)"),
    # Not a fourth comparison series: the hour-of-day chart plots one measured
    # quantity with no comparison, so it reuses the first slot and no legend.
    "mean": ("Mean arrival delay", "var(--series-1)"),
}

ANCHOR = "planned_departure"


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]



def mean(values) -> float:
    """Arithmetic mean over a numeric iterable.

    Not statistics.mean: that sums in exact rational arithmetic, which is the
    right default for a statistics library and about a hundred times the cost of
    what is needed for floats that came out of a JSON file. It was 312 of the
    report's 352 seconds before the bootstrap was vectorised, and 1.9 of the
    remaining 10 afterwards.
    """
    a = np.fromiter(values, dtype=float)
    return float(a.mean()) if a.size else float("nan")


def by_lead(rows: list[dict]) -> dict[str, list[dict]]:
    """Group scored events by how long before departure they were made."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        at = anchor_minutes(r, ANCHOR)
        if at is None:
            continue
        label = bucket_of((wall_to_epoch(at) - r["read_at"]) / 60)
        if label:
            out.setdefault(label, []).append(r)
    return out


def arrivals_table(live: list[dict], blind: list[dict]) -> list[dict]:
    """Per lead bucket: DB's error, and both variants of ours."""
    by_live, by_blind = by_lead(live), by_lead(blind)
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
            "db": mean(abs(r["db"] - r["truth"]) for r in g_blind),
            "db_bias": mean(r["db"] - r["truth"] for r in g_blind),
            "db_surprise": sum(1 for r in g_blind
                               if r["truth"] > r["db"] + SURPRISE_MINUTES) / len(g_blind),
            "blind": mean(r["crps"] for r in g_blind),
            "blind_cover": sum(1 for r in g_blind
                               if r["q10"] <= r["truth"] <= r["q90"]) / len(g_blind),
            "live": mean(r["crps"] for r in g_live),
            "live_cover": sum(1 for r in g_live
                              if r["q10"] <= r["truth"] <= r["q90"]) / len(g_live),
        })
    return rows


BANDS = ((2, 6, "2-5 min"), (6, 11, "6-10 min"), (11, 21, "11-20 min"),
         (21, 31, "21-30 min"))



# Where a forecast stops being a rounding error and starts costing a train.
BAD_MINUTES = 5
AWFUL_MINUTES = 15


def score_spread(rows: list[dict], value) -> dict:
    """Quantiles of a per-event score, plus how often it goes badly wrong."""
    a = np.sort(np.fromiter((value(r) for r in rows), dtype=float, count=len(rows)))
    q = lambda p: float(np.quantile(a, p))  # noqa: E731
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "p25": q(0.25), "p50": q(0.50), "p75": q(0.75),
        "p10": q(0.10), "p90": q(0.90), "p99": q(0.99),
        "max": float(a[-1]),
        "bad": float(np.mean(a > BAD_MINUTES)),
        "awful": float(np.mean(a > AWFUL_MINUTES)),
    }


def error_spread(live: list[dict]) -> list[dict]:
    """Per lead bucket, the whole distribution of the score, not its mean.

    The mean is the number the table above reports, and on its own it is
    misleading in both directions: half of both forecasts are inside a minute,
    so the median is a tie, and everything that separates them sits in the
    upper tail. A passenger does not notice the minute; they notice the
    twenty-minute miss nobody flagged.
    """
    rows = []
    for label, group in by_lead(live).items():
        rows.append({
            "bucket": label,
            "n": len(group),
            "db": score_spread(group, lambda r: abs(r["db"] - r["truth"])),
            "live": score_spread(group, lambda r: r["crps"]),
        })
    return [r for r in sorted(rows, key=lambda r: BUCKET_LABELS.index(r["bucket"]))]


def brier(rows: list[dict], key: str) -> float:
    return mean((r[key] - (1.0 if r["caught"] else 0.0)) ** 2 for r in rows)


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


# Draws per vectorised chunk. The bootstrap allocates a (chunk x clusters)
# count matrix, so this bounds peak memory independently of how many days have
# been collected.
BOOTSTRAP_CHUNK = 500


def cluster_ci(rows: list[dict], value, *, draws: int = 2000,
               seed: int = 20260819) -> tuple[float, float, float]:
    """Mean of `value`, with a 95% interval from a bootstrap over *trains*.

    Resampling events would be wrong and flattering: one late ICE contributes a
    dozen correlated predictions, so events are nowhere near independent and an
    event-level interval comes out far too narrow. Resampling whole trains keeps
    each cluster intact. Seeded, so the published interval is reproducible.

    A resampled mean depends only on how many times each train was drawn, not
    on the order they came out, so a draw is a multinomial count vector over the
    trains and the whole bootstrap is two matrix products against the per-train
    sums and sizes. Written as a Python loop over `statistics.mean` this was
    346 of the report's 352 seconds — `statistics.mean` sums in exact rational
    arithmetic, which is the right default for a statistics library and about a
    hundred times the cost of what is needed here.
    """
    sums: dict[tuple, float] = {}
    sizes: dict[tuple, int] = {}
    for r in rows:
        key = (r["cat"], r["num"])
        sums[key] = sums.get(key, 0.0) + value(r)
        sizes[key] = sizes.get(key, 0) + 1
    group_sum = np.fromiter(sums.values(), dtype=float, count=len(sums))
    group_size = np.fromiter(sizes.values(), dtype=float, count=len(sizes))
    clusters = len(group_sum)
    point = float(group_sum.sum() / group_size.sum())

    rng = np.random.default_rng(seed)
    share = np.full(clusters, 1.0 / clusters)
    means = np.empty(draws)
    for start in range(0, draws, BOOTSTRAP_CHUNK):
        take = min(BOOTSTRAP_CHUNK, draws - start)
        counts = rng.multinomial(clusters, share, size=take)
        means[start:start + take] = counts @ group_sum / (counts @ group_size)
    means.sort()
    return point, float(means[int(0.025 * draws)]), float(means[int(0.975 * draws)])


def crps_gap(rows: list[dict]) -> tuple[float, float, float]:
    """BayesianBahn's CRPS minus DB's, per prediction. Negative is the lower score."""
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
            # Stated as which score came out lower, not as a winner: the
            # interval decides whether there is a difference at all.
            "verdict": "BayesianBahn lower" if hi < 0 else
                       ("DB lower" if lo > 0 else "not separated"),
        })
    return out


def of_day(rows: list[dict], day: str) -> list[dict]:
    return [r for r in rows if r["day"] == day]


def per_day(days: list[str], live, blind, conn_live, conn_blind) -> list[dict]:
    """One row per collected day: the same headline numbers, unpooled.

    Pooling hides whether a result is a property of the model or of one day's
    weather. A claim that moves when a day is added was never a claim.

    Takes the rows the caller already loaded rather than the directory they came
    from; reading the same twelve files a second time was a third of what the
    report spent once the bootstrap stopped dominating it.
    """
    out = []
    for day in days:
        day_live, day_blind = of_day(live, day), of_day(blind, day)
        day_cl, day_cb = of_day(conn_live, day), of_day(conn_blind, day)
        if not day_blind:
            continue
        missed_l = [r for r in day_cl if r["caught"] is False]
        missed_b = [r for r in day_cb if r["caught"] is False]
        out.append({
            "day": day,
            "n": len(day_blind),
            "db": mean(abs(r["db"] - r["truth"]) for r in day_blind),
            "blind": mean(r["crps"] for r in day_blind),
            "live": mean(r["crps"] for r in day_live),
            "live_cover": sum(1 for r in day_live
                              if r["q10"] <= r["truth"] <= r["q90"]) / len(day_live),
            "blind_cover": sum(1 for r in day_blind
                               if r["q10"] <= r["truth"] <= r["q90"]) / len(day_blind),
            "missed": len(missed_b),
            "db_missed": brier(missed_b, "db_catch_p") if missed_b else float("nan"),
            "blind_missed": brier(missed_b, "p_catch") if missed_b else float("nan"),
            "live_missed": brier(missed_l, "p_catch") if missed_l else float("nan"),
        })
    return out


FULL_DAY_HOURS = 20

# The collected stations run almost no service in the small hours: 02:00 held
# two trains across the whole collection, and their mean was the largest number
# on the chart. An hour has to carry enough trains to mean anything before it
# earns a bar, let alone a sentence.
MIN_HOUR_EVENTS = 100


def full_days(rows: list[dict]) -> set[str]:
    """Days whose events span the clock.

    The first collected day started in the evening. Averaging it into an
    hour-of-day curve would put its (busy, late) evening on the same footing as
    a full day's evening while contributing nothing before 18:00, which tilts
    the whole shape. A day earns its place here by covering the clock.
    """
    seen: dict[str, set[int]] = {}
    for r in rows:
        seen.setdefault(r["day"], set()).add(hour_of(r))
    return {day for day, hours in seen.items() if len(hours) >= FULL_DAY_HOURS}


def hour_of(row: dict) -> int:
    """Local hour of the train's planned arrival."""
    return dt.datetime.fromtimestamp(wall_to_epoch(row["planned"]), BERLIN).hour


def hourly(rows: list[dict]) -> list[dict]:
    """Mean arrival delay by hour of day, one row per train event.

    Deduplicated on the event: a train polled thirty times is one arrival, not
    thirty, and counting the polls would weight the curve by how long each
    train sat in the collector's window rather than by how late it was.
    """
    days = full_days(rows)
    if not days:
        return []
    events: dict[tuple, dict] = {}
    for r in rows:
        if r["day"] in days:
            events[(r["day"], r["eva"], r["cat"], r["num"], r["planned"])] = r
    by_hour: dict[int, list[dict]] = {}
    for r in events.values():
        by_hour.setdefault(hour_of(r), []).append(r)
    out = []
    for hour in sorted(by_hour):
        group = by_hour[hour]
        if len(group) < MIN_HOUR_EVENTS:
            continue
        out.append({
            "bucket": f"{hour:02d}",
            "hour": hour,
            "n": len(group),
            "mean": mean(r["truth"] for r in group),
            "late": sum(1 for r in group if r["truth"] > SURPRISE_MINUTES) / len(group),
        })
    return out


def band_spread(rows: list[dict], bands: dict[str, tuple]) -> list[dict]:
    """Within-band spread of the hourly means — how much a band averages away.

    The model buckets its history by time band, which is only sound if the
    hours inside a band behave alike. The spread is the evidence for or against
    the cut that ships.
    """
    means = {r["hour"]: r["mean"] for r in rows}
    out = []
    for name, (label, hours) in bands.items():
        inside = [means[h] for h in hours if h in means]
        if not inside:
            continue
        out.append({
            "band": name,
            "hours": label,
            "lo": min(inside),
            "hi": max(inside),
            "spread": max(inside) - min(inside),
        })
    return out


# Mirrors TimeBand.fromEpochMillis in DelayModel.kt. A drift here would make the
# report describe a cut the app does not use.
SHIPPED_BANDS = {
    "MORNING_PEAK": ("06-08", range(6, 9)),
    "MIDDAY": ("09-15", range(9, 16)),
    "EVENING_PEAK": ("16-18", range(16, 19)),
    "NIGHT": ("19-05", tuple(range(19, 24)) + tuple(range(0, 6))),
}


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
            "blind_p": mean(r["p_catch"] for r in g_blind),
            "live": brier(g_live, "p_catch") if g_live else float("nan"),
            "live_p": mean(r["p_catch"] for r in g_live) if g_live else float("nan"),
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



def box_chart(rows, keys, *, y_label, height=300) -> str:
    """Box-and-whisker per category: box p25-p75, median rule, whiskers p10-p90.

    The whiskers stop at p90 on purpose. p99 is three to four times p90 here, so
    including it would flatten every box into a line and hide the comparison the
    chart exists to make; the tail past the whisker is given as numbers in the
    table underneath, where it can be read exactly rather than squinted at.
    """
    width, pad_l, pad_r, pad_t, pad_b = 720, 54, 16, 16, 44
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    top = max(r[k]["p90"] for r in rows for k in keys) * 1.15
    group_w = plot_w / len(rows)
    box_w = min(30.0, (group_w - 18) / len(keys))

    def y(v):
        return pad_t + plot_h - (min(v, top) / top) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(y_label)}">']
    for g in range(5):
        v = top * g / 4
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" '
                     f'y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 8}" y="{y(v) + 4:.1f}" '
                     f'text-anchor="end">{v:.1f}</text>')

    for i, row in enumerate(rows):
        centre = pad_l + group_w * (i + 0.5)
        span = box_w * len(keys) + 2 * (len(keys) - 1)
        for j, key in enumerate(keys):
            label, colour = SERIES[key]
            d = row[key]
            bx = centre - span / 2 + j * (box_w + 2)
            mid = bx + box_w / 2
            title = (f'{label} — {row["bucket"]}: median {d["p50"]:.2f}, '
                     f'p25-p75 {d["p25"]:.2f}-{d["p75"]:.2f}, '
                     f'p90 {d["p90"]:.2f}, p99 {d["p99"]:.2f} min')
            # Whisker behind the box, with caps at each end.
            parts.append(f'<line x1="{mid:.1f}" x2="{mid:.1f}" y1="{y(d["p10"]):.1f}" '
                         f'y2="{y(d["p90"]):.1f}" stroke="{colour}" stroke-width="2"/>')
            for end in ("p10", "p90"):
                parts.append(f'<line x1="{mid - box_w / 4:.1f}" x2="{mid + box_w / 4:.1f}" '
                             f'y1="{y(d[end]):.1f}" y2="{y(d[end]):.1f}" '
                             f'stroke="{colour}" stroke-width="2"/>')
            parts.append(
                f'<rect x="{bx:.1f}" y="{y(d["p75"]):.1f}" width="{box_w:.1f}" '
                f'height="{max(2.0, y(d["p25"]) - y(d["p75"])):.1f}" rx="3" '
                f'fill="{colour}" class="marker">'
                f'<title>{html.escape(title)}</title></rect>')
            # Median in the surface colour so it reads against the fill.
            parts.append(f'<line x1="{bx:.1f}" x2="{bx + box_w:.1f}" '
                         f'y1="{y(d["p50"]):.1f}" y2="{y(d["p50"]):.1f}" '
                         f'stroke="var(--panel)" stroke-width="2"/>')
        parts.append(f'<text class="tick" x="{centre:.1f}" y="{height - pad_b + 20}" '
                     f'text-anchor="middle">{html.escape(row["bucket"])}</text>')
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


def pct(x, digits=0):
    return "—" if x != x else f"{x * 100:.{digits}f}%"


def num(x, digits=2):
    return "—" if x != x else f"{x:.{digits}f}"




def provenance() -> dict:
    """Which code produced these numbers.

    A scored page and the model that produced it drift apart the moment either
    changes, and this page is generated from the working tree rather than from a
    release — the first five-day run scored a model that was not in any released
    version and the page called it "the one that ships". Reporting the commit,
    the tree's cleanliness and the app version it declares is what makes a
    figure on this page traceable to the code that produced it.
    """
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    gradle = (ROOT / "app/build.gradle.kts").read_text(encoding="utf-8")
    name = re.search(r'versionName = "([^"]+)"', gradle)
    code = re.search(r"versionCode = (\d+)", gradle)
    commit = git("rev-parse", "HEAD")
    # Only files that affect the model or the scoring count as dirty here; an
    # edited README does not change a number on this page.
    dirty = [line[3:] for line in git("status", "--porcelain").splitlines()
             if line[3:].startswith(("app/src/main/", "tools/", "pipeline/"))]
    described = git("describe", "--tags", "--exact-match") or ""
    return {
        "commit": commit,
        "short": commit[:12],
        "version": name.group(1) if name else "unknown",
        "code": code.group(1) if code else "?",
        "tag": described,
        "dirty": sorted(dirty),
    }


def weekday_caveat(days: list[str]) -> str:
    """Which parts of the week the collected days actually cover."""
    kinds = {dt.date.fromisoformat(d).weekday() >= 5 for d in days}
    if kinds == {False}:
        return "Weekdays only."
    if kinds == {True}:
        return "Weekend days only."
    return "Weekdays and weekend days are pooled."


def render(days, arrivals, connections, split, totals, out: Path, *,
           gaps=(), daily=(), clock=(), spread=()) -> None:
    span = ", ".join(days)
    cross = next((r["bucket"] for r in arrivals if r["blind"] < r["db"]), None)
    missed = next((r for r in split if "missed" in r["outcome"]), None)

    prov = provenance()
    if prov["dirty"]:
        release_note = (
            "<strong>That commit does not identify the code that ran:</strong> "
            f"{len(prov['dirty'])} uncommitted file(s) under app/src/main, tools/ "
            "or pipeline/ were present, so these figures cannot be reproduced "
            "from the commit alone.")
    elif prov["tag"]:
        release_note = (f"That commit is the released tag <code>{html.escape(prov['tag'])}</code>, "
                        "so this is the model in that version of the app.")
    else:
        release_note = ("That commit is <strong>not a released version</strong>: the app "
                        "published in the stores does not contain this model unless a "
                        "later release says so.")

    tiles = [
        ("tile", f"{totals['events']:,}", "arrival predictions scored"),
        ("tile", f"{totals['connections']:,}", "one-change connections scored"),
        ("tile", "history only" if cross else "—",
         f"scores below DB from {cross} before departure" if cross
         else "no crossover found"),
    ]
    if missed:
        tiles.append(("tile flag", pct(missed["db_right"]),
                      "of missed connections DB called correctly"))

    doc = [f"""<title>Forecasts against DB's own</title>
<style>{STYLE}</style>
<div class="wrap">
<header>
  <p class="eyebrow">BayesianBahn · evaluation · {html.escape(span)}</p>
  <h1>Forecasts against DB’s own</h1>
  <p class="lede">BayesianBahn’s arrival forecasts and DB’s, scored against what
  the trains actually did. Sample sizes, method and the limits of each figure
  are given alongside it.</p>
</header>

<section>
  <h2>What was measured</h2>
  <p>Every ten minutes, DB’s own forecast was recorded for twenty stations chosen
  in advance across the whole network — six major hubs down to three village
  halts. The next day the archive says when each train really arrived, and both
  forecasts are scored against that.</p>
  <p>The model is the app’s own prediction code at commit
  <code>{prov["short"]}</code>, which declares version {prov["version"]}
  (versionCode {prov["code"]}). {release_note} It is only ever shown history
  from before the day it predicts. It appears twice:
  <strong>as shipped</strong>, which adjusts DB’s live number using past runs
  <em>when DB actually reports a delay</em> and leans on the history alone when
  it does not, and <strong>history only</strong>, which never looks at the live
  number at all.</p>
  <p>Two kinds of answer are scored, and they correspond to the two kinds of
  journey the app plans. For a journey without a change the answer is an arrival
  time, scored as a distribution against the arrival that happened. For a journey
  with one change the answer is the probability of making that change, scored
  against whether it was made. The two use different scores and are not
  comparable with each other; each is compared only with DB’s answer to the same
  question. A complete two-leg journey — the predicted arrival at the far end of
  a change, against the arrival that happened — is <em>not</em> scored here, and
  would need the connecting train’s destination in the collected data.</p>
  <div class="tiles">"""]
    for cls, k, v in tiles:
        doc.append(f'<div class="{cls}"><span class="k">{html.escape(k)}</span>'
                   f'<span class="v">{html.escape(v)}</span></div>')
    doc.append("</div>\n</section>")

    if gaps:
        doc.append("""
<section>
  <h2>The comparison, with its uncertainty</h2>
  <p>Each row is BayesianBahn’s score minus DB’s over the same predictions, so
  a <strong>negative number is the lower score for BayesianBahn</strong>, and
  lower is better for both scores used here. The interval is what
  decides it: delays arrive in clusters — one late train produces a dozen
  correlated predictions — so the range comes from resampling whole trains, not
  individual predictions. Where the interval crosses zero, the collected days
  are not enough to claim anything, however suggestive the middle number
  looks.</p>""")
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
    worst = max(arrivals, key=lambda r: r["live"] - r["blind"])
    if worst["live"] > worst["blind"]:
        doc.append(f"""
  <p><strong>The two variants cross over in this bucket.</strong> In the
  <em>{html.escape(worst["bucket"])}</em> bucket the shipped model scores
  {worst["live"]:.2f} against history alone at {worst["blind"]:.2f} — leaning on
  DB’s number makes the answer {worst["live"] - worst["blind"]:.2f} minutes
  worse there. That is the shape this model was changed to remove, so a
  crossover appearing here means a reported delay is being believed in a range
  where it should not be.</p>""")
    if spread:
        scored_here = sum(r["n"] for r in spread)
        doc.append(f"""
  <h3>How the errors are distributed</h3>
  <p>The means above summarise a skewed distribution, and the two forecasts
  differ mainly in its upper tail: the medians are close, while the large
  errors are not equally common. Both parts are worth reading, since a forecast
  a minute out and one twenty minutes out have very different consequences for
  a passenger. The box spans the middle half of the
  predictions, the line across it is the median, and the whiskers reach the
  10th and 90th percentiles, so the worst tenth of each forecast reaches past
  the whisker and is given exactly in the table below.</p>""")
        doc.append('<div class="figure">')
        doc.append(legend(["db", "live"]))
        doc.append(box_chart(spread, ["db", "live"],
                             y_label="Error, minutes (lower is better)"))
        doc.append("</div>")
        doc.append(table(spread, [
            ("bucket", "Before departure", lambda r: html.escape(r["bucket"])),
            ("n", "Predictions", lambda r: f"{r['n']:,}"),
            ("dbm", "DB median", lambda r: num(r["db"]["p50"])),
            ("livem", "Our median", lambda r: num(r["live"]["p50"])),
            ("db90", "DB p90", lambda r: num(r["db"]["p90"])),
            ("live90", "Our p90", lambda r: num(r["live"]["p90"])),
            ("db99", "DB p99", lambda r: num(r["db"]["p99"])),
            ("live99", "Our p99", lambda r: num(r["live"]["p99"])),
            # One decimal: the two differ by fractions of a percent in the
            # near buckets, and rounding to whole percent shows them as equal.
            ("dbbad", f"DB over {AWFUL_MINUTES} min",
             lambda r: pct(r["db"]["awful"], 1)),
            ("livebad", f"Ours over {AWFUL_MINUTES} min",
             lambda r: pct(r["live"]["awful"], 1)),
        ]))
        doc.append(f"""
  <p>The last two columns give the share of forecasts out by more than
  {AWFUL_MINUTES} minutes. Over {scored_here:,} predictions, most of which are
  uneventful, this is the part of the distribution a mean is least able to
  convey.</p>""")

    doc.append("""</section>

<section>
  <h2>Does the 80% range hold?</h2>
  <p>The app gives not only a time but a range stated to contain the true
  arrival four times in five. That is checkable: count how often the real
  arrival fell inside it. A bar at the dashed line matches the stated
  probability; below it the range is narrower than the forecast’s accuracy
  supports, and above it wider.</p>""")
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
  <h2>Connections</h2>
  <p>A connection here is a train arriving at one of the sampled stations and
  another leaving it a few minutes later, judged from before the first train set
  off — the last moment at which the answer could still change a decision. DB’s
  timetable answers yes or no; the app answers with a probability.</p>""")
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
  <p>Almost every connection is caught, so answering “yes” every time scores
  well on the pooled average, and a yes/no answer taken from the timetable is
  close to doing that. The two outcomes are therefore worth reading apart: the
  connections that failed are the smaller group and the one the pooled figure
  says least about.</p>""")
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
  <p>Everything above pools the collected days. Pooling gives the more precise
  estimate; it cannot show whether a result is a property of the model or of one
  day’s conditions. Here each day stands alone. A column that points the same
  way in every row is the more durable result; one that changes sign between
  days is within the range of day-to-day variation.</p>
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

    if clock:
        spread = band_spread(clock, SHIPPED_BANDS)
        peak = max(clock, key=lambda r: r["mean"])
        trough = min(clock, key=lambda r: r["mean"])
        doc.append(f"""
<section>
  <h2>Delay through the day</h2>
  <p>Delay is not a property of a train alone: a late train occupies a platform
  and the next one can inherit some of it, so delay accumulated in the morning
  may still be on the network in the afternoon. If so, an hour-of-day term
  describes the data rather than merely fitting it.</p>
  <p>The collected days show that pattern. The mean arrival delay climbs from
  <strong>{trough["mean"]:.2f} min at {trough["bucket"]}:00</strong> to
  <strong>{peak["mean"]:.2f} min at {peak["bucket"]}:00</strong>, a factor of
  {peak["mean"] / max(0.01, trough["mean"]):.1f}, and then drains overnight. Each
  train counts once, at its scheduled arrival hour, however many times it was
  polled. Only days that cover the whole clock are included, and only hours
  carrying at least {MIN_HOUR_EVENTS} trains.</p>""")
        doc.append(bar_chart(clock, ["mean"], y_label="Mean arrival delay, minutes",
                             label_key="bucket", height=260))
        doc.append("""
  <p>The maximum falls in the mid-afternoon rather than at the evening rush.
  One reading is that accumulated delay is worked off after the last peak
  departures, so the worst hour to arrive is set by how much delay the network
  is still carrying rather than by passenger numbers.</p>
  <h3>What that means for the time bands</h3>
  <p>The model keeps separate delay statistics per time band, which only makes
  sense if the hours inside a band resemble each other. <em>Spread</em> is the
  distance between the latest and earliest hourly mean inside the band: a large
  spread means the band is averaging together hours that behave differently.</p>""")
        doc.append(table(spread, [
            ("band", "Band", lambda r: html.escape(r["band"])),
            ("hours", "Hours", lambda r: html.escape(r["hours"])),
            ("lo", "Quietest hour", lambda r: num(r["lo"])),
            ("hi", "Latest hour", lambda r: num(r["hi"])),
            ("spread", "Spread", lambda r: num(r["spread"])),
        ]))
        doc.append("""
  <p>The bands that ship were chosen from the commuter timetable rather than
  from this curve, and do not line up with it: the peak band ends before the
  peak, and the
  night band spans the range from the quietest hour of the night to the tail of
  the evening. This is an observation about the model’s bucketing rather than
  about the trains, and it is only as good as the days collected so far.</p>
</section>""")

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
    for the station, but only when that number is a reported delay: DB states a
    stop in four shapes and three of them mean “on time”, which is the plan
    restated rather than an observation, so those are passed through as no
    report at all. <em>History only</em> withholds the live number in every
    case. The model was built to take the measured delay at the train’s
    <em>previous</em> stop, and the live path substitutes this station’s forecast
    instead — an approximation documented in the source, and where DB does
    report a delay the gap between these two columns is what it costs.</dd>

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
    <li><strong>%s</strong> Traffic, staffing and the timetable itself differ
    between weekdays and weekends, so a figure drawn from one does not transfer
    to the other — the hour-of-day curve above least of all.</li>
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
  Generated %s from %s, by
  <a href="https://github.com/DerWeh/BayesianBahn">BayesianBahn</a> at commit %s.
  MIT-licensed; delay data is CC BY 4.0 by Deutsche Bahn via the
  piebro/deutsche-bahn-data archive. Not affiliated with Deutsche Bahn.
</footer>
</div>""" % (
        SURPRISE_MINUTES, TRANSFER_MINUTES,
        "One day" if len(days) == 1 else f"{len(days)} days",
        weekday_caveat(days),
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
        html.escape(prov["commit"] or "unknown"),
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
    # Every row carries the day it came from: the per-day table splits on it,
    # and the hour-of-day curve uses it to drop days that do not cover the
    # clock. Tagging on load is what lets the files be read exactly once.
    for day in args.days:
        base = args.scored_dir / day
        for target, name in ((live, "arrivals-live"), (blind, "arrivals-blind"),
                             (conn_live, "connections-live"),
                             (conn_blind, "connections-blind")):
            for row in load(base / f"{name}.jsonl"):
                row["day"] = day
                target.append(row)
    if not blind:
        raise SystemExit(f"no scored arrivals under {args.scored_dir} for {args.days}")

    render(args.days, arrivals_table(live, blind),
           connections_table(conn_live, conn_blind),
           outcome_split(conn_live, conn_blind),
           {"events": len(blind), "connections": len(conn_blind)}, args.out,
           gaps=headline(live, blind, conn_live, conn_blind),
           daily=per_day(args.days, live, blind, conn_live, conn_blind),
           clock=hourly(live), spread=error_spread(live))


if __name__ == "__main__":
    main()
