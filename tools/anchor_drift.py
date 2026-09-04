"""Watch the one number the archive cannot tell us: DB's own forecast error.

The app re-anchors a prediction on DB's live delay report and treats that number
as exact. It is not — `tools/calibrate_live.py` measured the spread of
(final delay − DB's report) at eleven to twelve minutes against a stated
interval two minutes wide. Fixing that means baking a spread-versus-lead curve
into the model, and a baked-in curve is a claim about the world that can go
stale: a timetable change, a new interlocking, a different disruption regime.

Nothing in the published archive can check it. The archive stores what trains
did, not what DB said they would do, so the residual exists only in the
collector's journal. This is therefore the drift monitor for that curve, and it
is what the collector runs for once the calibration is shipped: recompute the
curve on recent days, compare it against the reference committed alongside, and
say whether the difference is bigger than the sampling noise.

Two things make the comparison fair rather than merely arithmetic:

  * **The same window.** The collector runs a six-hour evening slice, not the
    whole day. Peak-hour residuals are not all-day residuals, so the reference
    is computed from the same hours of the historical journals — see
    `read_day(..., within=)`.
  * **A cluster bootstrap over trips.** One train contributes an event at every
    probed lead time, and those errors are the same train's, so an
    observation-level interval would be far too tight. Resampling whole trips
    keeps the correlation.

A bin is only reported as drifted when the reference width falls outside the
bootstrap interval *and* moved by more than [REL_TOLERANCE]. Statistical
significance alone is not the bar: with tens of thousands of events a half-minute
shift is detectable and means nothing for a model that answers in whole minutes.

Usage:
    python tools/anchor_drift.py reference --days 2026-08-18..2026-08-31
    python tools/anchor_drift.py check --days 2026-09-01..2026-09-07
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import collect_forecasts as cf  # noqa: E402
import score_events as se  # noqa: E402

REFERENCE = Path(__file__).parent / "anchor-reference.json"

# Below this the app does not anchor on the report at all: `fchg` states a stop
# in four shapes and three of them are the timetable restated, so a reported
# delay under a minute is not evidence (`LiveReport.MIN_INFORMATIVE_DELAY`).
# Conditioning on it here is not a detail. The unconditional residual is
# dominated by the 96% of stops DB calls on time, and its spread saturates
# around 9 minutes by an hour out; the residual DB is *actually anchored on*
# keeps widening past 50. A monitor watching the first would sit still through a
# regime change in the second.
MIN_REPORT = 1.0

# The hours the collector covers, from the collector, so that clipping a
# whole-day journal here reproduces what a scheduled run would have seen. A
# second copy of these numbers is the one way this comparison goes quietly
# wrong: a peak-hour curve measured against an all-day one differs by more than
# any drift it is looking for.
WINDOW_HOURS = cf.WINDOW_HOURS

# Lead-time bins, minutes — the grid `calibrate_live.py` fitted on, kept
# identical so the two are directly comparable.
EDGES = (0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 10 ** 9)
EDGE_LABELS = ("0-5", "5-10", "10-15", "15-20", "20-30", "30-45", "45-60",
               "60-90", "90-120", "120+")

# Lead times probed, minutes. Denser than `score_events.HORIZONS`, which exists
# to report four headline numbers; here the curve itself is the output, so the
# axis is sampled rather than sampled from. Every event is binned on the lead it
# really had, not on the tau that produced it — a ten-minute cadence means a
# reading is up to ten minutes older than asked for.
PROBES = tuple(range(0, 181, 5))

# A bin below this is reported as thin rather than drifted. Quantile spreads on
# a few dozen correlated events move on their own.
MIN_EVENTS = 300

# Relative change a bin must show before the monitor calls it drift, on top of
# being outside the bootstrap interval. Sized against the noise actually
# measured rather than picked: splitting the blockade fortnight in half moved
# the long-lead widths from 9 to 11 minutes, a 22% swing between two adjacent
# weeks of the same regime. The end of the blockade moved them by 35%. The
# threshold has to sit between those, and the check window has to be long
# enough that a single odd week cannot carry it — hence [MIN_DAYS].
REL_TOLERANCE = 0.30

# Days a window needs before its curve is worth comparing. Below this the
# week-to-week swing above dominates whatever is being looked for.
MIN_DAYS = 14

BOOTSTRAP = 400
CI = 0.90


def window_of(day: dt.date, hours: tuple[int, int] = WINDOW_HOURS
              ) -> tuple[float, float]:
    """The collection window on `day`, as epoch seconds."""
    start, end = hours
    return (dt.datetime.combine(day, dt.time(start), se.BERLIN).timestamp(),
            dt.datetime.combine(day, dt.time(end), se.BERLIN).timestamp())


def residuals(out: Path, days: list[dt.date], hours=WINDOW_HOURS,
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lead, residual, trip index) over `days`, from settled truth.

    The residual is DB's error in the sense the model needs it: final delay
    minus the delay DB was reporting `lead` minutes out, over the stops the app
    would actually anchor on — those DB called at least [MIN_REPORT] minute
    late.

    Truth is the settled forecast, not the archive: the archive lands weeks
    later and this has to be answerable the morning after a run. Settling is
    already guarded — `Stop.settled` returns nothing unless polling continued
    well past the predicted arrival — so a train the window cut off simply does
    not appear.

    Truth being DB's own last word rather than the train's real arrival still
    makes these widths narrower than the ones fitted against the archive, since
    DB's last word sits closer to DB's earlier word than the arrival does. The
    monitor therefore watches the curve *move*; it does not restate the
    calibration, and constants to ship should be fitted by
    `tools/sensitivity_live.py` against archive truth.
    """
    leads, errs, trips = [], [], []
    index: dict[str, int] = {}
    for day in days:
        stops, polls = se.read_day(out, day, within=window_of(day, hours))
        if not stops:
            continue
        for event in se.build_events(stops, polls, horizons=PROBES):
            if event["cancelled"] or event["settled"] is None:
                continue
            if event["db"] < MIN_REPORT:
                continue
            key = f"{day}:{event['trip']}"
            leads.append(event["lead"])
            errs.append(event["settled"] - event["db"])
            trips.append(index.setdefault(key, len(index)))
    return (np.array(leads, float), np.array(errs, float),
            np.array(trips, np.int64))


def bin_of(lead: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(EDGES[1:-1], float), lead, side="right")


def widths(err: np.ndarray, bins: np.ndarray) -> list[float | None]:
    """Central 80% width of the residual per bin; None where there is none."""
    out: list[float | None] = []
    for i in range(len(EDGE_LABELS)):
        v = err[bins == i]
        if v.size == 0:
            out.append(None)
        else:
            lo, hi = np.quantile(v, [0.1, 0.9])
            out.append(float(hi - lo))
    return out


def isotonic(values: list[float | None]) -> list[float | None]:
    """Widths made non-decreasing in lead, by pooling adjacent violators.

    The physics is not in dispute: a forecast cannot become less certain as the
    event it describes gets closer, so a dip is sampling noise. Reported
    alongside the raw curve rather than instead of it — when the two disagree
    badly the sample is telling us something about itself.
    """
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    y = [v for _, v in present]
    w = [1.0] * len(y)
    i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1]:
            total = w[i] + w[i + 1]
            y[i] = (y[i] * w[i] + y[i + 1] * w[i + 1]) / total
            w[i] = total
            del y[i + 1], w[i + 1]
            # Merging can break monotonicity with the block before it.
            i = max(i - 1, 0)
        else:
            i += 1
    out: list[float | None] = list(values)
    at = 0
    for value, weight in zip(y, w):
        for _ in range(int(round(weight))):
            out[present[at][0]] = value
            at += 1
    return out


def bootstrap(err: np.ndarray, bins: np.ndarray, trips: np.ndarray, *,
              draws: int = BOOTSTRAP, ci: float = CI, seed: int = 0,
              ) -> list[tuple[float, float] | None]:
    """Per-bin interval for the width, resampling whole trips.

    Whole trips because a train appears once per probed lead time and its
    errors are one train's luck. Resampling events would treat 37 readings of
    the same delayed ICE as 37 independent facts and shrink the interval by
    about six.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(trips, kind="stable")
    err, bins, trips = err[order], bins[order], trips[order]
    starts = np.searchsorted(trips, np.arange(trips[-1] + 1), side="left")
    ends = np.searchsorted(trips, np.arange(trips[-1] + 1), side="right")
    sizes = ends - starts
    n = len(starts)
    samples: list[list[float]] = [[] for _ in EDGE_LABELS]
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        take = np.repeat(starts[pick], sizes[pick]) + _offsets(sizes[pick])
        for i, width in enumerate(widths(err[take], bins[take])):
            if width is not None:
                samples[i].append(width)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return [None if not s else (float(np.quantile(s, lo)), float(np.quantile(s, hi)))
            for s in samples]


def _offsets(sizes: np.ndarray) -> np.ndarray:
    """0,1,..,k-1 for each k in `sizes`, concatenated."""
    total = int(sizes.sum())
    out = np.ones(total, np.int64)
    out[0] = 0
    out[np.cumsum(sizes)[:-1]] = 1 - sizes[:-1]
    return np.cumsum(out)


def curve(out: Path, days: list[dt.date], hours=WINDOW_HOURS, seed: int = 0
          ) -> dict:
    lead, err, trips = residuals(out, days, hours)
    if lead.size == 0:
        return {"days": [str(d) for d in days], "hours": list(hours),
                "events": 0, "trips": 0, "bins": []}
    bins = bin_of(lead)
    counts = [int((bins == i).sum()) for i in range(len(EDGE_LABELS))]
    raw = widths(err, bins)
    return {
        "days": [str(d) for d in days],
        "hours": list(hours),
        "edges": list(EDGES[:-1]) + ["inf"],
        "events": int(lead.size),
        "trips": int(len(set(trips.tolist()))),
        "bins": [
            {"label": label, "n": n, "width": w, "isotonic": m,
             "ci": list(c) if c else None}
            for label, n, w, m, c in zip(
                EDGE_LABELS, counts, raw, isotonic(raw),
                bootstrap(err, bins, trips, seed=seed))
        ],
    }


def compare(now: dict, reference: dict) -> list[dict]:
    """Bins whose width left the reference behind, with why."""
    was = {b["label"]: b for b in reference["bins"]}
    verdicts = []
    for b in now["bins"]:
        before = was.get(b["label"])
        state, note = "ok", ""
        if before is None or before["width"] is None or b["width"] is None:
            state, note = "new", "no reference for this bin"
        elif b["n"] < MIN_EVENTS:
            state, note = "thin", f"{b['n']} events, need {MIN_EVENTS}"
        else:
            rel = (b["width"] - before["width"]) / before["width"]
            outside = b["ci"] is not None and not (
                b["ci"][0] <= before["width"] <= b["ci"][1])
            if outside and abs(rel) >= REL_TOLERANCE:
                state = "drift"
            elif outside:
                note = f"significant but small ({rel:+.0%})"
            note = note or f"{rel:+.0%}"
        verdicts.append({"label": b["label"], "state": state, "note": note,
                         "n": b["n"], "width": b["width"],
                         "was": None if before is None else before["width"],
                         "ci": b["ci"]})
    return verdicts


def table(verdicts: list[dict]) -> str:
    rows = ["| lead (min) | events | width now | reference | 90% interval | |",
            "|---|---:|---:|---:|---|---|"]
    mark = {"drift": "**drift**", "thin": "thin", "new": "new", "ok": ""}
    for v in verdicts:
        ci = "" if not v["ci"] else "%.1f-%.1f" % tuple(v["ci"])
        now = "-" if v["width"] is None else "%.1f" % v["width"]
        was = "-" if v["was"] is None else "%.1f" % v["was"]
        rows.append(f"| {v['label']} | {v['n']} | {now} | {was} | {ci} "
                    f"| {mark[v['state']]} {v['note']} |")
    return "\n".join(rows)


def parse_days(text: str) -> list[dt.date]:
    if ".." in text:
        lo, hi = (dt.date.fromisoformat(p) for p in text.split(".."))
        return [lo + dt.timedelta(days=i) for i in range((hi - lo).days + 1)]
    return [dt.date.fromisoformat(p) for p in text.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["reference", "check"])
    ap.add_argument("--days", required=True,
                    help="YYYY-MM-DD..YYYY-MM-DD or a comma-separated list")
    ap.add_argument("--out", type=Path, default=cf.OUT,
                    help="collector journal directory")
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--note", default="", help="recorded in a new reference")
    args = ap.parse_args()

    days = parse_days(args.days)
    now = curve(args.out, days)
    print(f"{now['events']} events over {now['trips']} trips, "
          f"{len(now['days'])} day(s), {now['hours'][0]}:00-{now['hours'][1]}:00",
          file=sys.stderr)

    if args.command == "reference":
        now["note"] = args.note
        args.reference.write_text(json.dumps(now, indent=2) + "\n")
        print(f"wrote {args.reference}", file=sys.stderr)
        return 0

    if not args.reference.exists():
        print(f"no reference at {args.reference}", file=sys.stderr)
        return 2
    reference = json.loads(args.reference.read_text())
    verdicts = compare(now, reference)
    print(table(verdicts))
    for name, window in (("the reference", reference), ("this window", now)):
        if len(window["days"]) < MIN_DAYS:
            print(f"\nNote: {name} covers {len(window['days'])} day(s), "
                  f"fewer than the {MIN_DAYS} a stable curve needs.")
    if reference.get("note"):
        print(f"\nReference note: {reference['note']}")
    drifted = [v for v in verdicts if v["state"] == "drift"]
    if now["events"] < MIN_EVENTS:
        print(f"\nToo little data to judge: {now['events']} events.")
        return 2
    if drifted:
        print("\nDrifted: " + ", ".join(v["label"] for v in drifted)
              + " min. The live-anchor spread is no longer what "
              f"`{args.reference.name}` says; see notes/COLLECTOR.md.")
        return 1
    print("\nNo drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
