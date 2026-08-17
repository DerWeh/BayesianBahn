"""Turn collected forecast trajectories into scoreable prediction events.

Phase A of the comparison against DB: no model yet, deliberately. It builds the
event table, joins the ground truth, and scores two trivial forecasters — DB's
own number, and "always on time" — so the plumbing and the join rate can be
judged before any Kotlin exists. If the join comes out at 60% the design has to
change, and that is much cheaper to learn now.

The event universe is the *plan*, not the observations. `fchg` reports a stop in
four different shapes, all seen live at Aachen Hbf within one hour:

    ct moved            DB predicts a different time      -> delay = ct - planned
    ct == planned       DB confirms on time               -> delay 0
    present, no ct      entry exists (messages, platform) -> delay 0
    absent entirely     no entry at all                   -> delay 0

Only the first is a delay forecast; the other three all mean on time. Taking
events from what we recorded instead of from the plan would therefore keep the
trains DB flagged and drop the ones it got right — which is precisely the
comparison we are trying to make, biased in our favour.

Two clocks meet here and they are not the same. Journal `at` is real epoch
seconds; IRIS times are wall clock stored as if UTC (see collect_forecasts.
iris_time). Comparing them directly shifts everything by the German UTC offset,
which is one or two hours depending on the season.

Usage:
    python tools/score_events.py events --day 2026-08-17
    python tools/score_events.py score  --day 2026-08-17 --truth settled
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
import zoneinfo
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import collect_forecasts as cf  # noqa: E402
from route_bench import candidate_files  # noqa: E402

BERLIN = zoneinfo.ZoneInfo("Europe/Berlin")

# Lead times to planned arrival, in minutes. The long end is where DB publishes
# no forecast at all and we should win; the short end is where it can see the
# train standing still and we probably should not.
HORIZONS = (1440, 240, 120, 60, 30, 15, 5)

# A forecast still moving this long after the planned arrival is not settled;
# used as provisional truth until the archive lands, and to measure how long
# settling actually takes.
SETTLED_AFTER = 30

# "Materially later than you were told" — the error that misses a connection.
SURPRISE_MINUTES = 5


def wall_to_epoch(minutes: int) -> float:
    """IRIS wall-clock minutes to real epoch seconds.

    The journal timestamps a poll with `time.time()`, while IRIS times are naive
    German local time. Anything comparing the two has to cross this boundary
    exactly once, so it lives in one function with a test pinning both sides.
    """
    naive = dt.datetime(1970, 1, 1) + dt.timedelta(minutes=minutes)
    return naive.replace(tzinfo=BERLIN).timestamp()


@dataclass
class Stop:
    """One scheduled arrival, with DB's forecast for it over time."""

    eva: str
    trip: str
    cat: str
    num: str
    line: str | None
    planned: int                      # wall-clock minutes
    obs: list[tuple[int, int | None, bool]] = field(default_factory=list)

    def forecast_at(self, when: float) -> int | None:
        """DB's predicted delay in minutes as of `when` (epoch seconds).

        None means we cannot tell — no observation and no way to know whether
        DB had said anything by then. The caller decides using the poll log.
        """
        latest = None
        for at, arrival, _cancelled in self.obs:
            if at <= when:
                latest = arrival
            else:
                break
        if latest is None:
            return None
        return latest - self.planned

    def cancelled_at(self, when: float) -> bool:
        state = False
        for at, _arrival, cancelled in self.obs:
            if at <= when:
                state = cancelled
        return state

    def settled(self, last_poll: float | None) -> int | None:
        """DB's final word, once it has had time to become final.

        Only valid if we were still polling the station well after the train
        was due — otherwise "settled" is just the last thing we happened to see,
        which at short lead times is the very forecast being scored, and DB
        would be compared against itself. The cutoff runs from the *predicted*
        arrival, not the planned one: a train forecast 40 minutes late has not
        arrived 30 minutes after its planned time.
        """
        if last_poll is None:
            return None
        expected = self.planned + max(0, self.forecast_at(last_poll) or 0)
        cutoff = wall_to_epoch(expected + SETTLED_AFTER)
        if last_poll < cutoff:
            return None
        return self.forecast_at(cutoff)


def read_day(out: Path, day: dt.date) -> tuple[dict[tuple[str, str], Stop], dict[str, list[float]]]:
    """Stops keyed by (station, trip), plus successful poll times per station."""
    records, torn = cf.Journal.read(out / f"forecasts-{day}.jsonl")
    if torn:
        print(f"note: skipped {torn} torn line(s)", file=sys.stderr)
    stops: dict[tuple[str, str], Stop] = {}
    polls: dict[str, list[float]] = {}
    for r in records:
        if r["t"] == "poll":
            if r.get("ok"):
                polls.setdefault(r["eva"], []).append(float(r["at"]))
        elif r["t"] == "plan":
            if r.get("par") is None:
                continue  # a departure-only stop has no arrival to predict
            stops[(r["eva"], r["id"])] = Stop(
                eva=r["eva"], trip=r["id"], cat=r["cat"], num=r["num"],
                line=r.get("line"), planned=r["par"],
            )
    for r in records:
        if r["t"] != "obs":
            continue
        stop = stops.get((r["eva"], r["id"]))
        if stop is None:
            continue  # observed but never in the plan window; no planned time
        # "present, no ct" is not missing data: it means DB left the time alone.
        arrival = r["ar"] if r["ar"] is not None else stop.planned
        stop.obs.append((float(r["at"]), arrival, bool(r.get("arc"))))
    for stop in stops.values():
        stop.obs.sort()
    for times in polls.values():
        times.sort()
    return stops, polls


def polled_by(times: list[float], when: float) -> bool:
    """Whether the station had answered at least once by `when`."""
    return bool(times) and times[0] <= when


def last_poll_before(times: list[float], when: float) -> float | None:
    """The freshest reading we actually hold at `when`.

    Our knowledge of DB is only as current as the last successful poll, whether
    or not it produced an observation — a stop absent from `fchg` tells us "on
    time as of that poll", not "on time as of now". So the honest lead time runs
    from the poll, and with a ten-minute cadence a nominal five-minute forecast
    is really five to fifteen minutes old.
    """
    found = None
    for at in times:
        if at <= when:
            found = at
        else:
            break
    return found


def build_events(stops: dict, polls: dict, horizons=HORIZONS) -> list[dict]:
    events = []
    for stop in stops.values():
        station_polls = polls.get(stop.eva, [])
        last_poll = station_polls[-1] if station_polls else None
        settled = stop.settled(last_poll)
        for tau in horizons:
            when = wall_to_epoch(stop.planned - tau)
            if not polled_by(station_polls, when):
                continue  # we were not collecting yet; not DB's silence
            read_at = last_poll_before(station_polls, when)
            assert read_at is not None
            db = stop.forecast_at(when)
            # Bucket on what the lead time really was, not what we asked for.
            lead = (wall_to_epoch(stop.planned) - read_at) / 60
            # Absent from fchg, but we were polling: DB is saying on time.
            events.append({
                "eva": stop.eva, "trip": stop.trip, "cat": stop.cat,
                "num": stop.num, "line": stop.line, "planned": stop.planned,
                "tau": tau, "lead": round(lead, 1),
                "db": db if db is not None else 0,
                "db_explicit": db is not None,
                "cancelled": stop.cancelled_at(when),
                "settled": settled,
            })
    return events


def load_truth(data_dirs: list[Path], day: dt.date, evas: set[str]) -> dict:
    """Realised arrival delays from the archive, keyed as the journal keys them.

    Delay is `arrival_change_time - arrival_planned_time`, the same definition
    build_shards.py uses for the history the model learns from — a different one
    here would score the model against a target it was never trained on. A null
    change time is not missing data: it means the train ran to plan, exactly as
    a stop absent from `fchg` does.
    """
    import polars as pl

    files = candidate_files(data_dirs, day)
    if not files:
        raise SystemExit(f"no archive file covering {day}; run the pipeline's "
                         "fetch_raw_day.py for that day first")
    unit = pl.Datetime("us")
    planned = pl.col("arrival_planned_time").cast(unit)
    changed = pl.col("arrival_change_time").cast(unit)
    frames = [
        pl.scan_parquet(f)
        .with_columns(arrival_planned_time=planned, arrival_change_time=changed)
        .filter(planned.dt.date() == day)
        .with_columns(eva0=pl.col("eva").str.strip_chars_start("0"))
        .filter(pl.col("eva0").is_in(list(evas)))
        .select(
            eva=pl.col("eva0"),
            cat=pl.col("train_type"),
            num=pl.col("train_number"),
            # Wall clock as if UTC, the convention collect_forecasts.iris_time
            # stores journal times in; anything else shifts the join by an hour.
            planned=(planned.cast(pl.Int64) // 60_000_000),
            delay=((changed.cast(pl.Int64) - planned.cast(pl.Int64)) // 60_000_000),
            cancelled=pl.col("is_canceled"),
        )
        for f in files
    ]
    rows = pl.concat(frames).unique(subset=["eva", "cat", "num", "planned"]).collect()
    return {
        (r["eva"], r["cat"], r["num"], r["planned"]):
            {"delay": 0 if r["delay"] is None else int(r["delay"]),
             "cancelled": bool(r["cancelled"])}
        for r in rows.iter_rows(named=True)
    }


def attach_truth(events: list[dict], truth: dict) -> dict[str, tuple[int, int]]:
    """Join events to the archive; report the hit rate per train type.

    A type that silently fails to join disappears from the comparison rather
    than failing it, so the rate is reported by type and not just in total.
    """
    rate: dict[str, tuple[int, int]] = {}
    for e in events:
        key = (e["eva"], e["cat"], e["num"], e["planned"])
        found = truth.get(key)
        if found is not None:
            e["archive"] = found["delay"]
            e["cancelled"] = e["cancelled"] or found["cancelled"]
        hit, total = rate.get(e["cat"], (0, 0))
        rate[e["cat"]] = (hit + (found is not None), total + 1)
    return rate


def report_join(rate: dict[str, tuple[int, int]]) -> None:
    hit = sum(h for h, _ in rate.values())
    total = sum(t for _, t in rate.values())
    print(f"archive join: {hit}/{total} events matched "
          f"({hit / total:.0%})\n" if total else "no events\n")
    print(f"  {'type':<8}{'matched':>9}{'events':>8}{'rate':>7}")
    for cat, (h, t) in sorted(rate.items(), key=lambda p: -p[1][1]):
        flag = "  <- nothing joined" if h == 0 else ""
        print(f"  {cat:<8}{h:>9}{t:>8}{h / t:>7.0%}{flag}")


def score(events: list[dict], truth_key: str) -> None:
    usable = [e for e in events if e.get(truth_key) is not None and not e["cancelled"]]
    print(f"{len(events)} events, {len(usable)} with settled truth and not cancelled")
    if len(usable) < len(events):
        print(f"  ({len(events) - len(usable)} still in flight or cancelled — a run "
              "has to outlast its trains before they can be scored)")
    print()
    if not usable:
        return
    print(f"{'bucket':>7}{'n':>7}{'real lead':>11}{'DB MAE':>9}{'DB bias':>9}"
          f"{'surprise':>10}{'plan MAE':>10}{'explicit':>10}")
    for tau in sorted({e["tau"] for e in usable}, reverse=True):
        rows = [e for e in usable if e["tau"] == tau]
        truth = [e[truth_key] for e in rows]
        db_err = [e["db"] - t for e, t in zip(rows, truth)]
        surprise = sum(1 for e, t in zip(rows, truth)
                       if t > e["db"] + SURPRISE_MINUTES) / len(rows)
        explicit = sum(1 for e in rows if e["db_explicit"]) / len(rows)
        label = f"{tau // 60}h" if tau >= 60 else f"{tau}m"
        leads = sorted(e["lead"] for e in rows)
        real = f"{leads[0]:.0f}-{leads[-1]:.0f}m"
        print(f"{label:>7}{len(rows):>7}{real:>11}{st.mean(abs(x) for x in db_err):>9.1f}"
              f"{st.mean(db_err):>9.1f}{surprise:>10.0%}"
              f"{st.mean(abs(t) for t in truth):>10.1f}{explicit:>10.0%}")
    print("\n'plan MAE' is the always-on-time forecaster: the bar DB must clear.")
    print("'surprise' is P(actual more than "
          f"{SURPRISE_MINUTES} min later than predicted) — the missed connection.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["events", "score"])
    ap.add_argument("--day", required=True)
    ap.add_argument("--out", type=Path, default=cf.OUT)
    ap.add_argument("--truth", choices=["settled", "archive"], default="settled")
    ap.add_argument("--data-dir", type=Path, nargs="+",
                    default=[Path(__file__).resolve().parents[1] / "pipeline/data"])
    ap.add_argument("--stations", type=Path,
                    default=Path(__file__).parent / "forecast_stations.csv")
    args = ap.parse_args()

    day = dt.date.fromisoformat(args.day)
    stops, polls = read_day(args.out, day)
    events = build_events(stops, polls)
    print(f"{len(stops)} scheduled arrivals, {len(events)} events "
          f"over {len(polls)} stations", file=sys.stderr)
    if args.command == "events":
        for e in events[:10]:
            print(json.dumps(e, sort_keys=True))
        return
    if args.truth == "archive":
        truth = load_truth(args.data_dir, day, {s.eva for s in
                                                cf.load_stations(args.stations)})
        report_join(attach_truth(events, truth))
        print()
        score(events, "archive")
    else:
        score(events, "settled")


if __name__ == "__main__":
    main()
