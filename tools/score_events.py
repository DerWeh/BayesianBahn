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

# Minutes needed to change trains: JourneyPlanner.plan's default, and what the
# connection screen starts at before the user adjusts it.
TRANSFER_MINUTES = 5


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
    planned: int | None               # wall-clock minutes, planned arrival
    planned_dep: int | None = None    # planned departure from the same stop
    # Stations this train calls at after here, in order, as IRIS spells them.
    # The last is where it ends up, which is the far end of a two-leg journey
    # for anyone changing onto it.
    ppth: list[str] = field(default_factory=list)
    obs: list[tuple[int, int | None, int | None, bool]] = field(default_factory=list)
    # (seen at, DB reporting trouble on the route). Its own stream rather than a
    # fifth column on `obs`: a disruption is not a property of the times, and
    # keeping it apart means every reader of `obs` stays as it was.
    notices: list[tuple[float, bool]] = field(default_factory=list)

    def forecast_at(self, when: float) -> int | None:
        """DB's predicted delay in minutes as of `when` (epoch seconds).

        None means we cannot tell — no observation and no way to know whether
        DB had said anything by then. The caller decides using the poll log.
        """
        latest = None
        for at, arrival, _departure, _cancelled in self.obs:
            if at <= when:
                latest = arrival
            else:
                break
        if latest is None:
            return None
        return latest - self.planned

    def departure_forecast_at(self, when: float) -> int | None:
        """DB's predicted departure delay — the other half of a connection."""
        if self.planned_dep is None:
            return None
        latest = None
        for at, _arrival, departure, _cancelled in self.obs:
            if at <= when:
                latest = departure
            else:
                break
        return None if latest is None else latest - self.planned_dep

    def cancelled_at(self, when: float) -> bool:
        state = False
        for at, _arrival, _departure, cancelled in self.obs:
            if at <= when:
                state = cancelled
        return state

    def disrupted_at(self, when: float) -> bool:
        """Whether DB was reporting trouble on this train's route as of `when`.

        Separate from `cancelled_at` because it is a separate thing: a blocked
        section leaves every stop with its timetable intact, so a forecast built
        from those times can be confident and wrong at once.
        """
        state = False
        for at, disrupted in self.notices:
            if at <= when:
                state = disrupted
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

    def trip_start(self) -> int | None:
        """When this train set off, from the trip id (`...-YYMMDDHHMM-nn`).

        A passenger cannot have boarded before it, so a prediction made earlier
        is unambiguously a *before you commit* prediction — which is the only
        kind that can change a decision.
        """
        parts = self.trip.split("-")
        for part in reversed(parts):
            if len(part) == 10 and part.isdigit():
                return cf.iris_time(part)
        return None


def read_day(out: Path, day: dt.date, tiers: tuple[int, ...] = (1,),
             cohorts: tuple[int, ...] = (1,),
             ) -> tuple[dict[tuple[str, str], Stop], dict[str, list[float]]]:
    """Stops keyed by (station, trip), plus successful poll times per station.

    Cohort 1, tier 1 by default — the twenty stations registered before any data
    existed, which is the sample the headline speaks for.

    Both filters exist for the same reason. The second tier is polled solely to
    supply the far end of a change, so letting those stations originate an
    arrival or a connection would widen a registered set with stations picked
    after the data was in. And the cohorts were sampled on different axes and
    started on different days, so pooling them would produce a number that
    describes neither.
    """
    records, torn = cf.Journal.read(out / f"forecasts-{day}.jsonl")
    if torn:
        print(f"note: skipped {torn} torn line(s)", file=sys.stderr)
    roles = cf.roles_of(records)
    wanted = {eva for eva, (tier, cohort) in roles.items()
              if tier in tiers and cohort in cohorts}
    stops: dict[tuple[str, str], Stop] = {}
    polls: dict[str, list[float]] = {}
    for r in records:
        if r.get("eva") not in wanted:
            continue
        if r["t"] == "poll":
            if r.get("ok"):
                polls.setdefault(r["eva"], []).append(float(r["at"]))
        elif r["t"] == "plan":
            if r.get("par") is None and r.get("pdp") is None:
                continue
            stops[(r["eva"], r["id"])] = Stop(
                eva=r["eva"], trip=r["id"], cat=r["cat"], num=r["num"],
                line=r.get("line"), planned=r.get("par"), planned_dep=r.get("pdp"),
                ppth=list(r.get("ppth") or ()),
            )
    for r in records:
        if r["t"] != "obs":
            continue
        stop = stops.get((r["eva"], r["id"]))
        if stop is None:
            continue  # observed but never in the plan window; no planned time
        # "present, no ct" is not missing data: it means DB left the time alone.
        arrival = r["ar"] if r["ar"] is not None else stop.planned
        departure = r["dp"] if r["dp"] is not None else stop.planned_dep
        stop.obs.append((float(r["at"]), arrival, departure, bool(r.get("arc"))))
        if "msg" in r:
            stop.notices.append((float(r["at"]), cf.is_disruption(r["msg"])))
    for stop in stops.values():
        stop.obs.sort()
        stop.notices.sort()
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
        if stop.planned is None:
            continue  # departure-only: kept for connections, not an arrival event
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
                "tau": tau, "lead": round(lead, 1), "read_at": read_at,
                "planned_dep": stop.planned_dep,
                "db": db if db is not None else 0,
                "db_explicit": db is not None,
                "cancelled": stop.cancelled_at(when),
                # DB reporting the route in trouble while leaving the times
                # alone. Recorded, not acted on: it is a different failure from
                # a cancellation and deserves its own count before it is used.
                "disrupted": stop.disrupted_at(when),
                "settled": settled,
            })
    return events


# The four anchors a lead time can be measured to. Planned times are what a
# passenger reads off a timetable; real ones are when the train actually moved.
# Binning by a planned time conditions on something correlated with the outcome:
# inside a "five minutes before arrival" bucket, the late trains had far more
# real warning than the punctual ones, which flatters every forecaster at once.
# Departure matters more than arrival for a decision — once aboard, the choice
# is already made.
ANCHORS = ("planned_arrival", "actual_arrival", "planned_departure",
           "actual_departure")

BUCKET_EDGES = (0, 10, 20, 45, 90, 180, 10 ** 9)
BUCKET_LABELS = ("<10m", "10-20m", "20-45m", "45-90m", "1.5-3h", ">3h")


def anchor_minutes(event: dict, anchor: str) -> float | None:
    """The wall-clock minute the lead time is measured back from."""
    if anchor == "planned_arrival":
        return event.get("planned")
    if anchor == "actual_arrival":
        truth = event.get("archive")
        return None if truth is None else event["planned"] + truth
    if anchor == "planned_departure":
        return event.get("planned_dep")
    if anchor == "actual_departure":
        dep = event.get("planned_dep")
        delay = event.get("archive_dep")
        return None if dep is None or delay is None else dep + delay
    raise ValueError(anchor)


def bucket_of(minutes: float) -> str | None:
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= minutes < BUCKET_EDGES[i + 1]:
            return BUCKET_LABELS[i]
    return None


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
    # Coalesced, so a stop where the train starts — no arrival, only a
    # departure — survives. Those are the connecting trains.
    when = pl.coalesce(planned, pl.col("departure_planned_time").cast(unit))
    frames = [
        pl.scan_parquet(f)
        .with_columns(arrival_planned_time=planned, arrival_change_time=changed)
        .filter(when.dt.date() == day)
        .with_columns(eva0=pl.col("eva").str.strip_chars_start("0"))
        .filter(pl.col("eva0").is_in(list(evas)))
        .select(
            eva=pl.col("eva0"),
            cat=pl.col("train_type"),
            num=pl.col("train_number"),
            # Wall clock as if UTC, the convention collect_forecasts.iris_time
            # stores journal times in; anything else shifts the join by an hour.
            planned=(planned.cast(pl.Int64) // 60_000_000),
            planned_dep=(pl.col("departure_planned_time").cast(unit).cast(pl.Int64)
                         // 60_000_000),
            delay=((changed.cast(pl.Int64) - planned.cast(pl.Int64)) // 60_000_000),
            dep_delay=((pl.col("departure_change_time").cast(unit).cast(pl.Int64)
                        - pl.col("departure_planned_time").cast(unit).cast(pl.Int64))
                       // 60_000_000),
            cancelled=pl.col("is_canceled"),
        )
        for f in files
    ]
    rows = pl.concat(frames).unique(
        subset=["eva", "cat", "num", "planned", "planned_dep"]).collect()
    truth: dict = {}
    for r in rows.iter_rows(named=True):
        value = {"delay": 0 if r["delay"] is None else int(r["delay"]),
                 "dep_delay": 0 if r["dep_delay"] is None else int(r["dep_delay"]),
                 "cancelled": bool(r["cancelled"])}
        # Indexed by whichever planned time the caller holds: an arriving feeder
        # is found by its arrival, a connecting train by its departure.
        if r["planned"] is not None:
            truth[(r["eva"], r["cat"], r["num"], r["planned"])] = value
        if r["planned_dep"] is not None:
            truth[("dep", r["eva"], r["cat"], r["num"], r["planned_dep"])] = value
    return truth


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
            e["archive_dep"] = found["dep_delay"]
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



def run_key(trip: str) -> str:
    """The trip id without its stop number: `{run}-{yymmddHHMM}`.

    IRIS numbers each stop of a run separately, so the same train carries a
    different id at the transfer and at its destination. The run itself is what
    identifies it across stations — and the run segment can be negative, so the
    stop number has to be taken off the end rather than the run read off the
    front.
    """
    return trip.rsplit("-", 1)[0]


def build_connections(stops: dict, polls: dict, truth: dict, *,
                      min_slack: int = 2, max_slack: int = 30) -> list[dict]:
    """One-change journeys, scored from before the passenger boards.

    A connection is a feeder arriving at a station and another train leaving it
    with between [min_slack] and [max_slack] minutes to spare. Below that band
    nobody would plan the change; above it, the connection is caught whatever
    happens and the question is uninteresting.

    The prediction is taken from before the feeder set off, because that is the
    only moment the answer can change a decision — once aboard, the passenger
    has committed. DB answers this question with a yes or a no; the whole point
    of a distribution is that the honest answer is often "probably".
    """
    by_station: dict[str, list[Stop]] = {}
    for stop in stops.values():
        by_station.setdefault(stop.eva, []).append(stop)

    out = []
    for eva, at_station in by_station.items():
        station_polls = polls.get(eva, [])
        feeders = [s for s in at_station if s.planned is not None]
        onward = [s for s in at_station if s.planned_dep is not None]
        for feeder in feeders:
            start = feeder.trip_start()
            if start is None:
                continue
            read_at = last_poll_before(station_polls, wall_to_epoch(start))
            if read_at is None:
                continue      # we were not yet collecting when it set off
            feeder_truth = truth.get((eva, feeder.cat, feeder.num, feeder.planned))
            if feeder_truth is None or feeder_truth["cancelled"]:
                continue
            db_arrival = feeder.forecast_at(read_at) or 0
            for conn in onward:
                if conn.trip == feeder.trip:
                    continue  # staying on the same train is not a change
                slack = conn.planned_dep - feeder.planned - TRANSFER_MINUTES
                if not min_slack <= slack <= max_slack:
                    continue
                conn_truth = truth.get(
                    ("dep", eva, conn.cat, conn.num, conn.planned_dep))
                if conn_truth is None or conn_truth["cancelled"]:
                    continue
                db_departure = conn.departure_forecast_at(read_at) or 0
                # The feeder's arrival delay at which the change stops working,
                # using DB's own forecast for the connecting train so both
                # forecasters see the same information.
                threshold = (conn.planned_dep + db_departure - TRANSFER_MINUTES
                             - feeder.planned)
                caught = (conn.planned_dep + conn_truth["dep_delay"]
                          >= feeder.planned + feeder_truth["delay"] + TRANSFER_MINUTES)
                out.append({
                    "eva": eva, "trip": feeder.trip, "cat": feeder.cat,
                    "num": feeder.num, "line": feeder.line,
                    "planned": feeder.planned, "planned_dep": feeder.planned_dep,
                    "read_at": read_at,
                    "lead": round((wall_to_epoch(start) - read_at) / 60, 1),
                    "tau": 0, "slack": slack,
                    "conn": f"{conn.cat} {conn.num}",
                    "db": db_arrival,
                    "threshold": threshold,
                    "db_catch": db_arrival <= threshold,
                    "caught": caught,
                    "archive": feeder_truth["delay"],
                    "archive_dep": feeder_truth["dep_delay"],
                    "cancelled": False,
                })
    return out


# The app's own window: trains leaving up to half an hour before the feeder is
# due are candidates too, because a delayed one is sometimes the connection that
# works. ConnectionPlanner.MAX_CANDIDATES caps how many the model is given.
#
# It caps what the *forecasters* are asked about, and nothing else. Truth walks
# the whole day: a passenger who misses all six waits for the seventh, and the
# app's answer is then wrong by however long that took. Scoring only the
# journeys that fit inside the six discards exactly the ones the app got most
# wrong — on 2026-08-25 that was 30% of them, and 99% of those had a full list
# spanning a median of 22 minutes with the feeder a median of one minute late,
# so it was the window running out, not the trains.
CANDIDATE_WINDOW_BEFORE = 30
MAX_CANDIDATES = 6
# ConnectionPlanner.MAX_ALREADY_GONE: how many of the six may be trains that
# have already left. Without this the six were sometimes all in the past.
MAX_ALREADY_GONE = 2


def pick_candidates(resolved: list[dict], feeder_planned: int) -> list[dict]:
    """The app's own list, mirroring ConnectionPlanner.pickCandidates.

    Room goes to the trains ahead first; what is left over goes to the most
    recently missed ones, newest last. Kept in step with the Kotlin by
    test_the_python_and_kotlin_candidate_rules_agree.
    """
    gone = [c for c in resolved if c["planned_dep"] < feeder_planned]
    ahead = [c for c in resolved if c["planned_dep"] >= feeder_planned]
    reserved = min(MAX_ALREADY_GONE, len(gone))
    taken = ahead[:MAX_CANDIDATES - reserved]
    keep = MAX_CANDIDATES - len(taken)
    return (gone[-keep:] if keep else []) + taken


def build_journeys(stops: dict, far: dict, polls: dict, far_polls: dict,
                   truth: dict, destinations: dict[str, str], *,
                   min_slack: int = 2, max_slack: int = 30) -> list[dict]:
    """Two-leg journeys, scored end to end against the arrival that happened.

    The connection scorer asks whether the change worked. This asks the question
    a passenger actually has: *when do I get there* — the same question the
    direct-journey scorer asks, in the same units, so for the first time the two
    kinds of journey can be compared with each other rather than only each with
    DB.

    A journey is a feeder arriving at a registered station and a destination
    reachable from it, judged from before the feeder set off. Every train from
    the transfer towards that destination is a candidate, not just the one that
    happens to fit the timetable: missing the planned connection and taking the
    next train is an outcome with an arrival time, not a failure to predict.

    Both forecasters answer over the same candidates and from the same moment.
    Ours answers with a distribution over the final arrival; DB's answer is the
    arrival of whichever train *its* forecasts say the passenger catches. The
    truth is the arrival of whichever train they actually caught.

    `far` holds the same trains' stops at the destination — collected from the
    second tier, which exists for exactly this — keyed by (eva, run), and
    `far_polls` says when those stations answered. Both are needed, because a
    stop DB lists no change for is DB saying *on time*, not DB saying nothing:
    that is one of the four shapes, and reading it as silence would discard most
    of DB's answers and leave the comparison drawn from the trains DB flagged.
    Only a station we were not yet polling is genuinely unknown, and that is
    what the poll log distinguishes.
    """
    by_station: dict[str, list[Stop]] = {}
    for stop in stops.values():
        by_station.setdefault(stop.eva, []).append(stop)

    out = []
    for eva, at_station in by_station.items():
        station_polls = polls.get(eva, [])
        feeders = [s for s in at_station if s.planned is not None]
        onward = [s for s in at_station if s.planned_dep is not None and s.ppth]
        for feeder in feeders:
            start = feeder.trip_start()
            if start is None:
                continue
            read_at = last_poll_before(station_polls, wall_to_epoch(start))
            if read_at is None:
                continue      # we were not yet collecting when it set off
            feeder_truth = truth.get((eva, feeder.cat, feeder.num, feeder.planned))
            if feeder_truth is None or feeder_truth["cancelled"]:
                continue

            # Destinations worth asking about: the terminus of a train that
            # forms a change nobody would have to invent, and that the second
            # tier covers so DB's answer for the far end can be read at all.
            wanted: set[str] = set()
            for conn in onward:
                if conn.trip == feeder.trip:
                    continue
                slack = conn.planned_dep - feeder.planned - TRANSFER_MINUTES
                if min_slack <= slack <= max_slack and conn.ppth[-1] in destinations:
                    wanted.add(conn.ppth[-1])

            for name in sorted(wanted):
                dest_eva = destinations[name]
                resolved = []
                for conn in sorted(onward, key=lambda s: s.planned_dep):
                    if conn.trip == feeder.trip or name not in conn.ppth:
                        continue
                    if conn.planned_dep < feeder.planned - CANDIDATE_WINDOW_BEFORE:
                        continue
                    arrival = far.get((dest_eva, run_key(conn.trip)))
                    if arrival is None or arrival.planned is None:
                        continue    # nothing collected at the far end for it
                    if not polled_by(far_polls.get(dest_eva, []), read_at):
                        continue    # we were not yet watching the far end
                    arrival_truth = truth.get(
                        (dest_eva, conn.cat, conn.num, arrival.planned))
                    departure_truth = truth.get(
                        ("dep", eva, conn.cat, conn.num, conn.planned_dep))
                    resolved.append({
                        "id": conn.trip, "cat": conn.cat, "num": conn.num,
                        "line": conn.line,
                        "planned_dep": conn.planned_dep,
                        "planned_arr": arrival.planned,
                        # `or 0`, not `is None`: both stations had answered by
                        # now, so no change listed means on time.
                        "live_dep": conn.departure_forecast_at(read_at) or 0,
                        "cancelled_live": conn.cancelled_at(read_at),
                        "db_arr": arrival.forecast_at(read_at) or 0,
                        "truth_dep": (None if departure_truth is None
                                      else departure_truth["dep_delay"]),
                        "truth_arr": (None if arrival_truth is None
                                      else arrival_truth["delay"]),
                        # None, not True: a row the archive never joined is
                        # unknown, and calling it cancelled would quietly skip
                        # past the train the passenger may actually have taken.
                        "cancelled": (None if departure_truth is None
                                      else departure_truth["cancelled"]),
                    })
                if not resolved:
                    continue
                # The forecasters answer over the app's own list; truth may walk
                # past its end.
                candidates = pick_candidates(resolved, feeder.planned)
                out.append({
                    "eva": eva, "trip": feeder.trip, "cat": feeder.cat,
                    "num": feeder.num, "line": feeder.line,
                    "planned": feeder.planned, "planned_dep": feeder.planned_dep,
                    "read_at": read_at,
                    "lead": round((wall_to_epoch(start) - read_at) / 60, 1),
                    "tau": 0,
                    "dest_eva": dest_eva, "dest": name,
                    "db": feeder.forecast_at(read_at) or 0,
                    "archive": feeder_truth["delay"],
                    "archive_dep": feeder_truth["dep_delay"],
                    "cancelled": False,
                    "candidates": candidates,
                    "offered": len(resolved),
                    **boarded(feeder.planned, feeder_truth["delay"],
                              feeder.forecast_at(read_at) or 0, candidates,
                              resolved),
                })
    return out


def boarded(planned_arrival: int, feeder_delay: int, feeder_db: int,
            candidates: list[dict], every: list[dict] | None = None) -> dict:
    """Which train the passenger caught, and which one DB said they would.

    Pure bookkeeping over times that are already fixed — no model — so it lives
    here rather than in the harness. The rule is the model's own: board the
    first candidate that has not left by the time you reach the platform.

    The two answers differ only in which clock they read. Truth uses the delays
    that happened; DB uses the delays it was predicting at the moment we read
    it, for the feeder and for every candidate alike, so neither forecaster is
    given a fact the other was denied.

    Both come back as wall-clock minutes at the destination, absolute rather
    than relative, because the harness rebases them onto whatever reference the
    model ends up using. `None` means that forecaster names no train at all:
    for DB a journey it does not think is possible, for truth a passenger left
    behind by every train that ran that day. Those are dropped rather than
    scored, and counted, because a rule that drops the hard cases flatters
    everyone.

    `candidates` is what the app would have shown and is what both forecasters
    answer over. `every` is the whole day's onward list, and only truth reads
    it: a passenger who misses all six waits for the seventh, and a forecast
    that never mentioned the seventh is wrong by however long the wait was.
    Capping truth at six instead excused exactly those journeys.
    """
    def db_choice(ready: int) -> dict | None:
        """DB's own forecasts resolve completely: it names a time for every
        candidate, because a stop it lists no change for is a stop it says runs
        to plan."""
        for candidate in candidates:
            if candidate["cancelled_live"]:
                continue
            if candidate["planned_dep"] + candidate["live_dep"] >= ready:
                return candidate
        return None

    def actually_caught(ready: int) -> tuple[dict | None, int]:
        """The archive does not always join. A candidate we cannot resolve stops
        the walk rather than being stepped over: stepping over it would hand the
        passenger a later train than they may have taken, and the error would
        land in the tail, which is the part being measured."""
        for rank, candidate in enumerate(every or candidates):
            if candidate["cancelled"] is None or candidate["truth_dep"] is None:
                return None, rank
            if candidate["cancelled"]:
                continue
            if candidate["planned_dep"] + candidate["truth_dep"] >= ready:
                return (candidate if candidate["truth_arr"] is not None
                        else None), rank
        return None, len(every or candidates)

    caught, rank = actually_caught(
        planned_arrival + feeder_delay + TRANSFER_MINUTES)
    predicted = db_choice(planned_arrival + feeder_db + TRANSFER_MINUTES)
    return {
        "truth_arrival": (None if caught is None
                          else caught["planned_arr"] + caught["truth_arr"]),
        "caught_id": None if caught is None else caught["id"],
        # Where in the day's list the passenger's train sat, and whether that
        # was past the end of the list either forecaster was shown. Derived
        # here because only here are both lists in hand.
        "caught_rank": rank,
        "beyond_list": rank >= len(candidates),
        "db_arrival": (None if predicted is None
                       else predicted["planned_arr"] + predicted["db_arr"]),
        "db_id": None if predicted is None else predicted["id"],
    }


def report_catch(scored: Path) -> None:
    """Did the passenger make the change, and who said so.

    DB answers yes or no. A yes-or-no answer is a probability of 0 or 1, so the
    Brier score compares the two directly: for DB it is just the share it got
    wrong, for us it is the mean squared distance from what happened. A
    forecaster that says "70%" and is right 70% of the time beats one that says
    "yes" and is right 85% of the time only if the confident one is confidently
    wrong often enough — which is exactly what is worth knowing before trusting
    a connection.
    """
    rows = [json.loads(line) for line in scored.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"no scored connections in {scored}")
    caught = [r for r in rows if r["caught"]]
    missed = [r for r in rows if not r["caught"]]
    print(f"n = {len(rows)} connections from before the feeder set off; "
          f"{len(caught)} caught, {len(missed)} missed\n")

    def brier(group, key):
        return st.mean((r[key] - (1.0 if r["caught"] else 0.0)) ** 2 for r in group)

    print(f"{'slack':>10}{'n':>6}{'caught':>8}{'DB right':>10}{'DB Brier':>10}"
          f"{'our Brier':>11}{'our P(catch)':>14}")
    bands = ((2, 6), (6, 11), (11, 21), (21, 31))
    for lo, hi in bands:
        g = [r for r in rows if lo <= r["slack"] < hi]
        if not g:
            continue
        db_right = sum(1 for r in g if (r["db_catch_p"] == 1) == r["caught"]) / len(g)
        share = sum(1 for r in g if r["caught"]) / len(g)
        print(f"{f'{lo}-{hi - 1}m':>10}{len(g):>6}{share:>8.0%}{db_right:>10.0%}"
              f"{brier(g, 'db_catch_p'):>10.3f}{brier(g, 'p_catch'):>11.3f}"
              f"{st.mean(r['p_catch'] for r in g):>14.2f}")
    print(f"{'all':>10}{len(rows):>6}"
          f"{sum(1 for r in rows if r['caught']) / len(rows):>8.0%}"
          f"{sum(1 for r in rows if (r['db_catch_p'] == 1) == r['caught']) / len(rows):>10.0%}"
          f"{brier(rows, 'db_catch_p'):>10.3f}{brier(rows, 'p_catch'):>11.3f}"
          f"{st.mean(r['p_catch'] for r in rows):>14.2f}")

    print("\nwhere the two disagree, split by what actually happened:")
    for label, group in (("caught", caught), ("missed", missed)):
        if not group:
            continue
        db_right = sum(1 for r in group if (r["db_catch_p"] == 1) == r["caught"]) / len(group)
        print(f"  {label:<8} n={len(group):>5}  DB right {db_right:>4.0%}  "
              f"our mean P(catch) {st.mean(r['p_catch'] for r in group):.2f}  "
              f"Brier {brier(group, 'p_catch'):.3f} vs DB "
              f"{brier(group, 'db_catch_p'):.3f}")
    print("\nBrier = mean squared error of the probability against the 0/1 outcome;")
    print("lower is better, and a yes/no answer scores its own error rate.")


def compare(scored: Path, anchor: str = "planned_arrival") -> None:
    """Our model against DB's number, on the events the JVM harness scored.

    Both forecasters see the same information state at each lead time, so every
    row is paired by construction.

    CRPS does have a DB counterpart, which is easy to miss: a point forecast is
    a degenerate distribution, and its CRPS is exactly its absolute error. So
    the DB MAE column doubles as DB's CRPS, and ours against it is a proper
    scoring comparison that credits the whole distribution rather than only the
    median. Coverage is ours alone — DB publishes no interval to cover with.
    """
    rows = [json.loads(line) for line in scored.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"no scored events in {scored}")
    sources: dict[str, int] = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    print(f"n = {len(rows)} scored events; forecast source {sources}")
    print(f"lead time measured back from the {anchor.replace('_', ' ')}\n")
    binned: dict[str, list[dict]] = {}
    for r in rows:
        at = anchor_minutes(r, anchor)
        if at is None:
            continue
        # `read_at` is the poll that produced the reading, in epoch seconds.
        label = bucket_of((wall_to_epoch(at) - r["read_at"]) / 60)
        if label:
            binned.setdefault(label, []).append(r)
    print(f"{'bucket':>8}{'n':>6}{'DB MAE':>8}{'ours':>7}{'DB bias':>9}{'ours':>7}"
          f"{'DB surp':>9}{'ours':>7}{'CRPS':>7}{'cover80':>9}")
    for label in BUCKET_LABELS:
        g = binned.get(label)
        if not g:
            continue
        db = [r["db"] - r["truth"] for r in g]
        us = [r["q50"] - r["truth"] for r in g]
        db_surprise = sum(1 for r in g
                          if r["truth"] > r["db"] + SURPRISE_MINUTES) / len(g)
        our_surprise = sum(1 for r in g
                           if r["truth"] > r["q50"] + SURPRISE_MINUTES) / len(g)
        covered = sum(1 for r in g if r["q10"] <= r["truth"] <= r["q90"]) / len(g)
        print(f"{label:>8}{len(g):>6}{st.mean(abs(x) for x in db):>8.2f}"
              f"{st.mean(abs(x) for x in us):>7.2f}{st.mean(db):>9.2f}"
              f"{st.mean(us):>7.2f}{db_surprise:>9.0%}{our_surprise:>7.0%}"
              f"{st.mean(r['crps'] for r in g):>7.2f}{covered:>9.0%}")
    print(f"\nsurp = P(actual more than {SURPRISE_MINUTES} min later than predicted).")
    print("cover80 = share of truths inside q10..q90; nominal is 80%, so a lower")
    print("number means the distribution is too confident.")
    print()
    print("CRPS(F, y) = integral over x of (F(x) - 1{x >= y})^2 dx, in minutes.")
    print("For a point forecast F is a step at m and this reduces to |m - y|, so")
    print("the DB MAE column *is* DB's CRPS: comparing it against ours is a")
    print("like-for-like comparison under a proper scoring rule, and the only")
    print("one here that gives credit for the distribution rather than just the")
    print("median.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",
                    choices=["events", "score", "compare", "connections",
                             "journeys", "catch"])
    ap.add_argument("--day", default="1970-01-01")
    ap.add_argument("--out", type=Path, default=cf.OUT)
    ap.add_argument("--truth", choices=["settled", "archive"], default="settled")
    ap.add_argument("--cohort", type=int, default=1,
                    help="which registered group of origins to score")
    ap.add_argument("--data-dir", type=Path, nargs="+",
                    default=[Path(__file__).resolve().parents[1] / "pipeline/data"])
    ap.add_argument("--stations", type=Path,
                    default=Path(__file__).parent / "forecast_stations.csv")
    ap.add_argument("--scored", type=Path, help="JVM harness output, for `compare`")
    ap.add_argument("--anchor", choices=[*ANCHORS, "all"], default="planned_arrival",
                    help="what the lead time is measured back from")
    ap.add_argument("--events-out", type=Path,
                    help="write every event as JSONL here, for the JVM harness")
    ap.add_argument("--destinations", type=Path,
                    default=Path(cf.__file__).parent / "forecast_destinations.csv",
                    help="the far ends this cohort can be scored to")
    args = ap.parse_args()

    if args.command == "catch":
        if not args.scored:
            raise SystemExit("catch needs --scored, the JVM harness output")
        report_catch(args.scored)
        return

    if args.command == "compare":
        if not args.scored:
            raise SystemExit("compare needs --scored, the JVM harness output")
        for anchor in (ANCHORS if args.anchor == "all" else [args.anchor]):
            compare(args.scored, anchor)
            print()
        return

    day = dt.date.fromisoformat(args.day)
    stops, polls = read_day(args.out, day, cohorts=(args.cohort,))
    if args.command == "journeys":
        # The far end is read from the second tier, which is polled for exactly
        # this and never originates anything, and truth has to cover both.
        far_stops, far_polls = read_day(args.out, day, tiers=(2,),
                                        cohorts=(args.cohort,))
        far = {(stop.eva, run_key(stop.trip)): stop for stop in far_stops.values()}
        destinations = {s.name: s.eva
                        for s in cf.load_stations(args.destinations, 2)}
        truth = load_truth(args.data_dir, day,
                           {s.eva for s in cf.load_stations(args.stations)}
                           | set(destinations.values()))
        trips = build_journeys(stops, far, polls, far_polls, truth,
                               destinations)
        scoreable = [t for t in trips
                     if t["truth_arrival"] is not None and t["db_arrival"] is not None]
        no_truth = sum(1 for t in trips if t["truth_arrival"] is None)
        no_db = sum(1 for t in trips if t["db_arrival"] is None)
        # The journeys the old cap silently discarded: the passenger boarded a
        # train past the end of the list both forecasters were shown, so both
        # are wrong and both are now charged for it.
        beyond = sum(1 for t in scoreable
                     if t["caught_rank"] >= len(t["candidates"]))
        print(f"{len(trips)} journeys, {len(scoreable)} with both answers "
              f"({no_truth} nobody caught anything, {no_db} DB names no train, "
              f"{beyond} boarded past the app's own list)", file=sys.stderr)
        if args.events_out:
            with args.events_out.open("w", encoding="utf-8") as fh:
                for t in scoreable:
                    fh.write(json.dumps(t, sort_keys=True) + "\n")
            print(f"wrote {len(scoreable)} to {args.events_out}", file=sys.stderr)
        return
    if args.command == "connections":
        truth = load_truth(args.data_dir, day,
                           {s.eva for s in cf.load_stations(args.stations)})
        links = build_connections(stops, polls, truth)
        caught = sum(1 for c in links if c["caught"])
        print(f"{len(links)} connections, {caught} caught "
              f"({caught / max(1, len(links)):.0%}), "
              f"{len(links) - caught} missed", file=sys.stderr)
        if args.events_out:
            with args.events_out.open("w", encoding="utf-8") as fh:
                for c in links:
                    fh.write(json.dumps(c, sort_keys=True) + "\n")
            print(f"wrote {len(links)} to {args.events_out}", file=sys.stderr)
        return
    events = build_events(stops, polls)
    print(f"{len(stops)} scheduled arrivals, {len(events)} events "
          f"over {len(polls)} stations", file=sys.stderr)
    if args.command == "events":
        if args.truth == "archive":
            truth = load_truth(args.data_dir, day,
                               {s.eva for s in cf.load_stations(args.stations)})
            report_join(attach_truth(events, truth))
        if not args.events_out:
            for e in events[:10]:
                print(json.dumps(e, sort_keys=True))
            return
        with args.events_out.open("w", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e, sort_keys=True) + "\n")
        print(f"wrote {len(events)} events to {args.events_out}", file=sys.stderr)
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
