"""Measure the journey search offline, on a day of archived timetable.

`tools/journey_bench.py` is limited by politeness to a keyless public API: one
attempt costs a transfer board, a board costs four plan documents, and every
query needs both a witness and a replay. The 44-journey set behind
MAX_TRANSFER_ATTEMPTS cost ~3400 requests, and a sample that small cannot tell
a 93% strategy from a 98% one — the difference is two journeys.

It is also the wrong ceiling to be stuck at, because the archive already
contains the timetable. Every stop carries `train_line_ride_id` and
`train_line_station_num`, so one day reconstructs into the whole national
timetable graph — for 2026-06-10, 43k train runs over 5235 stations, which
covers 97% of the transfer-eligible stations in the app's own station list.
Routing questions can therefore be answered locally, on immutable data, at any
sample size, with zero requests.

What this measures, and what it deliberately cannot:

  * It measures the *routing heuristic*: the attempt budget, the candidate
    ranking, the detour tolerance, the weight floor. Ground truth is
    exhaustive within the very time windows the app searches, so a miss is the
    heuristic's fault by construction — not a journey needing two changes.
  * It cannot measure name resolution. Here stations are EVA numbers and
    identity is exact, so the spelling problem RouteStationMatcher exists to
    solve is invisible. That is what journey_bench.py stays for.
  * The archive records what ran, not what was planned: cancelled stops are
    dropped, and a day of heavy disruption is a different timetable. Compare
    two days before believing anything marginal.

Usage:

    python tools/route_bench.py snapshot --data-dir pipeline/data --day 2026-06-10
    python tools/route_bench.py bench --day 2026-06-10 --queries 600
    python tools/route_bench.py sweep --day 2026-06-10 --queries 600
"""

from __future__ import annotations

import argparse
import bisect
import calendar
import datetime as dt
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = Path(__file__).parent / ".timetable"

# --- mirrored from JourneyPlanner.kt / ConnectionPlanner.kt -------------------
# Changing a value here measures a different app, not a better one; keep these
# in step with the Kotlin constants (pipeline/tests asserts that they are).
MAX_DIRECT = 3
ORIGIN_HOURS = 3
TRANSFER_HOURS = 4
MAX_TRANSFER_SCAN = 15
MAX_TRANSFER_RESULTS = 3
MAX_TRANSFER_ATTEMPTS = 8
TRANSFERS_PER_FEEDER = 2
MIN_TRANSFER_WEIGHT = 40
DETOUR_TOLERANCE = 1.25
TRANSFER_MINUTES = 5

# TrainClass.LONG_DISTANCE in DelayModel.kt. Everything else — including the
# private regional operators (HLB, NWB, ARV, AVG, ag, …), which are thousands
# of runs a day — is covered by the Deutschland-Ticket and therefore usable.
LONG_DISTANCE = {"ICE", "IC", "EC", "ECE", "RJ", "RJX", "NJ", "EN", "FLX", "TGV",
                 "D", "IR", "WB"}


def covers(category: str) -> bool:
    """Mirrors DeutschlandTicket.covers."""
    return category.upper() not in LONG_DISTANCE


@dataclass(frozen=True)
class Config:
    """The knobs under test. Defaults are what the app ships."""

    max_attempts: int = MAX_TRANSFER_ATTEMPTS
    per_feeder: int = TRANSFERS_PER_FEEDER
    min_weight: int = MIN_TRANSFER_WEIGHT
    detour: float = DETOUR_TOLERANCE
    scan: int = MAX_TRANSFER_SCAN
    # "distance" ships; "weight" is the pre-0.1.2 behaviour; "hybrid" keeps the
    # detour filter but orders what survives it by station size.
    ranking: str = "distance"


# --- the timetable ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stop:
    """One train's call at one station, with the rest of its route."""

    cat: str
    num: str
    arr: int | None  # epoch minutes
    dep: int | None
    path: tuple[tuple[str, int | None], ...]  # onward (eva, arrival)

    @property
    def ride(self) -> tuple[str, str]:
        return (self.cat, self.num)


def _hour_floor(minutes: int) -> int:
    return minutes - minutes % 60


def wall_minutes(when: dt.datetime) -> int:
    """Minutes since the epoch, reading a naive time as German wall clock.

    IRIS and the archive both publish local wall-clock times without an offset,
    and `snapshot` keeps them that way — it casts the naive timestamp straight
    to an integer. `datetime.timestamp()` would instead apply whatever timezone
    the machine is in, which silently shifted every query by the UTC offset:
    `--times 08:00` benchmarked 06:00 in summer. Comparisons between strategies
    survived that (they all shifted together), but the times were not the times.
    """
    return calendar.timegm(when.timetuple()) // 60


class Timetable:
    """A day of the archive, indexed the way IRIS's plan endpoint serves it.

    IRIS publishes one document per station-hour; the app fetches three of them
    for the origin and four for a transfer, then filters by exact time. Bucketing
    by hour here reproduces that horizon, which matters: a feeder arriving 4h01
    after it departs is invisible to the app, and must be invisible here too.
    """

    def __init__(self, rides: dict[int, tuple[str, str, tuple]]) -> None:
        self.rides = rides
        self.by_station: dict[str, list[tuple[int, int, int]]] = {}
        for ride, (_cat, _num, stops) in rides.items():
            for idx, (eva, arr, dep) in enumerate(stops):
                when = dep if dep is not None else arr
                if when is None:
                    continue
                self.by_station.setdefault(eva, []).append((when, ride, idx))
        for entries in self.by_station.values():
            entries.sort()
        self._keys = {eva: [e[0] for e in entries]
                      for eva, entries in self.by_station.items()}

    def board(self, eva: str, start: int, hours: int) -> list[Stop]:
        """Every call at `eva` in the hour buckets IRIS would have returned."""
        entries = self.by_station.get(eva)
        if not entries:
            return []
        lo = _hour_floor(start)
        hi = lo + hours * 60
        keys = self._keys[eva]
        out = []
        for i in range(bisect.bisect_left(keys, lo), bisect.bisect_left(keys, hi)):
            _when, ride, idx = entries[i]
            cat, num, stops = self.rides[ride]
            _eva, arr, dep = stops[idx]
            out.append(Stop(cat, num, arr, dep,
                            tuple((s[0], s[1]) for s in stops[idx + 1:])))
        return out


# --- stations ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Station:
    eva: str
    name: str
    weight: int
    lat: float
    lon: float


def stations() -> dict[str, Station]:
    out = {}
    text = (ROOT / "app/src/main/assets/stations.csv").read_text(encoding="utf-8")
    for line in text.splitlines():
        p = line.split(";")
        if len(p) < 5:
            continue
        out[p[0]] = Station(p[0], p[1], int(p[2]), float(p[3]), float(p[4]))
    return out


def km(a: Station, b: Station) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    x = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371 * math.asin(min(1.0, math.sqrt(x)))


# --- snapshot ----------------------------------------------------------------


def snapshot(data_dirs: list[Path], day: dt.date, out: Path) -> None:
    """Extract one day into a small parquet, so `bench` starts in seconds."""
    import polars as pl

    files = sorted(f for d in data_dirs for f in d.glob("data-*.parquet"))
    if not files:
        raise SystemExit(f"no data-*.parquet in {[str(d) for d in data_dirs]}")
    # The monthly archive stores ns and build_recent.py us; normalise as
    # build_boards.py does, or the concat of the two rejects the mix.
    unit = pl.Datetime("us")
    when = (pl.col("arrival_planned_time").cast(unit)
            .fill_null(pl.col("departure_planned_time").cast(unit)))
    frames = [
        pl.scan_parquet(f)
        .with_columns(arrival_planned_time=pl.col("arrival_planned_time").cast(unit),
                      departure_planned_time=pl.col("departure_planned_time").cast(unit))
        .filter(when.dt.date() == day)
        for f in files
    ]
    minutes = 60_000_000  # microseconds
    df = (pl.concat(frames)
          .filter(~pl.col("is_canceled"))
          .select(
              eva=pl.col("eva").str.strip_chars_start("0"),
              cat=pl.col("train_type"),
              num=pl.col("train_number"),
              ride=pl.col("train_line_ride_id"),
              seq=pl.col("train_line_station_num"),
              arr=(pl.col("arrival_planned_time").cast(pl.Int64) // minutes),
              dep=(pl.col("departure_planned_time").cast(pl.Int64) // minutes),
          )
          .unique(subset=["ride", "seq"])
          .collect())
    if df.is_empty():
        raise SystemExit(f"no stops for {day} — is it inside the archive's range?")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"{out}: {df.height} stops, {df['ride'].n_unique()} rides, "
          f"{df['eva'].n_unique()} stations")


def load(path: Path) -> Timetable:
    import polars as pl

    # Sorting by seq is what turns a bag of observed stops back into a route.
    grouped = (pl.read_parquet(path).sort("ride", "seq")
               .group_by("ride", maintain_order=True)
               .agg(pl.col("cat").first(), pl.col("num").first(),
                    pl.col("eva"), pl.col("arr"), pl.col("dep")))
    # A ride id is a long string; interning to an int keeps the index small.
    rides: dict[int, tuple[str, str, tuple]] = {}
    for key, row in enumerate(grouped.iter_rows(named=True)):
        rides[key] = (row["cat"] or "", row["num"] or "",
                      tuple(zip(row["eva"], row["arr"], row["dep"])))
    return Timetable(rides)


# --- ground truth ------------------------------------------------------------


def feeders(tt: Timetable, origin: str, depart: int) -> list[Stop]:
    """The origin's departures, exactly as JourneyPlanner filters them."""
    board = [s for s in tt.board(origin, depart, ORIGIN_HOURS)
             if s.dep is not None and s.dep >= depart and covers(s.cat)]
    board.sort(key=lambda s: s.dep or 0)
    return board


def reachable(tt: Timetable, origin: str, depart: int) -> tuple[set[str], dict[str, set[str]]]:
    """Exhaustive reachability within the app's own windows.

    Returns the directly reachable stations and, for every station reachable
    with exactly one change, the set of transfer stations that achieve it —
    the witnesses a heuristic could have found.
    """
    origin_deps = feeders(tt, origin, depart)
    direct: set[str] = set()
    for f in origin_deps:
        direct.update(eva for eva, _ in f.path)
    witnesses: dict[str, set[str]] = {}
    for f in origin_deps:
        assert f.dep is not None
        for via, arrival in f.path:
            if arrival is None or via == origin:
                continue
            ready = arrival + TRANSFER_MINUTES
            for onward in tt.board(via, f.dep, TRANSFER_HOURS):
                if onward.dep is None or onward.dep < ready or not covers(onward.cat):
                    continue
                if onward.ride == f.ride:
                    continue  # staying on the same train is not a change
                for eva, _ in onward.path:
                    if eva != origin and eva not in direct:
                        witnesses.setdefault(eva, set()).add(via)
    return direct, witnesses


# --- the heuristic under test -------------------------------------------------


def candidates(path: tuple, origin: Station, dest: Station, tried: set[str],
               by_eva: dict[str, Station], cfg: Config) -> list[Station]:
    """Mirrors JourneyPlanner.transferCandidates."""
    usable = []
    for eva, _arr in path:
        st = by_eva.get(eva)
        if st is None or st.eva == dest.eva or st.weight < cfg.min_weight:
            continue
        if st.name in tried:
            continue
        usable.append(st)
    if cfg.ranking == "weight":  # pre-0.1.2: no filter, biggest station first
        return sorted(usable, key=lambda s: -s.weight)
    goal = km(origin, dest) * cfg.detour
    near = [s for s in usable if km(s, dest) <= goal]
    if cfg.ranking == "hybrid":  # drop what leads away, then prefer junctions
        return sorted(near, key=lambda s: -s.weight)
    return sorted(near, key=lambda s: km(s, dest))


def search(tt: Timetable, origin: Station, dest: Station, depart: int,
           by_eva: dict[str, Station], cfg: Config, *, budget: int) -> dict:
    """Replays JourneyPlanner; reports the attempt index of the first hit.

    `budget` is passed separately from `cfg.max_attempts` so one generous run
    yields the whole budget curve: solved-at-b is first_hit <= b.
    """
    board = feeders(tt, origin.eva, depart)
    others = [f for f in board if not any(eva == dest.eva for eva, _ in f.path)]
    direct = len(board) - len(others)

    attempts, first_hit, tried = 0, None, set()
    for f in others[:cfg.scan]:
        if attempts >= budget:
            break
        assert f.dep is not None
        for c in candidates(f.path, origin, dest, tried, by_eva, cfg)[:cfg.per_feeder]:
            if attempts >= budget:
                break
            attempts += 1
            tried.add(c.name)
            at = tt.board(c.eva, f.dep, TRANSFER_HOURS)
            here = [s for s in at if s.ride == f.ride and s.arr is not None]
            if not here:
                continue  # the feeder does not reach it inside the 4h window
            ready = here[0].arr + TRANSFER_MINUTES
            if any(s.dep is not None and s.dep >= ready and covers(s.cat)
                   and s.ride != f.ride
                   and any(eva == dest.eva for eva, _ in s.path)
                   for s in at):
                if first_hit is None:
                    first_hit = attempts
                break  # transferItinerary returns on its first success
    return {"direct": direct, "first_hit": first_hit, "attempts": attempts}


# --- query sets ---------------------------------------------------------------


def build_queries(tt: Timetable, by_eva: dict[str, Station], *, count: int,
                  times: list[int], seed: int, per_origin: int,
                  min_origin_weight: int, band: tuple[float, float]) -> list[dict]:
    """Sample journeys that provably need exactly one change.

    Ground truth is the exhaustive scan, not a witness found by the same
    board-walk the search uses — the old generator could only ever propose
    journeys its own mechanism could see, which biases recall upwards.
    """
    rng = random.Random(seed)
    pool = [eva for eva, st in by_eva.items()
            if st.weight >= min_origin_weight and eva in tt.by_station]
    rng.shuffle(pool)
    lo, hi = band
    queries: list[dict] = []
    for origin_eva in pool:
        if len(queries) >= count:
            break
        origin = by_eva[origin_eva]
        depart = times[len(queries) % len(times)]
        direct, witnesses = reachable(tt, origin_eva, depart)
        options = [
            (eva, vias) for eva, vias in witnesses.items()
            if eva in by_eva and lo <= km(origin, by_eva[eva]) <= hi
        ]
        rng.shuffle(options)
        for eva, vias in options[:per_origin]:
            queries.append({"from": origin_eva, "to": eva, "depart": depart,
                            "vias": sorted(vias)})
    return queries[:count]


def diagnose(q: dict, by_eva: dict[str, Station], cfg: Config) -> str:
    """Why an exhaustively-solvable journey was missed."""
    origin, dest = by_eva[q["from"]], by_eva[q["to"]]
    vias = [by_eva[v] for v in q["vias"] if v in by_eva]
    if not vias:
        return "transfer not in station list"
    heavy = [v for v in vias if v.weight >= cfg.min_weight]
    if not heavy:
        return f"all transfers below weight {cfg.min_weight}"
    goal = km(origin, dest)
    near = [v for v in heavy if km(v, dest) <= goal * cfg.detour]
    if not near:
        return f"all transfers beyond detour {cfg.detour}"
    return "admissible but never ranked/reached"


# --- reporting ----------------------------------------------------------------


def bench(tt: Timetable, by_eva: dict[str, Station], queries: list[dict],
          cfg: Config, *, budget: int, verbose: bool) -> list[dict]:
    rows = []
    for i, q in enumerate(queries, 1):
        origin, dest = by_eva[q["from"]], by_eva[q["to"]]
        r = search(tt, origin, dest, q["depart"], by_eva, cfg, budget=budget)
        r["query"] = q
        if r["first_hit"] is None:
            r["why"] = diagnose(q, by_eva, cfg)
        rows.append(r)
        if verbose and (i % 50 == 0 or i == len(queries)):
            solved = sum(1 for x in rows if x["first_hit"] is not None)
            print(f"  [{i}/{len(queries)}] solved-so-far {solved / i:.0%}",
                  file=sys.stderr, flush=True)
    return rows


def report(rows: list[dict], budget: int) -> None:
    n = len(rows)
    print(f"\nn = {n} journeys, each with a one-change connection that provably")
    print("exists inside the windows the app itself searches\n")
    print("share solved, by attempt budget")
    marks = [b for b in range(1, budget + 1) if b <= 12 or b % 4 == 0]
    print("budget " + "".join(f"{b:>5}" for b in marks))
    print("       " + "".join(
        f"{sum(1 for r in rows if r['first_hit'] and r['first_hit'] <= b) / n:>5.0%}"
        for b in marks))

    missed = [r for r in rows if r["first_hit"] is None]
    if missed:
        print(f"\nwhy the {len(missed)} unsolved journeys fail")
        reasons: dict[str, int] = {}
        for r in missed:
            reasons[r["why"]] = reasons.get(r["why"], 0) + 1
        for why, c in sorted(reasons.items(), key=lambda p: -p[1]):
            print(f"  {c:4}  ({c / n:4.0%})  {why}")

    hits = [r["first_hit"] for r in rows if r["first_hit"]]
    if hits:
        hits.sort()
        q = [hits[min(len(hits) - 1, int(len(hits) * p))] for p in (.5, .9, .99)]
        print(f"\nattempts to first hit: median {q[0]}, p90 {q[1]}, p99 {q[2]}, "
              f"max {hits[-1]}")


def sweep(tt: Timetable, by_eva: dict[str, Station], queries: list[dict],
          budget: int) -> None:
    print(f"\nparameter sweep on n = {len(queries)} journeys "
          f"(share solved at the shipped budget of {MAX_TRANSFER_ATTEMPTS}, "
          "and at 2x)")
    variants = [
        ("shipped (distance)", Config()),
        ("weight ranking (pre-0.1.2)", replace(Config(), ranking="weight")),
        ("hybrid: detour then weight", replace(Config(), ranking="hybrid")),
        ("hybrid, detour 1.5", replace(Config(), ranking="hybrid", detour=1.5)),
        ("hybrid, 3 per feeder", replace(Config(), ranking="hybrid", per_feeder=3)),
        ("hybrid, weight floor 100", replace(Config(), ranking="hybrid", min_weight=100)),
        ("1 per feeder", replace(Config(), per_feeder=1)),
        ("3 per feeder", replace(Config(), per_feeder=3)),
        ("no detour filter", replace(Config(), detour=99.0)),
        ("detour 1.5", replace(Config(), detour=1.5)),
        ("detour 1.1", replace(Config(), detour=1.1)),
        ("no weight floor", replace(Config(), min_weight=0)),
        ("weight floor 100", replace(Config(), min_weight=100)),
        ("scan 25 feeders", replace(Config(), scan=25)),
    ]
    n = len(queries)
    print(f"{'variant':<28}{'@8':>6}{'@16':>6}{'mean att':>10}")
    for name, cfg in variants:
        rows = bench(tt, by_eva, queries, cfg, budget=budget, verbose=False)
        at8 = sum(1 for r in rows
                  if r["first_hit"] and r["first_hit"] <= MAX_TRANSFER_ATTEMPTS) / n
        at16 = sum(1 for r in rows
                   if r["first_hit"] and r["first_hit"] <= 2 * MAX_TRANSFER_ATTEMPTS) / n
        spent = sum(min(r["attempts"], MAX_TRANSFER_ATTEMPTS)
                    if r["first_hit"] is None
                    else min(r["first_hit"], MAX_TRANSFER_ATTEMPTS) for r in rows) / n
        print(f"{name:<28}{at8:>6.0%}{at16:>6.0%}{spent:>10.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "bench", "sweep"])
    ap.add_argument("--data-dir", type=Path, nargs="+", default=[ROOT / "pipeline/data"])
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--per-origin", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget", type=int, default=32,
                    help="attempts to allow while measuring (>= the shipped cap)")
    ap.add_argument("--times", default="08:00,12:00,16:00,19:00")
    ap.add_argument("--min-origin-weight", type=int, default=MIN_TRANSFER_WEIGHT)
    ap.add_argument("--band", default="25,100", help="origin-destination km range")
    ap.add_argument("--save-queries", type=Path)
    args = ap.parse_args()

    day = dt.date.fromisoformat(args.day)
    snap = SNAPSHOTS / f"{day}.parquet"
    if args.command == "snapshot":
        snapshot(args.data_dir, day, snap)
        return
    if not snap.exists():
        raise SystemExit(f"no snapshot for {day}; run `snapshot --day {day}` first")

    tt = load(snap)
    by_eva = stations()
    times = [wall_minutes(dt.datetime.combine(day, dt.time(int(t[:2]), int(t[3:]))))
             for t in args.times.split(",")]
    lo, hi = (float(x) for x in args.band.split(","))

    print(f"{day}: {len(tt.rides)} rides, {len(tt.by_station)} stations", file=sys.stderr)
    queries = build_queries(tt, by_eva, count=args.queries, times=times,
                            seed=args.seed, per_origin=args.per_origin,
                            min_origin_weight=args.min_origin_weight, band=(lo, hi))
    print(f"sampled {len(queries)} solvable journeys", file=sys.stderr)
    if args.save_queries:
        args.save_queries.write_text(json.dumps(queries, indent=1), encoding="utf-8")

    if args.command == "bench":
        rows = bench(tt, by_eva, queries, Config(), budget=args.budget, verbose=True)
        report(rows, args.budget)
    else:
        sweep(tt, by_eva, queries, args.budget)


if __name__ == "__main__":
    main()
