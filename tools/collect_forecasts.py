"""Record DB's forecast for a train as it changes, so it can be scored later.

The archive tells us when a train *actually* arrived, but not what DB predicted
beforehand: it stores one row per stop, the final state. Comparing our
prediction against DB's therefore needs the forecast history captured live —
this is the only part of the evaluation that cannot be recomputed from data we
already have, which is why it runs for days and why losing a run hurts.

Two tiers of station are polled. Tier 1 is the twenty pre-registered stations
the comparison is built on; nothing may be added to it after the fact. Tier 2
is where the trains leaving those stations end up — `forecast_destinations.csv`,
derived from the timetable by `build_destinations.py`. A one-change journey
ends at the far end of the second leg, and IRIS serves forecasts one station at
a time, so DB's answer for that arrival can only be had by polling there too. A
tier-2 station is never the *origin* of a scored arrival or connection; the tier
is written into every poll record so that stays a fact about the data rather
than about whichever CSV is on disk when it is read back.

What it records, per station, every [CADENCE_MINUTES]:

  * `plan`   the planned time and route of each train, once per train and stop.
             Plan documents for a past hour never change, so they are cached on
             disk and fetched once.
  * `obs`    DB's changed time, but only when it *differs* from the last value
             seen — the trajectory is the deltas, and most stops do not move
             between polls. Reconstructing "what DB said at time T" is then the
             last delta at or before T.
  * `poll`   every attempt, successful or not. Without this a network outage is
             indistinguishable from DB not changing its mind, which would turn
             an outage into evidence.
  * `hafas`  a small sample cross-checked against the Navigator's own backend,
             to confirm `fchg` says what the app people actually use says.

`fchg` reports every change DB currently knows about, which includes stops that
have *already happened* — on the first round of a fresh run about 70% of them.
Those carry no trajectory (we never saw them while they were still ahead), so
they are not prediction targets and analysis must keep only stops first seen
while still in the future. They cost one baseline round of disk and nothing
after that, since a finished stop never changes again; filtering here instead
would throw away the tail of the trajectory, which is what answers when DB's
number stops moving.

`fchg` also reports trips further out than the plan horizon, so some `obs`
records have no matching `plan` yet. That is recoverable rather than lost: the
trip id embeds the trip's start (`...-2608171718-14`), so the analysis can fetch
the missing plan documents afterwards — they are immutable once written.

Recovery. The journal is append-only JSONL, flushed and fsynced per record, one
line per record: a kill at any point can lose at most the final line, and the
reader skips a torn line rather than failing. Restart replays the day's journal
to rebuild the last-seen values, so an interrupted run resumes without
duplicating or re-reporting unchanged values. The schedule is a pure function of
the wall clock, so nothing needs to be remembered across restarts; a missed slot
is a gap in `poll`, which analysis can see.

Usage:
    python tools/collect_forecasts.py run                  # until stopped
    python tools/collect_forecasts.py run --minutes 30     # bounded
    python tools/collect_forecasts.py status
"""

from __future__ import annotations

import argparse
import calendar
import collections
import datetime as dt
import json
import os
import random
import signal
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / ".forecasts"
# The registered sets, in the order they were registered: a cohort is a group
# of origins plus the far ends its trains reach. Cohorts are never pooled —
# they start on different days and were sampled on different axes, cohort 1 by
# station size and cohort 2 by the kind of line — so the journal records which
# one a poll belongs to and the scorer takes one at a time.
COHORTS: tuple[tuple[int, str, str], ...] = (
    (1, "forecast_stations.csv", "forecast_destinations.csv"),
    (2, "forecast_stations_cohort2.csv", "forecast_destinations_cohort2.csv"),
)
IRIS = "https://iris.noncd.db.de/iris-tts/timetable"
HAFAS = "https://v6.db.transport.rest"
UA = "BayesianBahn/0.1 (F-Droid; FOSS delay prediction; evaluation harness)"

CADENCE_MINUTES = 10
# Polling exactly on the ten-minute grid would sample DB at a fixed phase of
# whatever cycle it regenerates `fchg` on, so any staleness in what we read
# would be constant rather than averaging out — and scheduled arrivals cluster
# on clock-friendly minutes, which is the same grid. A uniform offset inside
# the slot de-correlates both, fully for a one- or five-minute cycle. It also
# spreads the lead times we end up holding, which widens the axis the
# comparison is plotted against. The slot bookkeeping is unaffected: the offset
# is smaller than a slot, so a poll still lands in its own bucket.
JITTER_SECONDS = 300
# Trains are polled while they are still ahead of us; four hours of plan covers
# everything that can still be running when the next poll comes round.
PLAN_HORIZON_HOURS = 4
REQUEST_TIMEOUT = 20
PAUSE_BETWEEN_REQUESTS = 0.4
# What a station actually costs a round, pause included: measured at 0.58 s
# across 24 two-tier rounds on 2026-08-24, rounded up for headroom. This and
# the two ceilings below are the budget the station set has to fit inside, and
# they are what decides how far the comparison may be widened. Growing the set
# is otherwise a change with no visible limit until slots start being missed —
# and a missed slot is collection lost for good.
SECONDS_PER_STATION = 0.7
# A round may occupy at most this much of its slot. Overrunning drops the next
# one, and the comparison rests on holding a reading close to the moment scored.
MAX_SLOT_FRACTION = 0.5
# Sustained requests a second, averaged over the slot, counting the hourly plan
# fetch each station also needs. Self-imposed: IRIS publishes no limit, and the
# archive we take ground truth from polls about three times this hard.
MAX_REQUESTS_PER_SECOND = 1.0
# The community HAFAS proxy is rate-limited and often down — as of 2026-08-17 it
# answers 503 on every data endpoint while its root still returns 200. It is a
# cross-check, not a data source, so it is sampled rarely, its failure is not an
# error, and repeated failure backs off instead of hammering a service that is
# already struggling. It recovers on its own the first time a sample succeeds.
HAFAS_EVERY_MINUTES = 60
HAFAS_SAMPLE = 3
HAFAS_MAX_BACKOFF = 8


def iris_time(text: str | None) -> int | None:
    """IRIS `YYMMDDHHMM` to epoch minutes, reading it as German wall clock.

    Same convention as route_bench.wall_minutes: the archive and IRIS both
    publish local time without an offset, and applying the machine's timezone
    would shift everything by the UTC offset.
    """
    if not text or len(text) != 10 or not text.isdigit():
        return None
    when = dt.datetime.strptime(text, "%y%m%d%H%M")
    return calendar.timegm(when.timetuple()) // 60


def parse_plan(xml: str) -> dict[str, dict]:
    """Trip id -> planned times and route, from a `plan` document."""
    out: dict[str, dict] = {}
    for s in ET.fromstring(xml).findall("s"):
        trip = s.get("id")
        if not trip:
            continue
        tl = s.find("tl")
        ar, dp = s.find("ar"), s.find("dp")
        out[trip] = {
            "cat": (tl.get("c") if tl is not None else "") or "",
            "num": (tl.get("n") if tl is not None else "") or "",
            "line": (tl.get("l") if tl is not None else None),
            "par": iris_time(ar.get("pt")) if ar is not None else None,
            "pdp": iris_time(dp.get("pt")) if dp is not None else None,
            # The whole onward path, not just the next stop: its last entry is
            # where the train ends up, which is the far end of the second leg
            # for anyone who changes onto it here.
            "ppth": [p for p in ((dp.get("ppth") or "") if dp is not None
                                 else "").split("|") if p],
        }
    return out


def parse_changes(xml: str) -> dict[str, dict]:
    """Trip id -> DB's current forecast, from an `fchg` document.

    `ct` is the changed time and `cs == "c"` the cancellation, exactly as
    IrisParser.parseChanges reads them.
    """
    out: dict[str, dict] = {}
    for s in ET.fromstring(xml).findall("s"):
        trip = s.get("id")
        if not trip:
            continue
        ar, dp = s.find("ar"), s.find("dp")
        out[trip] = {
            "ar": iris_time(ar.get("ct")) if ar is not None else None,
            "dp": iris_time(dp.get("ct")) if dp is not None else None,
            "arc": (ar.get("cs") == "c") if ar is not None else False,
            "dpc": (dp.get("cs") == "c") if dp is not None else False,
        }
    return out


class Journal:
    """Append-only JSONL that survives being killed mid-write."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def append(self, record: dict) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        # Durability is the point of this class: a power cut must not lose the
        # last ten minutes of a multi-day run.
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def read(path: Path) -> tuple[list[dict], int]:
        """Records and the number of torn lines skipped.

        A process killed mid-write leaves a partial final line. That is expected,
        not corruption: skip it and keep everything before it.
        """
        if not path.exists():
            return [], 0
        records, torn = [], 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
        return records, torn


def roles_of(records: list[dict]) -> dict[str, tuple[int, int]]:
    """Each station's (tier, cohort) as it was polled, from the journal itself.

    Not from the station CSVs: those can be edited, and a later edit must not be
    able to re-label data already collected. A journal written before either
    field existed carries neither, and every station in it was a cohort-1
    origin — hence the defaults.
    """
    out: dict[str, tuple[int, int]] = {}
    for record in records:
        if record["t"] == "poll":
            out.setdefault(record["eva"],
                           (int(record.get("tier", 1)), int(record.get("cohort", 1))))
    return out


def tiers_of(records: list[dict]) -> dict[str, int]:
    """Each station's tier alone, for callers that do not care about cohorts."""
    return {eva: tier for eva, (tier, _) in roles_of(records).items()}


def slot_start(now: float, cadence: int = CADENCE_MINUTES) -> float:
    """The next poll time, aligned to the clock so restarts land in step."""
    minutes = now / 60
    return (int(minutes // cadence) + 1) * cadence * 60


@dataclass
class Station:
    eva: str
    name: str
    # 1 = registered origin, 2 = the far end of a change. Only tier 1 may start
    # a scored connection; tier 2 exists so DB's forecast for the second leg's
    # arrival can be read, and widening tier 1 after the fact would let the
    # station set be chosen in hindsight. The journal records the tier per poll
    # so this stays a property of the data, not of the current CSVs.
    tier: int = 1
    # Which registered group this station belongs to. Same reasoning: a station
    # cannot be moved between cohorts after its data exists.
    cohort: int = 1


def load_stations(path: Path, tier: int = 1, cohort: int = 1) -> list[Station]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        eva, name = line.split(";")[:2]
        out.append(Station(eva.strip(), name.strip(), tier, cohort))
    return out


def station_set(directory: Path | None = None,
                cohorts: tuple[tuple[int, str, str], ...] = COHORTS) -> list[Station]:
    """Every station polled: each cohort's origins, then the far ends it reaches.

    One entry per station, first registration winning. A station can appear in
    more than one file — three of the first twenty are also termini of their own
    trains — and polling it twice a round would double its requests for nothing.
    Order decides the collision, which is why the cohorts are listed in the order
    they were registered and origins come before far ends: the narrower role and
    the earlier cohort are the honest readings, since data already collected
    under them cannot be re-labelled by a later file.
    """
    directory = directory or Path(__file__).parent
    out: list[Station] = []
    seen: set[str] = set()
    for cohort, origins, destinations in cohorts:
        for path, tier in ((directory / origins, 1), (directory / destinations, 2)):
            if not path.exists():
                continue
            for station in load_stations(path, tier, cohort):
                if station.eva not in seen:
                    seen.add(station.eva)
                    out.append(station)
    return out


class Collector:
    def __init__(self, stations: list[Station], out: Path = OUT, *,
                 fetch=None, now=time.time) -> None:
        self.stations = stations
        self.out = out
        self.plan_dir = out / "plan"
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self._fetch = fetch or self._http
        self._now = now
        self.journal: Journal | None = None
        self.journal_day: dt.date | None = None
        # (eva, trip) -> last forecast we wrote, so unchanged values are not
        # re-appended. Rebuilt from the journal on restart.
        self.last: dict[tuple[str, str], dict] = {}
        self.planned: set[tuple[str, str]] = set()
        self.hafas_failures = 0
        self.stopping = False

    # --- io ---------------------------------------------------------------

    def _http(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.read().decode("utf-8", errors="replace")

    def _today(self) -> dt.date:
        return dt.datetime.fromtimestamp(self._now()).date()

    def _journal_for(self, day: dt.date) -> Journal:
        if self.journal_day != day:
            if self.journal:
                self.journal.close()
            self.journal = Journal(self.out / f"forecasts-{day}.jsonl")
            self.journal_day = day
            self.restore(day)
        assert self.journal is not None
        return self.journal

    def restore(self, day: dt.date) -> int:
        """Rebuild in-memory state from the day's journal after a restart."""
        records, torn = Journal.read(self.out / f"forecasts-{day}.jsonl")
        for record in records:
            if record.get("t") == "obs":
                self.last[(record["eva"], record["id"])] = {
                    k: record.get(k) for k in ("ar", "dp", "arc", "dpc")
                }
            elif record.get("t") == "plan":
                self.planned.add((record["eva"], record["id"]))
        return torn

    # --- collection --------------------------------------------------------

    def ensure_plan(self, eva: str, when: dt.datetime) -> dict[str, dict]:
        """Plan documents never change once written, so fetch each hour once."""
        key = f"{eva}-{when:%y%m%d}-{when:%H}.xml"
        cached = self.plan_dir / key
        if cached.exists():
            text = cached.read_text(encoding="utf-8")
        else:
            try:
                text = self._fetch(f"{IRIS}/plan/{eva}/{when:%y%m%d}/{when:%H}")
            except Exception:
                return {}
            # Written via a temp file so a crash cannot leave a half document
            # that later looks like a complete cache entry.
            tmp = cached.with_suffix(".part")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(cached)
        try:
            return parse_plan(text)
        except ET.ParseError:
            return {}

    def poll_station(self, station: Station) -> None:
        now = self._now()
        journal = self._journal_for(dt.datetime.fromtimestamp(now).date())
        start = dt.datetime.fromtimestamp(now)

        plans: dict[str, dict] = {}
        for hour in range(PLAN_HORIZON_HOURS):
            plans.update(self.ensure_plan(station.eva, start + dt.timedelta(hours=hour)))

        try:
            xml = self._fetch(f"{IRIS}/fchg/{station.eva}")
            changes = parse_changes(xml)
        except Exception as error:  # network, HTTP, malformed XML
            journal.append({"t": "poll", "at": int(now), "eva": station.eva,
                            "tier": station.tier, "cohort": station.cohort,
                            "ok": False, "err": type(error).__name__})
            return

        journal.append({"t": "poll", "at": int(now), "eva": station.eva,
                        "tier": station.tier, "cohort": station.cohort,
                        "ok": True, "stops": len(changes)})

        for trip, change in changes.items():
            key = (station.eva, trip)
            if key not in self.planned and trip in plans:
                journal.append({"t": "plan", "at": int(now), "eva": station.eva,
                                "id": trip, **plans[trip]})
                self.planned.add(key)
            if self.last.get(key) == change:
                continue  # DB has not moved; the trajectory records only moves
            journal.append({"t": "obs", "at": int(now), "eva": station.eva,
                            "id": trip, **change})
            self.last[key] = change

    def cross_check_hafas(self, rng: random.Random) -> None:
        """Confirm `fchg` agrees with what the Navigator's own backend says.

        Sampled, not systematic: the proxy is rate-limited and frequently
        unreachable, and a failure here says nothing about the data we keep.
        """
        now = self._now()
        journal = self._journal_for(dt.datetime.fromtimestamp(now).date())
        succeeded = False
        # Tier 1 only: this checks the feed the comparison is built on, and
        # spending the proxy's small budget on the far ends would dilute it.
        origins = [s for s in self.stations if s.tier == 1] or self.stations
        for station in rng.sample(origins, min(HAFAS_SAMPLE, len(origins))):
            try:
                body = self._fetch(
                    f"{HAFAS}/stops/{station.eva}/departures?duration=60&results=10",
                )
                payload = json.loads(body)
            except Exception as error:
                journal.append({"t": "hafas", "at": int(now), "eva": station.eva,
                                "ok": False, "err": type(error).__name__})
                continue
            rows = []
            for row in (payload.get("departures") or [])[:10]:
                line = (row.get("line") or {})
                rows.append({
                    "name": line.get("name"), "when": row.get("when"),
                    "planned": row.get("plannedWhen"), "delay": row.get("delay"),
                })
            journal.append({"t": "hafas", "at": int(now), "eva": station.eva,
                            "ok": True, "rows": rows})
            succeeded = True
            time.sleep(PAUSE_BETWEEN_REQUESTS)
        self.hafas_failures = 0 if succeeded else self.hafas_failures + 1

    def hafas_interval(self) -> float:
        """Seconds until the next cross-check, doubling while the proxy is down."""
        return HAFAS_EVERY_MINUTES * 60 * min(HAFAS_MAX_BACKOFF,
                                              2 ** self.hafas_failures)

    # --- the loop ----------------------------------------------------------

    def run(self, minutes: int | None = None, *, sleep=time.sleep,
            rng: random.Random | None = None) -> None:
        rng = rng or random.Random()
        deadline = self._now() + minutes * 60 if minutes else None
        last_hafas = 0.0
        while not self.stopping:
            target = slot_start(self._now()) + rng.uniform(0, JITTER_SECONDS)
            if deadline and target > deadline:
                break
            while not self.stopping and self._now() < target:
                sleep(min(5.0, target - self._now()))
            if self.stopping:
                break
            for station in self.stations:
                if self.stopping:
                    break
                self.poll_station(station)
                sleep(PAUSE_BETWEEN_REQUESTS)
            if self._now() - last_hafas >= self.hafas_interval():
                self.cross_check_hafas(rng)
                last_hafas = self._now()
        if self.journal:
            self.journal.close()


def health(records: list[dict], expected_stations: int, cadence: int = CADENCE_MINUTES
           ) -> dict:
    """Whether a run is actually collecting, not merely alive.

    A process can sit there polling nothing — wrong station file, DNS gone, a
    slot silently skipped every time — and `ps` cannot tell the difference. What
    distinguishes a healthy run is that every scheduled slot produced a round,
    every station answered, and the last round was recent.
    """
    polls = [r for r in records if r["t"] == "poll"]
    slots = sorted({r["at"] // (cadence * 60) for r in polls})
    stations = {r["eva"] for r in polls}
    return {
        "rounds": len(slots),
        # Slots between the first and the last that produced no poll at all: a
        # crash, a suspend, or a machine that was asleep.
        "missed_slots": (slots[-1] - slots[0] + 1 - len(slots)) if slots else 0,
        "polls": len(polls),
        "failed": sum(1 for r in polls if not r.get("ok")),
        "stations_seen": len(stations),
        "stations_expected": expected_stations,
        "stops": sum(r.get("stops", 0) for r in polls if r.get("ok")),
        "last_at": max((r["at"] for r in records if "at" in r), default=None),
        "first_at": min((r["at"] for r in records if "at" in r), default=None),
    }


def status(out: Path, directory: Path | None = None, now=time.time) -> None:
    days = sorted(out.glob("forecasts-*.jsonl"))
    if not days:
        print(f"nothing collected in {out}")
        return
    try:
        configured = station_set(directory)
    except OSError:
        configured = []
    for path in days:
        records, torn = Journal.read(path)
        kinds: dict[str, int] = {}
        for record in records:
            kinds[record["t"]] = kinds.get(record["t"], 0) + 1
        # Against the roles that day was collected under, not against today's
        # station files: the days before the second tier existed are complete
        # with twenty stations, and judging them by 281 would report every one
        # of them as broken.
        seen = set(roles_of(records).values()) or {(1, 1)}
        expected = len([s for s in configured if (s.tier, s.cohort) in seen])
        h = health(records, expected)
        span = ""
        if h["first_at"]:
            span = ("  " + dt.datetime.fromtimestamp(h["first_at"]).strftime("%H:%M")
                    + "-" + dt.datetime.fromtimestamp(h["last_at"]).strftime("%H:%M"))
        print(f"{path.name}{span}  "
              + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
              + (f"  torn-lines={torn}" if torn else ""))
        warn = []
        if h["failed"]:
            warn.append(f"{h['failed']} failed polls")
        if h["missed_slots"]:
            warn.append(f"{h['missed_slots']} missed slots")
        if expected and h["stations_seen"] < expected:
            warn.append(f"only {h['stations_seen']}/{expected} stations")
        print(f"  {h['rounds']} rounds, {h['stops']} stops seen"
              + (("  ** " + ", ".join(warn)) if warn else "  (clean)"))
        age = (now() - h["last_at"]) / 60 if h["last_at"] else None
        if age is not None and path == days[-1]:
            state = "alive" if age < 2 * CADENCE_MINUTES else "STALLED?"
            print(f"  last record {age:.1f} min ago  [{state}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "status"])
    ap.add_argument("--stations-dir", type=Path, default=Path(__file__).parent,
                    help="where the cohorts' station files live")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--minutes", type=int, help="stop after roughly this long")
    args = ap.parse_args()

    if args.command == "status":
        status(args.out, args.stations_dir)
        return

    collector = Collector(station_set(args.stations_dir), args.out)
    # A shutdown must not tear a record: finish the station in flight, then go.
    def stop(_signum, _frame):
        collector.stopping = True
        print("stopping after the current station", file=sys.stderr)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    torn = collector.restore(collector._today())
    roles = collections.Counter((s.tier, s.cohort) for s in collector.stations)
    shape = ", ".join(f"cohort {c} tier {t}: {n}"
                      for (t, c), n in sorted(roles.items(), key=lambda kv: kv[0][::-1]))
    print(f"{len(collector.stations)} stations ({shape}), "
          f"every {CADENCE_MINUTES} min -> {args.out}"
          + (f" (resumed; {torn} torn lines skipped)" if torn else ""), file=sys.stderr)
    collector.run(minutes=args.minutes)


if __name__ == "__main__":
    main()
