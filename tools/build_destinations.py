"""Freeze the second tier of polled stations: where the connecting trains go.

The comparison's twenty stations were pre-registered before any data was
collected, and nothing here may widen that set — they are the origins, and
adding one after the fact would let the choice be made in hindsight. A journey
with a change, though, *ends* somewhere else, and scoring it end to end needs
DB's own forecast for the second leg's arrival. IRIS serves forecasts one
station at a time, so the far end has to be polled as well.

That second tier is derived rather than chosen. It is every terminus of a train
departing a pre-registered station, kept when the timetable sends at least
[MIN_DEPARTURES_PER_DAY] trains there — a floor that drops the long tail of
one-a-day workings without touching any regular service. The rule is a function
of the timetable, not of anything observed, so it cannot be tuned towards a
result: no delay, forecast or score is read here.

The output is committed. A timetable change would otherwise re-derive a
different set later and silently re-interpret data already collected, and the
file records the window it came from so that stays visible.

Names come from the plan documents' `ppth`, which is IRIS's own spelling, so
they are resolved through IRIS's own station lookup rather than matched against
a station list — an exact answer instead of a fuzzy one. Stops IRIS does not
serve a timetable for (`db="false"`, mostly replacement bus stops) are dropped:
there is no forecast to read there.

Usage:
    python tools/build_destinations.py --plan-dir tools/.forecasts/plan \
        --out tools/forecast_destinations.csv
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
IRIS = "https://iris.noncd.db.de/iris-tts/timetable"
UA = "BayesianBahn/0.1 (F-Droid; FOSS delay prediction; evaluation harness)"

# Below this a destination is a single daily working — a depot run, a seasonal
# extra — and polling it all day buys almost no journeys.
MIN_DEPARTURES_PER_DAY = 5
PAUSE_BETWEEN_LOOKUPS = 0.4


def plan_day(name: str) -> str | None:
    """The `yymmdd` a cached plan document covers, from `{eva}-{yymmdd}-{hh}`."""
    parts = name.removesuffix(".xml").split("-")
    if len(parts) != 3 or not parts[1].isdigit() or len(parts[1]) != 6:
        return None
    return parts[1]


def terminus_counts(plan_dir: Path) -> tuple[collections.Counter, set[str]]:
    """How many departures head for each terminus, and the days that covers.

    The last entry of a departure's `ppth` is where the train ends up, which is
    the far end of the second leg for every passenger who boards it here.
    """
    counts: collections.Counter = collections.Counter()
    days: set[str] = set()
    for path in sorted(plan_dir.glob("*.xml")):
        day = plan_day(path.name)
        if day is None:
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue  # a half-written cache entry; the collector rewrites it
        days.add(day)
        for stop in root.findall("s"):
            departure = stop.find("dp")
            if departure is None:
                continue
            path_stops = [s for s in (departure.get("ppth") or "").split("|") if s]
            if path_stops:
                counts[path_stops[-1]] += 1
    return counts, days


def frequent(counts: collections.Counter, days: int,
             minimum: int = MIN_DEPARTURES_PER_DAY) -> list[str]:
    """Termini served often enough to be worth a poll, in a stable order."""
    if days <= 0:
        return []
    kept = [name for name, n in counts.items() if n >= minimum * days]
    # Sorted by name, not by count: the file is a set, and ordering it by
    # frequency would make it re-shuffle every time it is rebuilt.
    return sorted(kept)


def query_for(name: str) -> str:
    """What to ask IRIS for. The lookup is a prefix search over station names.

    A `/` in the name cannot be sent: the service routes on the raw path and
    answers 404 even for `%2F`, which is how `Köln Messe/Deutz` went missing.
    Asking for the part before the slash finds it anyway, because the search is
    by prefix — and the caller still requires the full name back, so the
    shorter query cannot resolve to a different station.
    """
    return name.split("/")[0]


def lookup(name: str, fetch) -> tuple[str, str] | None:
    """Resolve an IRIS station name to `(eva, name)`, or None if it has no feed."""
    try:
        body = fetch(f"{IRIS}/station/{urllib.parse.quote(query_for(name), safe='')}")
        stations = ET.fromstring(body).findall("station")
    except Exception:
        return None
    for station in stations:
        eva, found = station.get("eva"), station.get("name")
        # `db="false"` marks a stop IRIS knows of but publishes no timetable
        # for. Exact-name only: the lookup is a prefix search, so a near miss
        # would otherwise quietly poll a different station for the whole study.
        if station.get("db") != "true" or found != name or not eva:
            continue
        if eva.isdigit():
            return eva, found
    return None


def http(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def render(rows: list[tuple[str, str, int]], days: list[str], minimum: int) -> str:
    span = f"{days[0]}..{days[-1]}" if days else "no plan documents"
    lines = [
        "# Second tier for the forecast comparison: the far end of a change.",
        "# Derived, not chosen — rebuild with tools/build_destinations.py.",
        "# Rule: terminus of a train departing a pre-registered station, with",
        f"#   at least {minimum} departures a day over {span} ({len(days)} days).",
        "# These stations are polled so DB's forecast for the second leg can be",
        "# read; they are never used as the origin of a scored connection.",
        "# eva;name;departures",
    ]
    lines += [f"{eva};{name};{n}" for eva, name, n in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-dir", type=Path, default=TOOLS / ".forecasts" / "plan")
    ap.add_argument("--out", type=Path, default=TOOLS / "forecast_destinations.csv")
    ap.add_argument("--min-per-day", type=int, default=MIN_DEPARTURES_PER_DAY)
    args = ap.parse_args()

    counts, days = terminus_counts(args.plan_dir)
    if not days:
        raise SystemExit(f"no cached plan documents under {args.plan_dir}")
    names = frequent(counts, len(days), args.min_per_day)
    print(f"{len(counts)} termini over {len(days)} days, "
          f"{len(names)} at >= {args.min_per_day}/day")

    rows, dropped = [], []
    for name in names:
        found = lookup(name, http)
        if found is None:
            dropped.append(name)
        else:
            rows.append((found[0], found[1], counts[name]))
        time.sleep(PAUSE_BETWEEN_LOOKUPS)

    # A station can appear under more than one spelling; one row per eva, so
    # the collector cannot end up polling the same feed twice.
    seen, unique = set(), []
    for eva, name, n in sorted(rows, key=lambda r: int(r[0])):
        if eva not in seen:
            seen.add(eva)
            unique.append((eva, name, n))

    args.out.write_text(render(unique, sorted(days), args.min_per_day),
                        encoding="utf-8")
    covered = sum(counts[name] for _, name, _ in unique)
    print(f"wrote {len(unique)} stations to {args.out} "
          f"({covered / max(sum(counts.values()), 1):.0%} of departures)")
    if dropped:
        print(f"no IRIS timetable for {len(dropped)}: {', '.join(dropped[:8])}"
              + (" ..." if len(dropped) > 8 else ""))


if __name__ == "__main__":
    main()
