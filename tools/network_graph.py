"""The German railway network as the timetable describes it, for one day.

The comparison's first twenty stations were sampled by how busy the *station*
is. That misses what actually drives delay, which is a property of the line the
train is on: a single-track branch propagates a late train into the opposite
direction because the two have to pass somewhere, and a line shared with
long-distance services inherits their disruptions. A sample stratified by
station size can contain twenty stations and one kind of railway.

Neither track count nor freight is in the feed — IRIS publishes passenger
timetables, and says nothing about the infrastructure underneath. What it does
publish is every train's route, and that is enough for two observable proxies:

  * how many trains a day use the busiest *segment* the station sits on. A
    single-track section between passing loops cannot carry many; capacity is
    what the number is bounded by, so it separates branch from main line.
  * whether long-distance services use it at all, from `tl/@f == "F"` — IRIS's
    own traffic class, which beats curating a list of the 84 category codes in
    use. Lines that carry ICE and EC are the lines built for mixed traffic, so
    this is also the closest observable proxy for freight sharing the tracks.

A segment is a scheduled hop between consecutive stops. Every stop of every
train appears in its own station's plan document, so reading the whole day's
documents once and taking (this station, next stop) from each departure counts
every traversal exactly once.

Used by build_cohort.py to stratify, and reusable for anything else that needs
to know what kind of railway a station is on.
"""

from __future__ import annotations

import collections
import glob
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# IRIS's own traffic class. F is long distance, which includes the
# international services; S, N and D are the three flavours of regional.
LONG_DISTANCE = "F"


@dataclass
class Station:
    eva: str
    name: str
    calls: int = 0
    long_distance: int = 0
    neighbours: set[str] = field(default_factory=set)
    # Trains a day on the busiest segment this station sits on.
    peak_segment: int = 0

    @property
    def long_share(self) -> float:
        return self.long_distance / self.calls if self.calls else 0.0

    @property
    def degree(self) -> int:
        """Distinct adjacent stations: a through halt has two, a junction more."""
        return len(self.neighbours)


# `.../plan/{eva}/{yymmdd}/{hh}` — the eva is in the middle, not at the end.
# Taking the last segment yields the *hour*, which collapses five thousand
# stations onto twenty-four keys and is silent about it.
PLAN_EVA = re.compile(r"/plan/0*(\d+)/\d{6}/\d{2}$")


def plan_documents(raw_dir: Path):
    """Every `plan` response in an archive day, as (eva, xml)."""
    import polars as pl

    for path in sorted(glob.glob(str(raw_dir / "*.parquet"))):
        frame = pl.read_parquet(path, columns=["url", "api_name", "response_data"])
        frame = frame.filter(pl.col("api_name") == "timetables/v1/plan")
        for url, xml in zip(frame["url"], frame["response_data"]):
            found = PLAN_EVA.search(url or "")
            if xml and found:
                yield found.group(1), xml


def read_stops(eva: str, xml: str):
    """(station name, traffic class, previous stop, next stop) per stop."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return
    here = root.get("station")
    if not here:
        return
    for stop in root.findall("s"):
        line = stop.find("tl")
        arrival, departure = stop.find("ar"), stop.find("dp")

        def path(element):
            if element is None:
                return []
            return [s for s in (element.get("ppth") or "").split("|") if s]

        incoming, outgoing = path(arrival), path(departure)
        yield (here,
               line.get("f") if line is not None else None,
               incoming[-1] if incoming else None,
               outgoing[0] if outgoing else None)


def build(raw_dir: Path) -> tuple[dict[str, Station], dict[frozenset, int]]:
    """Stations by eva, and trains a day per undirected segment.

    Undirected because capacity is shared: on a single-track line the two
    directions are the same resource, and counting them apart would make such a
    line look like two quiet ones.
    """
    stations: dict[str, Station] = {}
    by_name: dict[str, Station] = {}
    directed: collections.Counter = collections.Counter()

    for eva, xml in plan_documents(raw_dir):
        for here, traffic, previous, following in read_stops(eva, xml):
            station = stations.get(eva)
            if station is None:
                station = stations[eva] = Station(eva, here)
                by_name.setdefault(here, station)
            station.calls += 1
            if traffic == LONG_DISTANCE:
                station.long_distance += 1
            for other in (previous, following):
                if other:
                    station.neighbours.add(other)
            if following:
                directed[(here, following)] += 1

    segments: collections.Counter = collections.Counter()
    for (a, b), n in directed.items():
        segments[frozenset((a, b))] += n
    for pair, n in segments.items():
        for name in pair:
            station = by_name.get(name)
            if station is not None and n > station.peak_segment:
                station.peak_segment = n
    return stations, dict(segments)
