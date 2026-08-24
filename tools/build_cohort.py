"""Register a second cohort of origins, stratified by the kind of line they sit on.

The first twenty stations were sampled by how busy the *station* is. Seven days
in, that turns out to leave the comparison narrow in a way station size cannot
show: three high-frequency urban S-Bahn stops supply about half of all scored
predictions, and delay behaviour is a property of the line, not of the platform.
A single-track branch propagates a late train into the opposite direction
because the two have to pass somewhere; a line shared with long-distance
services inherits their disruptions and their priority.

So this samples on two axes the timetable can actually see (see network_graph
for why these and not track count):

  * trains a day on the busiest segment the station sits on, cut into thirds —
    a capacity signature, because a single-track section cannot carry many;
  * whether long-distance services use the station at all.

Six cells, sampled equally. Equal allocation and not proportional: the point is
to contrast the kinds of railway, and the rare ones would otherwise not appear
at all — 5% of stations see a long-distance train. **Cohort 2 is therefore not
a representative sample of the network.** It answers "does the result hold
across kinds of line", never "what is the network-wide number"; cohort 1 stays
the sample that speaks for itself, and the two are never pooled.

No two picks are within [MIN_HOPS] segments of each other, so forty-two picks
are forty-two pieces of railway rather than one line sampled forty-two times.

Registered before the first day of collection under it, from an archive day
that was published before the choice was made, and committed — the same
discipline as the first twenty. Nothing about delay, forecast or score is read
here: the inputs are the timetable and nothing else.

Usage:
    python tools/build_cohort.py --raw-dir tools/.scored/2026-08-18/raw \
        --exclude tools/forecast_stations.csv tools/forecast_destinations.csv \
        --out tools/forecast_stations_cohort2.csv
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

import network_graph as ng

TOOLS = Path(__file__).resolve().parent

# The day this cohort was registered, used as the sampling seed so the draw can
# be repeated exactly. Chosen before the draw was looked at.
SEED = 20260824
# Below this a station sees too little traffic for a day to say anything.
MIN_CALLS = 20
# Segments between two picks. One would only stop them being neighbours; two
# also rules out the pair either side of a single intermediate stop.
MIN_HOPS = 2
PER_CELL = 7


def features(stations: dict[str, ng.Station], minimum: int = MIN_CALLS
             ) -> list[ng.Station]:
    """The stations a day can say something about, in a fixed order."""
    return sorted((s for s in stations.values()
                   if s.calls >= minimum and s.peak_segment > 0),
                  key=lambda s: int(s.eva))


def bands(eligible: list[ng.Station]) -> tuple[float, float]:
    """The two cuts that split peak-segment traffic into thirds.

    Taken from the eligible pool rather than from round numbers, so the three
    cells stay comparable in size whatever the timetable looks like that year.
    """
    import numpy as np

    peaks = np.array([s.peak_segment for s in eligible])
    return tuple(float(x) for x in np.percentile(peaks, [100 / 3, 200 / 3]))


def cell_of(station: ng.Station, cuts: tuple[float, float]) -> tuple[str, bool]:
    low, high = cuts
    band = "low" if station.peak_segment <= low else (
        "mid" if station.peak_segment <= high else "high")
    return band, station.long_distance > 0


def within_hops(start: str, stations_by_name: dict[str, ng.Station], hops: int
                ) -> set[str]:
    """Station names reachable from `start` in at most `hops` segments."""
    seen, frontier = {start}, {start}
    for _ in range(hops):
        nxt: set[str] = set()
        for name in frontier:
            station = stations_by_name.get(name)
            if station is not None:
                nxt |= station.neighbours
        frontier = nxt - seen
        seen |= nxt
    return seen


def draw(eligible: list[ng.Station], cuts: tuple[float, float], *,
         per_cell: int = PER_CELL, seed: int = SEED, hops: int = MIN_HOPS
         ) -> list[tuple[ng.Station, tuple[str, bool]]]:
    """Sample each cell, refusing a pick too close to one already taken."""
    by_name = {s.name: s for s in eligible}
    cells: dict[tuple[str, bool], list[ng.Station]] = collections.defaultdict(list)
    for station in eligible:
        cells[cell_of(station, cuts)].append(station)

    rng = random.Random(seed)
    taken: set[str] = set()
    picked = []
    # Cells in a fixed order, rarest first: the thin cells have the least room
    # to avoid an exclusion zone, so they choose before the crowded ones do.
    for cell in sorted(cells, key=lambda c: (len(cells[c]), c)):
        pool = list(cells[cell])
        rng.shuffle(pool)
        chosen = 0
        for station in pool:
            if chosen == per_cell:
                break
            if station.name in taken:
                continue
            picked.append((station, cell))
            taken |= within_hops(station.name, by_name, hops)
            chosen += 1
    return picked


def render(picked, cuts: tuple[float, float], day: str, pool: int) -> str:
    low, high = cuts
    lines = [
        "# Second cohort for the forecast comparison: sampled by kind of line.",
        "# Derived, not chosen — rebuild with tools/build_cohort.py.",
        f"# From the timetable of {day}: every station with >= {MIN_CALLS} calls",
        f"#   that day and not already polled ({pool} of them), split into six",
        f"#   cells by busiest-segment traffic (cuts at {low:.0f} and {high:.0f}",
        "#   trains a day) and by whether long-distance services call, then",
        f"#   {PER_CELL} sampled per cell with seed {SEED}, refusing any pick",
        f"#   within {MIN_HOPS} segments of one already taken.",
        "# Equal allocation, so this is NOT a representative sample of the",
        "#   network and is never pooled with the first twenty.",
        "# eva;name;calls;peak_segment;long_distance_share;degree;stratum",
    ]
    for station, (band, has_long) in sorted(picked, key=lambda p: int(p[0].eva)):
        stratum = f"{band}-{'mixed' if has_long else 'regional'}"
        lines.append(f"{station.eva};{station.name};{station.calls};"
                     f"{station.peak_segment};{station.long_share:.3f};"
                     f"{station.degree};{stratum}")
    return "\n".join(lines) + "\n"


def excluded(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                out.add(line.split(";")[0].strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="an archive day's raw IRIS parquet files")
    ap.add_argument("--day", required=True, help="the day those files cover")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="station files whose stations are already polled")
    ap.add_argument("--out", type=Path,
                    default=TOOLS / "forecast_stations_cohort2.csv")
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    args = ap.parse_args()

    stations, segments = ng.build(args.raw_dir)
    print(f"{len(stations)} stations, {len(segments)} segments in the timetable")

    # Cuts come from every eligible station, so the bands describe the network
    # rather than whatever is left after the exclusions.
    eligible = features(stations)
    cuts = bands(eligible)
    already = excluded(args.exclude)
    pool = [s for s in eligible if s.eva not in already]
    print(f"{len(eligible)} eligible, {len(pool)} not already polled; "
          f"peak-segment cuts at {cuts[0]:.0f} and {cuts[1]:.0f} trains a day")

    picked = draw(pool, cuts, per_cell=args.per_cell)
    counts = collections.Counter(cell for _, cell in picked)
    for cell, n in sorted(counts.items()):
        band, has_long = cell
        print(f"  {band:>4} {'mixed' if has_long else 'regional':>8}: {n}")
    args.out.write_text(render(picked, cuts, args.day, len(pool)), encoding="utf-8")
    print(f"wrote {len(picked)} stations to {args.out}")


if __name__ == "__main__":
    main()
