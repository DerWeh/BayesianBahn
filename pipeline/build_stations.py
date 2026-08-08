"""Build the bundled station list from db-stations (DB open data, CC BY 4.0).

Writes `eva;name;weight;lat;lon`, sorted by weight descending so the search
field can offer the important stations first.

The coordinates are what let the journey search pick a transfer that lies
*towards* the destination. Ranking transfers by station size alone sent a
search from Ulm to Türkheim (Bay) through Stuttgart Hbf, which is 130 km the
wrong way, and the attempt budget ran out before the useful change at
Memmingen was reached.

Usage:
    python pipeline/build_stations.py [--out app/src/main/assets/stations.csv]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

SOURCE = "https://unpkg.com/db-stations/full.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("app/src/main/assets/stations.csv"))
    ap.add_argument("--source", default=SOURCE)
    args = ap.parse_args()

    with urllib.request.urlopen(args.source, timeout=120) as resp:  # noqa: S310
        raw = json.loads(resp.read())

    rows = []
    for station in raw:
        location = station.get("location") or {}
        lat, lon = location.get("latitude"), location.get("longitude")
        if station.get("id") is None or not station.get("name") or lat is None:
            continue
        rows.append((
            station["id"],
            station["name"],
            station.get("weight") or 0.0,
            lat,
            lon,
        ))
    # Sort on the unrounded weight, then round: that is how the shipped file was
    # produced, and keeping the order identical keeps the diff to the two new
    # columns.
    rows.sort(key=lambda r: -r[2])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for eva, name, weight, lat, lon in rows:
            fh.write(f"{eva};{name};{round(weight)};{lat:.5f};{lon:.5f}\n")
    print(f"wrote {len(rows)} stations to {args.out}")


if __name__ == "__main__":
    main()
