"""Build per-train delay-history shards from the piebro/deutsche-bahn-data archive.

Input:  monthly processed parquet files (CC BY 4.0, collected from DB's IRIS API).
Output: one gzipped JSON shard per train identity (train_type + train_number),
        holding for every station the train calls at its historical runs:
        [date, planned "HH:MM", arrival delay, departure delay,
         arrival delay at previous stop, canceled flag].

The app fetches exactly one shard per prediction, so shards must stay small
(a few KB gzipped). With --stations, only trains calling at those EVA numbers
are kept — used to bundle a draft subset as app assets.

Usage:
    pixi run -e pipeline python pipeline/build_shards.py \
        --data-dir DATA_DIR --out-dir OUT_DIR [--stations 8000013,8000261]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import polars as pl

# Delays outside this range are data glitches (e.g. day-crossing rewrites).
MIN_DELAY, MAX_DELAY = -30, 360

COLUMNS = [
    "station_name",
    "eva",
    "train_type",
    "train_number",
    "line_number",
    "train_line_ride_id",
    "train_line_station_num",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
    "is_canceled",
]


def train_key(train_type: str, train_number: str) -> str:
    """Filesystem/URL-safe shard key, mirrored in the app's HistoryRepository."""
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{train_type} {train_number}").strip("_").upper()


def scan(data_dir: Path) -> pl.LazyFrame:
    files = sorted(data_dir.glob("data-*.parquet"))
    if not files:
        raise SystemExit(f"no data-*.parquet files in {data_dir}")
    print(f"scanning {len(files)} monthly files")
    return pl.concat([pl.scan_parquet(f).select(COLUMNS) for f in files]).with_columns(
        # piebro zero-pads EVA numbers; IRIS and the app use unpadded ones.
        eva=pl.col("eva").str.strip_chars_start("0"),
        identity=pl.col("train_type") + " " + pl.col("train_number"),
    )


def load_events(data_dir: Path, station_evas: list[str] | None) -> pl.DataFrame:
    lf = scan(data_dir)
    if station_evas:
        wanted = (
            lf.filter(pl.col("eva").is_in(station_evas))
            .select("identity")
            .unique()
        )
        lf = lf.join(wanted, on="identity", how="semi")
    return lf.collect()


def build(events: pl.DataFrame) -> dict[str, dict]:
    minutes = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_minutes()  # noqa: E731

    df = (
        events.with_columns(
            arr_delay=minutes("arrival_change_time", "arrival_planned_time"),
            dep_delay=minutes("departure_change_time", "departure_planned_time"),
            planned=pl.coalesce("arrival_planned_time", "departure_planned_time"),
        )
        .filter(pl.col("planned").is_not_null())
        .with_columns(
            date=pl.col("planned").dt.date().cast(pl.String),
            hhmm=pl.col("planned").dt.strftime("%H:%M"),
        )
        # Arrival delay at the previous stop of the same ride, for conditioning
        # predictions on the live state of an approaching train.
        .sort("train_line_ride_id", "train_line_station_num")
        .with_columns(
            prev_delay=pl.coalesce("arr_delay", "dep_delay")
            .shift(1)
            .over("train_line_ride_id")
        )
    )

    shards: dict[str, dict] = defaultdict(lambda: {"stations": {}})
    clamp = lambda v: None if v is None else max(MIN_DELAY, min(MAX_DELAY, int(v)))  # noqa: E731

    for row in df.iter_rows(named=True):
        key = train_key(row["train_type"], row["train_number"])
        shard = shards[key]
        shard["train"] = row["identity"]
        shard["type"] = row["train_type"]
        if row["line_number"]:
            shard["line"] = row["line_number"]
        station = shard["stations"].setdefault(
            row["station_name"], {"eva": row["eva"], "runs": []}
        )
        station["runs"].append(
            [
                row["date"],
                row["hhmm"],
                clamp(row["arr_delay"]),
                clamp(row["dep_delay"]),
                clamp(row["prev_delay"]),
                1 if row["is_canceled"] else 0,
            ]
        )
    return shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stations", help="comma-separated EVA numbers; keep only trains calling there")
    args = ap.parse_args()

    station_evas = args.stations.split(",") if args.stations else None
    events = load_events(args.data_dir, station_evas)
    print(f"{events.height} events after filtering")

    shards = build(events)
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing {len(shards)} shards to {shard_dir}")

    index = {}
    for key, shard in shards.items():
        for station in shard["stations"].values():
            station["runs"].sort()
        blob = json.dumps(shard, ensure_ascii=False, separators=(",", ":"))
        (shard_dir / f"{key}.json.gz").write_bytes(gzip.compress(blob.encode(), 6))
        index[key] = sum(len(s["runs"]) for s in shard["stations"].values())

    (args.out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    total_mb = sum(f.stat().st_size for f in shard_dir.glob("*.json.gz")) / 2**20
    print(f"total shard size: {total_mb:.1f} MB, index entries: {len(index)}")


if __name__ == "__main__":
    main()
