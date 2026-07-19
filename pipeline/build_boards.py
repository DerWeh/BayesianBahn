"""Build per-station synthetic timetable boards from the archive.

IRIS publishes plan data only ~a day ahead; for trips further out the app
reconstructs a station's board from history. One small gzipped JSON per
station lists every train that recently called there with its typical
times and weekday pattern:

    {"name": "Ulm Hbf", "trains": [
        [category, number, line, arrTod|null, depTod|null,
         weekdayMask, lastSeenEpochDay], ...]}

weekdayMask bit i = ran on ISO weekday i+1 (bit 0 = Monday) within the
lookback window. Entries seen fewer than MIN_RUNS times are dropped.

Usage:
    python pipeline/build_boards.py --data-dir DATA_DIR --out-dir OUT_DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import polars as pl

# Only the recent timetable matters for planning ahead; older schedules
# (before the last timetable change) would poison the boards.
LOOKBACK_DAYS = 42
MIN_RUNS = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("data-*.parquet"))
    if not files:
        raise SystemExit(f"no data-*.parquet files in {args.data_dir}")

    lf = pl.concat([
        pl.scan_parquet(f).select(
            "station_name",
            "eva",
            "train_type",
            "train_number",
            "line_number",
            "arrival_planned_time",
            "departure_planned_time",
        )
        for f in files
    ]).with_columns(
        eva=pl.col("eva").str.strip_chars_start("0"),
        planned=pl.coalesce("departure_planned_time", "arrival_planned_time"),
    ).filter(
        pl.col("planned").is_not_null()
        & pl.col("train_type").is_not_null()
        & pl.col("train_number").is_not_null()
    )

    max_day = lf.select(pl.col("planned").dt.date().max()).collect(engine="streaming").item()
    tod = lambda c: (  # noqa: E731
        pl.col(c).dt.hour().cast(pl.Int32) * 60 + pl.col(c).dt.minute().cast(pl.Int32)
    )
    grouped = (
        lf.filter(pl.col("planned").dt.date() > pl.lit(max_day).dt.offset_by(f"-{LOOKBACK_DAYS}d"))
        .with_columns(
            arr_tod=tod("arrival_planned_time"),
            dep_tod=tod("departure_planned_time"),
            day=pl.col("planned").dt.date().cast(pl.Int32),
            weekday=pl.col("planned").dt.weekday().cast(pl.Int32),
        )
        .group_by(
            "eva", "station_name", "train_type", "train_number",
            "line_number", "arr_tod", "dep_tod",
        )
        .agg(
            mask=pl.lit(2, dtype=pl.Int32).pow(pl.col("weekday") - 1).cast(pl.Int32).unique().sum(),
            last=pl.col("day").max(),
            n=pl.len(),
        )
        .filter(pl.col("n") >= MIN_RUNS)
        .collect(engine="streaming")
    )
    print(f"{grouped.height} board entries at {grouped['eva'].n_unique()} stations "
          f"(window {LOOKBACK_DAYS}d up to {max_day})")

    boards: dict[str, dict] = defaultdict(lambda: {"trains": []})
    for row in grouped.iter_rows(named=True):
        board = boards[row["eva"]]
        board["name"] = row["station_name"]
        board["trains"].append([
            row["train_type"],
            row["train_number"],
            row["line_number"],
            row["arr_tod"],
            row["dep_tod"],
            row["mask"],
            row["last"],
        ])

    out = args.out_dir / "boards"
    out.mkdir(parents=True, exist_ok=True)
    for eva, board in boards.items():
        board["trains"].sort(key=lambda t: t[4] if t[4] is not None else t[3] or 0)
        blob = json.dumps(board, ensure_ascii=False, separators=(",", ":"))
        (out / f"{eva}.jgz").write_bytes(gzip.compress(blob.encode(), 9))
    total_mb = sum(f.stat().st_size for f in out.glob("*.jgz")) / 2**20
    print(f"wrote {len(boards)} station boards, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
