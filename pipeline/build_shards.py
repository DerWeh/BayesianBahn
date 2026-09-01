"""Build per-train delay-history shards from the piebro/deutsche-bahn-data archive.

Input:  monthly processed parquet files (CC BY 4.0, collected from DB's IRIS API).
Output: one gzipped JSON shard per train identity (train_type + train_number),
        holding for every station the train calls at its historical runs in a
        columnar v2 layout (~63% smaller than the naive row format):

        {"eva": ..., "tod": [minutes-of-day...],   # deduplicated planned times
         "days": [epochDay, delta, delta, ...],    # delta-coded run dates
         "a": [arrival delays], "p": [previous-stop delays],
         "t": [tod index per run]     # omitted when only one planned time
         "d": [departure delays]      # omitted entirely / null where == arrival
         "c": [cancelled run indices]}

A second set of shards is written under the same format, keyed by line *and*
station rather than by run number. IRIS renumbers a run at every timetable
change, so a train that has run for years can arrive with almost no history;
its line has run all along, and the app falls back to it before it falls back
to the class-wide prior. Keyed by line alone these would be unusable — "S1"
names eight unrelated networks and one shard would be 3.4 MB — so each holds
one station and 45 days, a median of 1.4 KB.

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
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import polars as pl

# Delays outside this range are data glitches (e.g. day-crossing rewrites).
MIN_DELAY, MAX_DELAY = -30, 360

# Days of history a line shard keeps. A line shard exists to answer where a run
# number has nothing, and the model needs a couple of weeks of one time-of-day
# slot to clear its effective-sample floor; past that the recency decay has all
# but silenced the runs anyway (a 45-day-old run is worth an eighth of
# yesterday's). Keeping less is what makes a line affordable to fetch: a busy
# S-Bahn calls five hundred times a day at a station where a numbered run calls
# once.
LINE_DAYS = 45

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


def shard_key(name: str) -> str:
    """Filesystem/URL-safe shard key, mirrored in the app's HistoryRepository."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def train_key(train_type: str, train_number: str) -> str:
    """Key of the per-run shard: "ICE 512" -> ICE_512."""
    return shard_key(f"{train_type} {train_number}")


def line_name(train_type: str, line_number: str) -> str:
    """Display name of a line shard, mirroring HistoryRepository.candidateKeys.

    IRIS writes the line with its product already in it ("S7", "RE9"), but not
    always the *train's* product: a rail-replacement bus on the S7 arrives as
    type "Bus" line "S7", and its history has nothing to do with the trains on
    the S7. So the type is prepended whenever the line does not already start
    with it, which keeps those two apart — BUS_S7 against S7 — and the app
    builds the same string from the same rule.
    """
    return line_number if line_number.startswith(train_type) else f"{train_type} {line_number}"


def line_key(train_type: str, line_number: str, eva: str) -> str:
    """Key of a line shard: the fallback for a run number with no history.

    One shard per line *and station*, not per line. A line shard is fetched
    over the network for a single station's forecast, and a whole line is far
    too much to send for that: "S1" names a different line in Berlin, Hamburg,
    München, Stuttgart and four more networks, and pooling them all gives one
    3.4 MB file where the station the user asked about is 8 KB of it. Split
    this way the median line shard is a kilobyte and a half.
    """
    return shard_key(f"{line_name(train_type, line_number)} {eva}")


def prepare_month(
    file: Path,
    station_evas: list[str] | None,
    bucket: tuple[int, int] | None = None,
    keyed_by: str = "identity",
) -> pl.DataFrame:
    """One monthly file → filtered events with the prev-stop feature.

    Months are processed independently to bound memory: ride ids never span
    months, so the previous-stop computation loses nothing, and the semi-join
    (or the [bucket] hash for country-wide builds) shrinks the data before the
    expensive sort.

    [keyed_by] names the column both of those work on — "identity" for the
    per-run shards, "line_id" (line *and* station) for the line-keyed ones. It
    has to be the shard's own key for the bucketing to be sound: a bucket that
    splits a shard's runs across passes writes it twice, and the second write
    wins with only its own share of the history.
    """
    minutes = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_minutes()  # noqa: E731

    lf = pl.scan_parquet(file).select(COLUMNS).with_columns(
        # piebro zero-pads EVA numbers; IRIS and the app use unpadded ones.
        eva=pl.col("eva").str.strip_chars_start("0"),
        identity=pl.col("train_type") + " " + pl.col("train_number"),
        line_id=pl.col("train_type") + " " + pl.col("line_number") + " "
        + pl.col("eva").str.strip_chars_start("0"),
    )
    if keyed_by == "line_id":
        lf = lf.filter(pl.col("line_number").is_not_null() & (pl.col("line_number") != ""))
    if station_evas:
        wanted = (
            lf.filter(pl.col("eva").is_in(station_evas))
            .select(keyed_by)
            .unique()
        )
        lf = lf.join(wanted, on=keyed_by, how="semi")
    if bucket is not None:
        b, n = bucket
        lf = lf.filter(pl.col(keyed_by).hash(seed=0) % n == b)

    return (
        lf.with_columns(
            arr_delay=minutes("arrival_change_time", "arrival_planned_time"),
            dep_delay=minutes("departure_change_time", "departure_planned_time"),
            planned=pl.coalesce("arrival_planned_time", "departure_planned_time"),
        )
        .filter(
            pl.col("planned").is_not_null()
            & pl.col("train_type").is_not_null()
            & pl.col("train_number").is_not_null()
        )
        # Arrival delay at the previous stop of the same daily run, for
        # conditioning predictions on the live state of an approaching train.
        # ride_id identifies the route pattern (shared by all days of a
        # month), so partition additionally by service day — planned time
        # shifted by 4h keeps post-midnight stops with their run.
        .with_columns(
            service_day=(pl.col("planned") - pl.duration(hours=4)).dt.date()
        )
        .sort("train_line_ride_id", "service_day", "train_line_station_num")
        .with_columns(
            prev_delay=pl.coalesce("arr_delay", "dep_delay")
            .shift(1)
            .over("train_line_ride_id", "service_day")
        )
        .select(
            "station_name",
            "eva",
            "identity",
            "line_id",
            "train_type",
            "train_number",
            "line_number",
            "is_canceled",
            "arr_delay",
            "dep_delay",
            "prev_delay",
            day=pl.col("planned").dt.date().cast(pl.Int32),
            tod=pl.col("planned").dt.hour().cast(pl.Int32) * 60
            + pl.col("planned").dt.minute().cast(pl.Int32),
        )
        .collect(engine="streaming")
    )


def build_into(shards: dict[str, dict], df: pl.DataFrame,
               keyed_by: str = "identity") -> None:
    """Accumulates a month's events into the shards they belong to.

    A line shard is the same structure under a different key: every run of the
    line at each station, rather than every run of one train number. Nothing
    downstream needs to tell them apart — the app reads both with the same
    parser and the same model — so they are built by the same code.
    """
    clamp = lambda v: None if v is None else max(MIN_DELAY, min(MAX_DELAY, int(v)))  # noqa: E731
    by_line = keyed_by == "line_id"

    for row in df.iter_rows(named=True):
        key = (line_key(row["train_type"], row["line_number"], row["eva"]) if by_line
               else train_key(row["train_type"], row["train_number"]))
        shard = shards[key]
        shard["train"] = (line_name(row["train_type"], row["line_number"]) if by_line
                          else row["identity"])
        shard["type"] = row["train_type"]
        if row["line_number"]:
            shard["line"] = row["line_number"]
        station = shard["stations"].setdefault(
            row["station_name"], {"eva": row["eva"], "runs": []}
        )
        station["runs"].append(
            (
                row["day"],
                row["tod"],
                clamp(row["arr_delay"]),
                clamp(row["dep_delay"]),
                clamp(row["prev_delay"]),
                row["is_canceled"],
            )
        )


def station_v2(eva: str, runs: list[tuple], since: int | None = None) -> dict:
    """Columnar station block; see the module docstring for the layout.

    [since] drops runs before that epoch day. Whole days rather than a count of
    runs, because the model asks for the runs within twenty minutes of one
    planned time: cutting a busy line off after N runs would leave every slot
    with a couple of days, while cutting at a date leaves every slot with the
    same span. And an absolute date rather than "the last N days of this
    station", so that a line which stopped running months ago publishes nothing
    instead of publishing a stale shard the app would answer from.
    """
    runs.sort(key=lambda r: (r[0], r[1]))
    if since is not None:
        runs = [r for r in runs if r[0] >= since]
    tods = sorted({r[1] for r in runs})
    tod_index = {tod: i for i, tod in enumerate(tods)}
    days, t, a, d, p, c = [], [], [], [], [], []
    prev_day = None
    for i, (day, tod, arr, dep, prv, canc) in enumerate(runs):
        days.append(day if prev_day is None else day - prev_day)
        prev_day = day
        t.append(tod_index[tod])
        a.append(arr)
        # None also when dep == arr: the app falls back to arrival anyway.
        d.append(None if dep == arr else dep)
        p.append(prv)
        if canc:
            c.append(i)
    block = {"eva": eva, "tod": tods, "days": days, "a": a, "p": p}
    if len(tods) > 1:
        block["t"] = t
    if any(x is not None for x in d):
        block["d"] = d
    if c:
        block["c"] = c
    return block


def write_shards(shards: dict[str, dict], shard_dir: Path, index: dict[str, int],
                 since: int | None = None) -> None:
    for key, shard in shards.items():
        stations = {
            name: station_v2(st["eva"], st["runs"], since)
            for name, st in shard["stations"].items()
        }
        # A trim can empty a station, and a whole shard with it — a line that
        # stopped running before the window. Writing it would publish an empty
        # history that the app cannot tell from a missing one.
        stations = {name: block for name, block in stations.items() if block["a"]}
        if not stations:
            continue
        out = {
            "v": 2,
            "train": shard["train"],
            "type": shard["type"],
            "stations": stations,
        }
        if "line" in shard:
            out["line"] = shard["line"]
        blob = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
        (shard_dir / f"{key}.jgz").write_bytes(gzip.compress(blob.encode(), 9))
        index[key] = sum(len(s["a"]) for s in out["stations"].values())


def last_days(files) -> dict[Path, int]:
    """Each file's newest planned day, as the epoch-day integer shards store.

    One column max per file, so parquet statistics do most of it. Cheap enough
    to pay for what it buys: the line pass keeps a fixed window, and every file
    ending before that window is one the pass would read in full — once per
    hash bucket, sixteen times over on a country-wide build — only to throw all
    of it away at write time.
    """
    ends = {}
    for f in files:
        value = (
            pl.scan_parquet(f)
            .select(
                pl.coalesce("arrival_planned_time", "departure_planned_time")
                .max().dt.date().cast(pl.Int32)
            )
            .collect()
            .item()
        )
        if value is not None:
            ends[f] = int(value)
    return ends


def build_pass(files, station_evas, buckets: int, shard_dir: Path,
               index: dict[str, int], keyed_by: str,
               since: int | None = None) -> None:
    """One hash-partitioned sweep over the archive, writing one kind of shard."""
    total = 0
    for b in range(buckets):
        bucket = (b, buckets) if buckets > 1 else None
        # Stream one file at a time through the shard dict to bound memory.
        shards: dict[str, dict] = defaultdict(lambda: {"stations": {}})
        bucket_total = 0
        for f in files:
            df = prepare_month(f, station_evas, bucket, keyed_by)
            build_into(shards, df, keyed_by)
            bucket_total += df.height
        write_shards(shards, shard_dir, index, since)
        total += bucket_total
        label = f"bucket {b + 1}/{buckets}: " if buckets > 1 else ""
        print(f"  {label}{keyed_by}: {bucket_total} events, {len(shards)} shards",
              flush=True)
    print(f"{total} events after filtering ({keyed_by})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stations", help="comma-separated EVA numbers; keep only trains calling there")
    ap.add_argument(
        "--buckets",
        type=int,
        default=1,
        help="hash-partition trains into N passes; use ~16 for country-wide "
        "builds so only 1/N of the data is in memory at a time",
    )
    ap.add_argument(
        "--line-days",
        type=int,
        default=LINE_DAYS,
        help="days of history a line shard keeps per station; 0 writes none",
    )
    args = ap.parse_args()

    station_evas = args.stations.split(",") if args.stations else None
    files = sorted(args.data_dir.glob("data-*.parquet"))
    if not files:
        raise SystemExit(f"no data-*.parquet files in {args.data_dir}")

    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, int] = {}
    build_pass(files, station_evas, args.buckets, shard_dir, index, "identity")
    trains = len(index)
    print(f"wrote {trains} shards to {shard_dir}")

    if args.line_days > 0:
        ends = last_days(files)
        since = max(ends.values()) - args.line_days + 1 if ends else None
        recent_files = [f for f in files if ends.get(f, since or 0) >= (since or 0)]
        lines: dict[str, int] = {}
        build_pass(recent_files, station_evas, args.buckets, shard_dir, lines,
                   "line_id", since)
        # A line shard that lands on a train shard's key would replace a
        # train's own history with its line's, and the app would answer from it
        # without a word. The two rules cannot collide today — a run number is
        # digits, a line name is not — but nothing upstream promises that.
        clash = sorted(set(index) & set(lines))
        if clash:
            raise SystemExit(
                f"{len(clash)} line shards would overwrite a train shard: "
                f"{', '.join(clash[:5])}"
            )
        index.update(lines)
        print(f"wrote {len(lines)} line shards to {shard_dir}")

    (args.out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    meta = {
        "generated": date.today().isoformat(),
        "stations": station_evas or [],
        "months": sorted(
            f.stem.removeprefix("data-")
            for f in args.data_dir.glob("data-*.parquet")
            if re.fullmatch(r"data-\d{4}-\d{2}", f.stem)
        ),
        "trains": trains,
        "lines": len(index) - trains,
        "line_days": args.line_days,
    }
    # Daily files from build_recent.py (data-recent-YYYY-MM-DD.parquet).
    recent_days = sorted(
        f.stem.removeprefix("data-recent-")
        for f in args.data_dir.glob("data-recent-*.parquet")
    )
    if recent_days:
        meta["recent_from"] = recent_days[0]
        meta["recent_through"] = recent_days[-1]
    (args.out_dir / "meta.json").write_text(json.dumps(meta))
    total_mb = sum(f.stat().st_size for f in shard_dir.glob("*.jgz")) / 2**20
    print(f"total shard size: {total_mb:.1f} MB, index entries: {len(index)}")

    # Single flat archive the app's DataUpdater downloads; shards are already
    # gzipped, so store without recompression.
    with zipfile.ZipFile(args.out_dir / "history.zip", "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(shard_dir.glob("*.jgz")):
            zf.write(f, f.name)
        zf.write(args.out_dir / "index.json", "index.json")
        zf.write(args.out_dir / "meta.json", "meta.json")
    print(f"wrote {args.out_dir / 'history.zip'}")


if __name__ == "__main__":
    main()
