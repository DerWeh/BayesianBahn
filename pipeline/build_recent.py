"""Condense piebro raw_data day partitions into monthly-schema parquet.

The archive's `raw_data/year=Y/month=M/day=D/hour_*.parquet` files log the
IRIS `plan` and `fchg` XML responses for every station, updated daily —
weeks fresher than the monthly processed files. This script reduces one day
to the same columns as `data-YYYY-MM.parquet`, so `build_shards.py` can
consume monthly and daily files together.

The IRIS stop id has the form `{run}-{yymmddHHMM}-{stopnum}`: the first two
segments identify one daily run (no monthly ride-id ambiguity), the last is
the position on the route — exactly what the previous-stop feature needs.

A stop with no `fchg` entry counts as on time (change = planned), matching
the monthly files' semantics.

Usage:
    python pipeline/build_recent.py --date 2026-07-17 \
        --raw-dir RAW_DIR --out FILE.parquet [--stations 8000013,8000261]
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import polars as pl

SCHEMA = {
    "station_name": pl.String,
    "eva": pl.String,
    "train_type": pl.String,
    "train_number": pl.String,
    "line_number": pl.String,
    "train_line_ride_id": pl.String,
    "train_line_station_num": pl.Int64,
    "arrival_planned_time": pl.Datetime("us"),
    "arrival_change_time": pl.Datetime("us"),
    "departure_planned_time": pl.Datetime("us"),
    "departure_change_time": pl.Datetime("us"),
    "is_canceled": pl.Boolean,
}


def parse_time(raw: str | None) -> datetime | None:
    """IRIS times are yyMMddHHmm in German local time (naive, like monthly data)."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%y%m%d%H%M")
    except ValueError:
        return None


def iter_docs(files: list[Path]):
    """Yields (api_name, eva, xml_text) chronologically; fchg order matters (last wins)."""
    for f in sorted(files):
        df = (
            pl.read_parquet(f, columns=["timestamp", "api_name", "url", "response_data"])
            .filter(pl.col("api_name").is_in(["timetables/v1/plan", "timetables/v1/fchg"]))
            .sort("timestamp")
        )
        for api, url, xml_text in df.select("api_name", "url", "response_data").iter_rows():
            if not xml_text:
                continue
            # Plan documents carry no eva attribute — it only appears in the
            # request URL (…/plan/{eva}/{date}/{hour}, …/fchg/{eva}).
            m = re.search(r"/(?:plan|fchg)/0*(\d+)", url or "")
            if m:
                yield api, m.group(1), xml_text


def process_day(files: list[Path], station_evas: list[str] | None) -> pl.DataFrame:
    plans: dict[str, dict] = {}
    changes: dict[str, dict] = {}

    for api, eva, xml_text in iter_docs(files):
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            continue
        station = root.get("station")
        for s in root.iter("s"):
            sid = s.get("id")
            if not sid:
                continue
            ar, dp = s.find("ar"), s.find("dp")
            if api == "timetables/v1/plan":
                tl = s.find("tl")
                if sid in plans or tl is None:
                    continue
                line = None
                for ev in (ar, dp):
                    if ev is not None and ev.get("l"):
                        line = ev.get("l")
                        break
                plans[sid] = {
                    "station": station,
                    "eva": eva,
                    "type": tl.get("c") or "",
                    "number": tl.get("n") or "",
                    "line": line,
                    "ar_pt": parse_time(ar.get("pt") if ar is not None else None),
                    "dp_pt": parse_time(dp.get("pt") if dp is not None else None),
                }
            else:  # fchg: keep the latest state per stop
                ch = changes.setdefault(sid, {})
                for key, ev in (("ar", ar), ("dp", dp)):
                    if ev is None:
                        continue
                    ct = parse_time(ev.get("ct"))
                    if ct is not None:
                        ch[f"{key}_ct"] = ct
                    cs = ev.get("cs")
                    if cs is not None:
                        ch[f"{key}_cs"] = cs

    def run_id(sid: str) -> str:
        return sid.rsplit("-", 1)[0]

    def stop_num(sid: str) -> int | None:
        try:
            return int(sid.rsplit("-", 1)[1])
        except ValueError:
            return None

    if station_evas:
        wanted = {run_id(sid) for sid, p in plans.items() if p["eva"] in station_evas}
    else:
        wanted = None

    rows = []
    for sid, p in plans.items():
        if wanted is not None and run_id(sid) not in wanted:
            continue
        num = stop_num(sid)
        if num is None:
            continue
        ch = changes.get(sid, {})
        cancelled = ch.get("ar_cs") == "c" or ch.get("dp_cs") == "c"
        rows.append(
            {
                "station_name": p["station"],
                "eva": p["eva"],
                "train_type": p["type"],
                "train_number": p["number"],
                "line_number": p["line"] or None,
                "train_line_ride_id": run_id(sid),
                "train_line_station_num": num,
                "arrival_planned_time": p["ar_pt"],
                "arrival_change_time": ch.get("ar_ct") or p["ar_pt"],
                "departure_planned_time": p["dp_pt"],
                "departure_change_time": ch.get("dp_ct") or p["dp_pt"],
                "is_canceled": cancelled,
            }
        )
    return pl.DataFrame(rows, schema=SCHEMA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True, help="dir with that day's hour_*.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stations", help="comma-separated EVA numbers; keep only runs calling there")
    args = ap.parse_args()

    files = sorted(args.raw_dir.glob("hour_*.parquet"))
    if not files:
        raise SystemExit(f"no hour_*.parquet files in {args.raw_dir}")
    evas = args.stations.split(",") if args.stations else None
    df = process_day(files, evas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"{args.date}: {df.height} events -> {args.out}")


if __name__ == "__main__":
    main()
