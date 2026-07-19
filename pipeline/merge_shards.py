"""Merge the monthly base and the daily recent shard sets into one directory.

The result is the per-shard on-demand set the app fetches train by train
(hosted on the repo's `shards` branch): every train's base runs plus the
freshest recent days, re-encoded in the columnar v2 layout. Base and recent
never overlap by construction (recent covers only days newer than the
newest monthly file), so merging is concatenation.

Usage:
    python pipeline/merge_shards.py --base history.zip \
        --recent RECENT_SHARD_DIR --out OUT_DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from datetime import date
from pathlib import Path

from build_shards import station_v2


def parse_station(block: dict) -> tuple[str, list[tuple]]:
    """v2 station block → (eva, run tuples) as build_shards produces them."""
    if "runs" in block:  # legacy v1 rows, e.g. an old downloaded base
        epoch = date(1970, 1, 1).toordinal()
        runs = [
            (
                date.fromisoformat(r[0]).toordinal() - epoch,
                int(r[1][:2]) * 60 + int(r[1][3:]),
                r[2],
                r[3] if r[3] is not None else r[2],
                r[4],
                bool(r[5]),
            )
            for r in block["runs"]
        ]
        return block["eva"], runs
    days = block["days"]
    tods = block["tod"]
    t = block.get("t")
    a = block["a"]
    d = block.get("d")
    p = block["p"]
    cancelled = set(block.get("c", []))
    runs = []
    day = 0
    for i in range(len(days)):
        day += days[i]
        arr = a[i]
        dep = d[i] if d and d[i] is not None else arr
        runs.append(
            (day, tods[t[i]] if t else tods[0], arr, dep, p[i], i in cancelled)
        )
    return block["eva"], runs


def merge(base: dict | None, recent: dict | None) -> dict:
    out = dict(recent or base)
    stations = {}
    for name in {*(base or {}).get("stations", {}), *(recent or {}).get("stations", {})}:
        merged_eva, runs = None, []
        for src in (base, recent):
            block = (src or {}).get("stations", {}).get(name)
            if block:
                eva, r = parse_station(block)
                merged_eva = merged_eva or eva
                runs.extend(r)
        stations[name] = station_v2(merged_eva, runs)
    out["stations"] = stations
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="base history.zip")
    ap.add_argument("--recent", type=Path, required=True, help="recent shards dir")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base_shards: dict[str, dict] = {}
    meta = {}
    with zipfile.ZipFile(args.base) as zf:
        for name in zf.namelist():
            if name.endswith(".jgz"):
                blob = gzip.decompress(zf.read(name))
                base_shards[name.removesuffix(".jgz")] = json.loads(blob)
            elif name == "meta.json":
                meta = json.loads(zf.read(name))

    recent_shards: dict[str, dict] = {}
    recent_meta = {}
    for f in args.recent.glob("*.jgz"):
        recent_shards[f.stem] = json.loads(gzip.decompress(f.read_bytes()))
    recent_meta_file = args.recent.parent / "meta.json"
    if recent_meta_file.is_file():
        recent_meta = json.loads(recent_meta_file.read_text())

    args.out.mkdir(parents=True, exist_ok=True)
    index = {}
    for key in {*base_shards, *recent_shards}:
        merged = merge(base_shards.get(key), recent_shards.get(key))
        blob = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
        (args.out / f"{key}.jgz").write_bytes(gzip.compress(blob.encode(), 9))
        index[key] = sum(len(s["days"]) for s in merged["stations"].values())

    for key in ("recent_from", "recent_through"):
        if key in recent_meta:
            meta[key] = recent_meta[key]
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    (args.out / "meta.json").write_text(json.dumps(meta))
    total_mb = sum(f.stat().st_size for f in args.out.glob("*.jgz")) / 2**20
    print(f"merged {len(index)} shards, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
