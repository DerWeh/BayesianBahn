"""Fetch the delay-history shards the evaluated trains need.

The published data bundled with the app covers sixteen stations around
Augsburg and München; the twenty stations this evaluation samples are a
different, nationwide set with no overlap. A real user searching Aachen gets
those trains through the app's *on-demand* path instead — one small shard per
train from the `shards` branch, with the daily `shards-recent` overlay on top.
This fetches exactly that, for exactly the trains the collected plan records
name, so the model is evaluated on the data a user would actually have.

The cutoff is a property of the published data, not of this script: the base
covers 2026-04..07 and the recent overlay ends the day before the evaluation
day. The harness asserts it rather than trusting it.

Usage:
    python tools/fetch_shards.py --day 2026-08-17 --out tools/.shards
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import collect_forecasts as cf

BRANCHES = {
    "base": "https://raw.githubusercontent.com/DerWeh/BayesianBahn/refs/heads/shards/",
    "recent": "https://raw.githubusercontent.com/DerWeh/BayesianBahn/refs/heads/shards-recent/",
}
PAUSE = 0.05


def shard_key(train_name: str) -> str:
    """Mirrors HistoryRepository.shardKey."""
    return re.sub(r"[^A-Za-z0-9]+", "_", train_name.strip()).strip("_").upper()


def candidate_keys(category: str, number: str, line: str | None) -> list[str]:
    """Mirrors HistoryRepository.candidateKeys: number first, then line."""
    keys = []
    if number.strip():
        keys.append(shard_key(f"{category} {number}"))
    if line and line.strip():
        keys.append(shard_key(line if line.startswith(category) else f"{category} {line}"))
    return list(dict.fromkeys(keys))


def trains_of(day, out: Path) -> set[tuple[str, str, str | None]]:
    records, _ = cf.Journal.read(out / f"forecasts-{day}.jsonl")
    return {(r["cat"], r["num"], r.get("line")) for r in records if r["t"] == "plan"}


def fetch(url: str) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": cf.UA})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None          # a train with no history is a real case
        raise
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True)
    ap.add_argument("--journal", type=Path, default=cf.OUT)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / ".shards")
    args = ap.parse_args()

    trains = trains_of(args.day, args.journal)
    keys = sorted({k for cat, num, line in trains for k in candidate_keys(cat, num, line)})
    print(f"{len(trains)} trains -> {len(keys)} shard keys", file=sys.stderr)

    counts = {"base": 0, "recent": 0, "missing": 0}
    for i, key in enumerate(keys, 1):
        got_any = False
        for tier, base_url in BRANCHES.items():
            target = args.out / tier / f"{key}.jgz"
            if target.exists():
                counts[tier] += 1
                got_any = True
                continue
            body = fetch(f"{base_url}{key}.jgz")
            time.sleep(PAUSE)
            if body is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Temp then rename: an interrupted run must not leave a truncated
            # shard that later looks like a complete one.
            tmp = target.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(target)
            counts[tier] += 1
            got_any = True
        if not got_any:
            counts["missing"] += 1
        if i % 100 == 0:
            print(f"  {i}/{len(keys)}", file=sys.stderr, flush=True)
    print(json.dumps(counts), file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
