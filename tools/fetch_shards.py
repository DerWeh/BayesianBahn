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
import datetime as dt
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import collect_forecasts as cf

BRANCHES = {
    "base": "https://raw.githubusercontent.com/DerWeh/BayesianBahn/refs/heads/shards/",
    "recent": "https://raw.githubusercontent.com/DerWeh/BayesianBahn/refs/heads/shards-recent/",
}
# One request at a time against a 275 ms round trip is 43 minutes for a day's
# ~4,000 trains, which was most of the time an evaluation took. The requests are
# independent, so they overlap; the cap keeps the burst polite against a public
# host and matches SyntheticTimetable.MAX_CONCURRENT_SHARDS, which asks the same
# host for the same files from the app.
WORKERS = 8
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


def is_fresh(target: Path, tier: str, day: dt.date) -> bool:
    """Whether a cached shard can stand in for the one published on `day`.

    The base branch is rebuilt monthly and covers months before any evaluated
    day, so once fetched it is always good. The *recent* overlay is rebuilt
    every morning and carries the last few days of runs — the freshest history a
    user would have had. A copy downloaded before the evaluated day cannot
    contain the runs from the days in between, so it silently starves the model
    of exactly the history that matters most.

    Nothing caught this: the harness asserts there is no *leak* (no run dated on
    or after the evaluated day) and a stale overlay passes that happily, being
    short of data rather than ahead of it. A copy fetched later than the
    evaluated day is fine — the harness trims the surplus away.
    """
    if not target.exists():
        return False
    if tier != "recent":
        return True
    fetched = dt.date.fromtimestamp(target.stat().st_mtime)
    return fetched >= day


def fetch_key(key: str, out: Path, day: dt.date) -> tuple[str, ...]:
    """Fetch one shard key from every tier. Returns the tiers that produced one.

    All the work for one key stays on one thread, so the temp-then-rename below
    is the only concurrency the filesystem sees: two threads never write the
    same path, because keys are unique.
    """
    got = []
    for tier, base_url in BRANCHES.items():
        target = out / tier / f"{key}.jgz"
        if is_fresh(target, tier, day):
            got.append(tier)
            continue
        body = fetch(f"{base_url}{key}.jgz")
        if PAUSE:
            time.sleep(PAUSE)
        if body is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Temp then rename: an interrupted run must not leave a truncated
        # shard that later looks like a complete one.
        tmp = target.with_suffix(f".{os.getpid()}-{threading.get_ident()}.part")
        tmp.write_bytes(body)
        tmp.replace(target)
        got.append(tier)
    return tuple(got)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True)
    ap.add_argument("--journal", type=Path, default=cf.OUT)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / ".shards")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"concurrent requests (default {WORKERS}; 1 is the old behaviour)")
    args = ap.parse_args()

    day = dt.date.fromisoformat(args.day)
    trains = trains_of(args.day, args.journal)
    keys = sorted({k for cat, num, line in trains for k in candidate_keys(cat, num, line)})
    print(f"{len(trains)} trains -> {len(keys)} shard keys "
          f"({args.workers} at a time)", file=sys.stderr)

    counts = {"base": 0, "recent": 0, "missing": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_key, key, args.out, day) for key in keys]
        for future in as_completed(futures):
            tiers = future.result()
            for tier in tiers:
                counts[tier] += 1
            if not tiers:
                counts["missing"] += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(keys)}", file=sys.stderr, flush=True)
    print(json.dumps(counts), file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
