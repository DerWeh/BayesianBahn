"""Fetch the delay-history shards the evaluated trains need.

The published data bundled with the app covers sixteen stations around
Augsburg and München; the twenty stations this evaluation samples are a
different, nationwide set with no overlap. A real user searching Aachen gets
those trains through the app's *on-demand* path instead — one small shard per
train from the `shards` branch, with the daily `shards-recent` overlay on top.
This fetches exactly that, for exactly the trains the scoring reaches, so the
model is evaluated on the data a user would actually have.

Which trains those are is read from the event files, not from the day's plan
records. The collector polls 281 stations across two cohorts and two tiers; the
scoring reaches one cohort's origins and the far ends of their changes. Taking
the trains from the journal fetched a shard for every train seen anywhere —
29,600 of them on 2026-08-25, where the scoring needed 4,122. The other 86%
were half an hour of requests for histories nothing read.

The cutoff is a property of the published data, not of this script: the base
covers 2026-04..07 and the recent overlay ends the day before the evaluation
day. The harness asserts it rather than trusting it.

Usage:
    python tools/fetch_shards.py --day 2026-08-17 --out tools/.shards
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
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
    """Every train the day's plan records name — the whole polled network."""
    records, _ = cf.Journal.read(out / f"forecasts-{day}.jsonl")
    return {(r["cat"], r["num"], r.get("line")) for r in records if r["t"] == "plan"}


def trains_scored(events: list[Path]) -> set[tuple[str, str, str | None]]:
    """The trains the event files actually ask the model about.

    Three shapes, because the three event kinds name their trains differently:
    the feeder is `cat`/`num`/`line` on every one, a connection is the single
    string `conn` ("RE 4711", no line), and a journey carries its candidates as
    objects. Missing one of these would quietly starve the model of history for
    those trains — which reads as "no history" and is not otherwise visible.
    """
    trains: set[tuple[str, str, str | None]] = set()
    for path in events:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                event = json.loads(line)
                trains.add((event["cat"], event["num"], event.get("line")))
                if event.get("conn"):
                    cat, _, num = event["conn"].partition(" ")
                    trains.add((cat, num, None))
                for candidate in event.get("candidates") or ():
                    trains.add((candidate["cat"], candidate["num"],
                                candidate.get("line")))
    return trains


def fetch(url: str, tries: int = 3) -> bytes | None:
    """A shard's bytes, or None when the train genuinely has no history.

    Only a 404 means "no history". Everything else is retried, because a
    transport failure returned as None is indistinguishable from an empty
    history downstream: the train would silently be scored as one the model has
    never seen, which moves the numbers without leaving a trace.
    """
    request = urllib.request.Request(url, headers={"User-Agent": cf.UA})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
                declared = response.headers.get("Content-Length")
            if declared is not None and len(body) != int(declared):
                raise http.client.IncompleteRead(body, int(declared) - len(body))
            return body
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None      # a train with no history is a real case
            if attempt == tries - 1:
                raise
        except (OSError, http.client.HTTPException):
            if attempt == tries - 1:
                return None
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
    ap.add_argument("--events", type=Path, nargs="*", default=(),
                    help="event files to take the train list from; without "
                         "them every train in the day's journal is fetched")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / ".shards")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"concurrent requests (default {WORKERS}; 1 is the old behaviour)")
    args = ap.parse_args()

    day = dt.date.fromisoformat(args.day)
    trains = trains_scored(list(args.events)) if args.events else trains_of(
        args.day, args.journal)
    if args.events and not trains:
        raise SystemExit(
            f"{args.day}: the event files named no trains at all "
            f"({', '.join(str(p) for p in args.events)}). Fetching nothing "
            "would score every train as one the model has never seen.")
    keys = sorted({k for cat, num, line in trains for k in candidate_keys(cat, num, line)})
    source = "scored events" if args.events else "the whole journal"
    print(f"{len(trains)} trains from {source} -> {len(keys)} shard keys "
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
