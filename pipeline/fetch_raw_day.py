"""Download one day's raw IRIS logs from the piebro archive.

Upstream's file naming is not stable, and the old approach of hardcoding the
names hid that badly. Until 2026-07-26 a day's partition held
`hour_<hours>.parquet` in fixed six-hour groups; since then it holds
`date_<day>_hour_<hours>.parquet` with variable groups, and a day's first
hours (00-02) are published in the *previous* day's partition. The workflow
asked for the four old names, got 404 for each, and logged "raw files not
fully published" — so the daily overlay silently froze for two weeks while
the job stayed green.

So: list what is actually there, and treat "the partition exists but holds
nothing I recognise" as an error rather than as a day to skip.

A day is only ever condensed once, so a day that is still being uploaded must
not be fetched: it would be cached half-empty forever and quietly bias the
delay statistics. Hence the completeness check — the file names carry the hours
they cover, and all 24 have to be there.

Exit codes:
    0  files downloaded
    2  the day is not published, or not published in full, yet
    1  anything else, including an unrecognised layout — needs a human

Usage:
    python pipeline/fetch_raw_day.py --date 2026-08-07 --out-dir raw
"""

from __future__ import annotations

import argparse
import http.client
import json
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

REPO = "piebro/deutsche-bahn-data"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"

NOT_PUBLISHED = 2


def partition(day: date) -> str:
    """The archive's Hive-style path for a day, with unpadded month and day."""
    return f"raw_data/year={day.year}/month={day.month}/day={day.day}"


def _get(url: str, tries: int = 3) -> bytes:
    """Fetch a URL, retrying anything that is not a definitive answer.

    A raw day is 100-200 MB in four or five files and the CDN does sometimes
    close a connection mid-body. That surfaces as `http.client.IncompleteRead`,
    which is an `HTTPException` and *not* an `OSError`, so it used to escape the
    retry loop and abort a nine-day run on its last day.
    """
    last: Exception | None = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
                body = resp.read()
                declared = resp.headers.get("Content-Length")
            if declared is not None and len(body) != int(declared):
                # A clean close short of Content-Length; urllib only raises for
                # some of these, so the length is checked rather than trusted.
                raise http.client.IncompleteRead(body, int(declared) - len(body))
            return body
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise
            last = err
        except (OSError, http.client.HTTPException) as err:
            last = err
    raise RuntimeError(f"giving up on {url}: {last}")


def list_partition(day: date) -> list[str]:
    """File names in a day's partition; empty when the partition does not exist."""
    url = f"{TREE_URL}/{partition(day)}"
    try:
        entries = json.loads(_get(url))
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return []
        raise
    return sorted(e["path"].rsplit("/", 1)[-1] for e in entries if e.get("type") == "file")


def select_files(day: date, listings: dict[date, list[str]]) -> list[tuple[date, str]]:
    """Which files, in which partitions, hold `day`'s fetches.

    The date-prefixed layout is authoritative when present: it also tells us
    that a file sitting in this partition belongs to the *next* day, which the
    hour numbers alone cannot.
    """
    prefix = f"date_{day.isoformat()}_"
    chosen = [
        (part, name)
        for part, names in listings.items()
        for name in names
        if name.startswith(prefix) and name.endswith(".parquet")
    ]
    if chosen:
        return sorted(chosen, key=lambda pair: pair[1])
    # Pre-2026-07-26 layout: unprefixed, and never spilling into a neighbour.
    return sorted(
        (day, name)
        for name in listings.get(day, [])
        if name.startswith("hour_") and name.endswith(".parquet")
    )


def hours_covered(names: list[str]) -> set[int]:
    """The hours a set of file names covers; both layouts spell them out."""
    hours: set[int] = set()
    for name in names:
        _, _, tail = name.partition("hour_")
        hours.update(int(h) for h in tail.removesuffix(".parquet").split("_") if h.isdigit())
    return hours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    day = args.date
    # A day's early hours land in the previous day's partition, so both matter.
    listings = {d: list_partition(d) for d in (day - timedelta(days=1), day)}
    if not listings[day]:
        print(f"{day}: not published yet")
        raise SystemExit(NOT_PUBLISHED)

    files = select_files(day, listings)
    if not files:
        raise SystemExit(
            f"{day}: partition exists but holds no file this script recognises "
            f"({', '.join(listings[day])}). The archive's naming changed again — "
            f"update select_files() rather than letting the overlay go stale."
        )

    missing = sorted(set(range(24)) - hours_covered([name for _, name in files]))
    if missing:
        print(f"{day}: published only in part, missing hours {missing}")
        raise SystemExit(NOT_PUBLISHED)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for part, name in files:
        target = args.out_dir / name
        # Only a complete file is ever named `.parquet` (see below), so one that
        # is already there is one a previous attempt finished: a run resumed
        # after a network failure re-downloads only what is actually missing.
        if target.exists():
            continue
        body = _get(f"{FILE_URL}/{partition(part)}/{name}")
        # Write beside the target and rename, so an interrupted fetch leaves no
        # half a parquet file for a later stage to read as if it were the day.
        staging = target.with_suffix(".part")
        staging.write_bytes(body)
        staging.replace(target)
        fetched += 1
    print(f"{day}: fetched {fetched} raw files, {len(files) - fetched} already present")


if __name__ == "__main__":
    main()
