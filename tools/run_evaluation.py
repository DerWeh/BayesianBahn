"""Score one or more collected days against DB's own forecasts, end to end.

This was a shell script, which made it the one part of the evaluation that only
ran on Linux. pixi's task shell is cross-platform but deliberately small — no
loops, no conditionals, no `grep`/`cut` — and every stage here needs at least
one of those. Python is already required for the rest of the pipeline, so the
driver lives here and the pixi task is a one-line wrapper around it.

Each stage is skipped when its output already exists, so re-running after adding
a day costs only the new day. Everything lands under `tools/.scored/<day>/`.

The collector must have been running on the day in question, and the piebro
archive publishes that day's ground truth the following morning — so a day can
only be scored from the next day onwards.

Usage:
    pixi run -e evaluate evaluate 2026-08-17 [2026-08-18 ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def stations(*paths: Path) -> str:
    """The EVA numbers to keep ground truth for, comma separated.

    In the shell version this was `grep -v '^#' | cut -d';' -f1 | paste -sd,`,
    three tools that do not exist on Windows.

    Both tiers, because this list only trims the archive: the far end of a
    change needs a recorded arrival too. It does not widen what is scored —
    events and connections are built from the journal, which marks its own
    tiers — so adding the second tier leaves every published number unchanged
    and only makes the truth file a superset.
    """
    evas = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                evas.append(line.split(";")[0].strip())
    # Three of the twenty origins are also termini; the archive filter is a set.
    return ",".join(dict.fromkeys(evas))


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    """A stage. Failure stops the run — the shell needed `set -e` for this."""
    print("  $ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **(env or {})})
    if result.returncode != 0:
        raise SystemExit(f"stage failed ({result.returncode}): {' '.join(command)}")


def python(*args: str) -> list[str]:
    return [sys.executable, *args]


def gradle_wrapper() -> str:
    return str(ROOT / ("gradlew.bat" if os.name == "nt" else "gradlew"))


def score_day(day: str, scored: Path, station_list: str,
              cohort: int = 1) -> None:
    out = scored / day
    (out / "raw").mkdir(parents=True, exist_ok=True)
    # Which registered group of origins to score. The cohorts are never pooled:
    # they were sampled on different axes and started on different days, so a
    # figure over both describes neither. Cohort 1 is what the published page
    # speaks for; a later one is scored by asking for it.
    suffix = [] if cohort == 1 else ["--cohort", str(cohort)]
    ends = TOOLS / ("forecast_destinations.csv" if cohort == 1
                    else f"forecast_destinations_cohort{cohort}.csv")
    truth_dir = out / "truthdir"

    # The archive fetch is the slow, network-bound stage and its output never
    # changes, so it is the one most worth not repeating.
    if not any(truth_dir.glob("data-*.parquet")):
        print(f"== {day}: fetching the archive")
        run(python("pipeline/fetch_raw_day.py", "--date", day,
                   "--out-dir", str(out / "raw")))
        truth_dir.mkdir(parents=True, exist_ok=True)
        run(python("pipeline/build_recent.py", "--date", day,
                   "--raw-dir", str(out / "raw"),
                   "--out", str(truth_dir / f"data-recent-{day}.parquet"),
                   "--stations", station_list))

    print(f"== {day}: building events")
    run(python("tools/score_events.py", "events", "--day", day, "--truth", "archive",
               "--data-dir", str(truth_dir), *suffix,
               "--events-out", str(out / "arrivals.jsonl")))
    run(python("tools/score_events.py", "connections", "--day", day,
               "--data-dir", str(truth_dir), *suffix,
               "--events-out", str(out / "connections.jsonl")))
    # Two-leg journeys need a forecast at the far end, which only exists from
    # the day the second tier started being polled. Before that the builder
    # produces nothing, which is the honest outcome and not an error.
    run(python("tools/score_events.py", "journeys", "--day", day,
               "--data-dir", str(truth_dir), *suffix,
               "--destinations", str(ends),
               "--events-out", str(out / "journeys.jsonl")))

    # Shards are cached across days and trimmed inside the harness to runs
    # before the evaluated day, so a fresh download cannot leak the answer.
    print(f"== {day}: fetching history shards")
    run(python("tools/fetch_shards.py", "--day", day))

    for kind in ("arrivals", "connections"):
        for mode in ("live", "blind"):
            print(f"== {day}: scoring {kind} ({mode})")
            env = {
                "HARNESS_EVENTS": str(out / f"{kind}.jsonl"),
                "HARNESS_SHARDS": str(TOOLS / ".shards"),
                "HARNESS_OUT": str(out / f"{kind}-{mode}.jsonl"),
                "HARNESS_DAY": day,
            }
            if mode == "blind":
                env["HARNESS_BLIND"] = "1"
            run([gradle_wrapper(), "testDebugUnitTest", "--tests", "*ForecastHarness",
                 "-q", "--rerun-tasks"], env=env)

    # Skipped rather than run empty: a day before the second tier existed has
    # no journeys at all, and a gradle round trip per mode to discover that
    # costs minutes across a week.
    if (out / "journeys.jsonl").stat().st_size == 0:
        print(f"== {day}: no two-leg journeys (the far end was not yet polled)")
        return
    for mode in ("live", "blind"):
        print(f"== {day}: scoring journeys ({mode})")
        env = {
            "HARNESS_JOURNEYS": str(out / "journeys.jsonl"),
            "HARNESS_SHARDS": str(TOOLS / ".shards"),
            "HARNESS_OUT": str(out / f"journeys-{mode}.jsonl"),
            "HARNESS_DAY": day,
        }
        if mode == "blind":
            env["HARNESS_BLIND"] = "1"
        run([gradle_wrapper(), "testDebugUnitTest", "--tests", "*JourneyHarness",
             "-q", "--rerun-tasks"], env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("days", nargs="+", help="YYYY-MM-DD, one per collected day")
    ap.add_argument("--scored-dir", type=Path, default=TOOLS / ".scored")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--cohort", type=int, default=1,
                    help="which registered group of origins to score")
    # The collector runs on a laptop, so nothing in CI can rebuild this page:
    # publishing is a local render into the tree, then a commit.
    ap.add_argument("--publish", action="store_true",
                    help="render into docs/ for GitHub Pages instead of tools/.scored/")
    ap.add_argument("--publish-to", type=Path,
                    default=ROOT / "docs/evaluation/index.html")
    args = ap.parse_args()

    station_list = stations(TOOLS / "forecast_stations.csv",
                            TOOLS / "forecast_destinations.csv",
                            TOOLS / "forecast_stations_cohort2.csv",
                            TOOLS / "forecast_destinations_cohort2.csv")
    for day in args.days:
        score_day(day, args.scored_dir, station_list, args.cohort)

    if not args.skip_report:
        print("== rendering the report")
        out = args.publish_to if args.publish else args.scored_dir / "report.html"
        run(python("tools/report.py", "--scored-dir", str(args.scored_dir),
                   "--days", *args.days, "--out", str(out)))
        if args.publish:
            print(f"== wrote {out.relative_to(ROOT)}")
            print("   commit and push it; GitHub Pages serves /docs from the "
                  "default branch.")


if __name__ == "__main__":
    main()
