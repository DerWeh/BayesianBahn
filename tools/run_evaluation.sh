#!/usr/bin/env bash
# Score one or more collected days against DB, end to end.
#
# The collector must have been running on the day in question, and the piebro
# archive publishes that day's ground truth the following morning — so a day can
# only be scored from the next day onwards.
#
# Each stage is skipped when its output already exists, so a re-run after adding
# a day costs only the new day. Everything lands under tools/.scored/<day>/.
#
#   tools/run_evaluation.sh 2026-08-17 [2026-08-18 ...]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
scored="$root/tools/.scored"
stations="$(grep -v '^#' tools/forecast_stations.csv | cut -d';' -f1 | paste -sd,)"

for day in "$@"; do
  out="$scored/$day"
  mkdir -p "$out/raw"

  if [ ! -f "$out/truth.parquet" ]; then
    echo "== $day: fetching the archive"
    pixi run -e pipeline python pipeline/fetch_raw_day.py --date "$day" --out-dir "$out/raw"
    pixi run -e pipeline python pipeline/build_recent.py --date "$day" \
      --raw-dir "$out/raw" --out "$out/truth.parquet" --stations "$stations"
  fi

  echo "== $day: building events"
  # build_recent writes one file; load_truth globs a directory for data-*.parquet.
  mkdir -p "$out/truthdir" && cp -f "$out/truth.parquet" "$out/truthdir/data-recent-$day.parquet"
  pixi run -e pipeline python tools/score_events.py events --day "$day" \
    --truth archive --data-dir "$out/truthdir" --events-out "$out/arrivals.jsonl"
  pixi run -e pipeline python tools/score_events.py connections --day "$day" \
    --data-dir "$out/truthdir" --events-out "$out/connections.jsonl"

  # Shards are the history the model is allowed to see. They are cached across
  # days and trimmed to before the evaluated day inside the harness, so a fresh
  # download cannot leak the answer.
  echo "== $day: fetching history shards"
  pixi run -e pipeline python tools/fetch_shards.py --day "$day"

  for kind in arrivals connections; do
    for mode in live blind; do
      echo "== $day: scoring $kind ($mode)"
      env $([ "$mode" = blind ] && echo HARNESS_BLIND=1) \
        HARNESS_EVENTS="$out/$kind.jsonl" \
        HARNESS_SHARDS="$root/tools/.shards" \
        HARNESS_OUT="$out/$kind-$mode.jsonl" \
        HARNESS_DAY="$day" \
        pixi run ./gradlew testDebugUnitTest --tests '*ForecastHarness' -q --rerun-tasks
    done
  done
done

echo "== rendering the report"
pixi run -e pipeline python tools/report.py --scored-dir "$scored" --days "$@" \
  --out "$scored/report.html"
