"""Tests for the evaluation driver.

It replaced a shell script, which was the one part of this pipeline that only
ran on Linux — pixi's task shell is cross-platform but has no loops, no
conditionals and none of the text tools the script leaned on. The pieces tested
here are the ones the shell did implicitly and Python has to do on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import run_evaluation as re_  # noqa: E402


def test_station_list_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    """`grep -v '^#' | cut -d';' -f1 | paste -sd,`, without needing any of them."""
    path = tmp_path / "stations.csv"
    path.write_text("# a comment\n\n8000001;Aachen Hbf;387\n8000310;Remagen;122\n",
                    encoding="utf-8")
    assert re_.stations(path) == "8000001,8000310"


def test_the_real_station_list_parses() -> None:
    got = re_.stations(TOOLS / "forecast_stations.csv")
    assert len(got.split(",")) == 20
    assert "8000310" in got, "Remagen, the one station serving TR"


def test_the_gradle_wrapper_is_chosen_per_platform(monkeypatch) -> None:
    monkeypatch.setattr(re_.os, "name", "nt")
    assert re_.gradle_wrapper().endswith("gradlew.bat")
    monkeypatch.setattr(re_.os, "name", "posix")
    assert re_.gradle_wrapper().endswith("gradlew")


def test_stages_run_through_the_interpreter_running_the_driver() -> None:
    """Not a bare "python", which on Windows may be a different one entirely."""
    assert re_.python("tools/report.py")[0] == sys.executable


def test_the_archive_fetch_is_skipped_when_its_output_exists(tmp_path, monkeypatch):
    """The slow, network-bound stage whose output never changes."""
    calls: list[list[str]] = []
    monkeypatch.setattr(re_, "run", lambda cmd, env=None: calls.append(cmd))
    day = "2026-08-17"
    truth = tmp_path / day / "truthdir"
    truth.mkdir(parents=True)
    (truth / f"data-recent-{day}.parquet").touch()

    re_.score_day(day, tmp_path, "8000001")

    joined = " ".join(" ".join(c) for c in calls)
    assert "fetch_raw_day" not in joined and "build_recent" not in joined
    assert "score_events.py" in joined and "fetch_shards.py" in joined


def test_the_archive_is_fetched_when_it_is_missing(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(re_, "run", lambda cmd, env=None: calls.append(cmd))
    re_.score_day("2026-08-17", tmp_path, "8000001")
    joined = " ".join(" ".join(c) for c in calls)
    assert "fetch_raw_day" in joined and "build_recent" in joined


def test_both_model_variants_are_scored_for_both_event_kinds(tmp_path, monkeypatch):
    envs: list[dict] = []
    monkeypatch.setattr(re_, "run",
                        lambda cmd, env=None: envs.append(env) if env else None)
    re_.score_day("2026-08-17", tmp_path, "8000001")
    outs = {Path(e["HARNESS_OUT"]).name for e in envs}
    assert outs == {"arrivals-live.jsonl", "arrivals-blind.jsonl",
                    "connections-live.jsonl", "connections-blind.jsonl"}
    blind = {Path(e["HARNESS_OUT"]).name for e in envs if "HARNESS_BLIND" in e}
    assert blind == {"arrivals-blind.jsonl", "connections-blind.jsonl"}
    assert all(e["HARNESS_DAY"] == "2026-08-17" for e in envs)


def test_a_failing_stage_stops_the_run(monkeypatch) -> None:
    """`set -e`, which the task shell has no equivalent for."""
    class Failed:
        returncode = 1
    monkeypatch.setattr(re_.subprocess, "run", lambda *a, **k: Failed())
    try:
        re_.run(["false"])
    except SystemExit as exit_:
        assert "stage failed" in str(exit_)
    else:
        raise AssertionError("a failing stage must stop the run")


def test_pixi_exposes_the_driver_as_a_task() -> None:
    """The task and the script must not drift apart."""
    manifest = (ROOT / "pixi.toml").read_text(encoding="utf-8")
    assert 'evaluate = "python tools/run_evaluation.py"' in manifest
    # It needs polars and the JDK, so it cannot run in the pipeline environment.
    assert 'evaluate = { features = ["pipeline", "evaluate"] }' in manifest


def test_the_truth_filter_covers_both_tiers(tmp_path: Path) -> None:
    """The archive is trimmed to the stations we might need a real arrival for,
    and the far end of a change is one of them."""
    origins = tmp_path / "o.csv"
    origins.write_text("8000001;Aachen Hbf;387\n", encoding="utf-8")
    ends = tmp_path / "d.csv"
    ends.write_text("# derived\n8000041;Bochum Hbf;223\n", encoding="utf-8")
    assert re_.stations(origins, ends) == "8000001,8000041"


def test_a_station_in_both_tiers_is_listed_once(tmp_path: Path) -> None:
    origins = tmp_path / "o.csv"
    origins.write_text("8000310;Remagen;122\n", encoding="utf-8")
    ends = tmp_path / "d.csv"
    ends.write_text("8000310;Remagen;40\n8000041;Bochum Hbf;223\n", encoding="utf-8")
    assert re_.stations(origins, ends) == "8000310,8000041"


def test_a_missing_tier_file_is_not_an_error(tmp_path: Path) -> None:
    origins = tmp_path / "o.csv"
    origins.write_text("8000001;Aachen Hbf;387\n", encoding="utf-8")
    assert re_.stations(origins, tmp_path / "nope.csv") == "8000001"


def test_the_real_two_tier_list_parses() -> None:
    got = re_.stations(TOOLS / "forecast_stations.csv",
                       TOOLS / "forecast_destinations.csv")
    evas = got.split(",")
    assert len(evas) == len(set(evas)), "the archive filter is a set"
    assert evas[:20] == re_.stations(TOOLS / "forecast_stations.csv").split(",")
