"""Tests for the evaluation driver.

It replaced a shell script, which was the one part of this pipeline that only
ran on Linux — pixi's task shell is cross-platform but has no loops, no
conditionals and none of the text tools the script leaned on. The pieces tested
here are the ones the shell did implicitly and Python has to do on purpose.
"""

from __future__ import annotations

import sys

import pytest
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


def stub_run(monkeypatch, day: str, scored: Path, *, journeys: str = "",
             unpublished: tuple[str, ...] = ()):
    """Record the stages, and leave behind the file the driver decides on.

    `journeys` is what the event builder would have written: empty means the
    far end was not yet polled that day, which is a real state for every day
    before 2026-08-24 and the one the driver has to skip rather than run.

    `unpublished` names the days whose archive fetch reports NOT_PUBLISHED, so
    a test can put an unready day in the middle of a run.
    """
    calls: list[list[str]] = []
    envs: list[dict] = []

    def record(cmd, env=None, allow=()):
        calls.append(cmd)
        if env:
            envs.append(env)
        if "pipeline/fetch_raw_day.py" in " ".join(cmd):
            for d in unpublished:
                if d in cmd:
                    assert re_.NOT_PUBLISHED in allow, (
                        "the fetch stage must be allowed to report a missing day"
                    )
                    return re_.NOT_PUBLISHED
        if "journeys" in cmd:
            out = scored / day
            out.mkdir(parents=True, exist_ok=True)
            (out / "journeys.jsonl").write_text(journeys, encoding="utf-8")
        return 0

    monkeypatch.setattr(re_, "run", record)
    return calls, envs


def test_the_archive_fetch_is_skipped_when_its_output_exists(tmp_path, monkeypatch):
    """The slow, network-bound stage whose output never changes."""
    day = "2026-08-17"
    calls, _ = stub_run(monkeypatch, day, tmp_path)
    truth = tmp_path / day / "truthdir"
    truth.mkdir(parents=True)
    (truth / f"data-recent-{day}.parquet").touch()

    re_.score_day(day, tmp_path, "8000001")

    joined = " ".join(" ".join(c) for c in calls)
    assert "fetch_raw_day" not in joined and "build_recent" not in joined
    assert "score_events.py" in joined and "fetch_shards.py" in joined


def test_the_archive_is_fetched_when_it_is_missing(tmp_path, monkeypatch) -> None:
    calls, _ = stub_run(monkeypatch, "2026-08-17", tmp_path)
    re_.score_day("2026-08-17", tmp_path, "8000001")
    joined = " ".join(" ".join(c) for c in calls)
    assert "fetch_raw_day" in joined and "build_recent" in joined


def test_both_model_variants_are_scored_for_both_event_kinds(tmp_path, monkeypatch):
    _, envs = stub_run(monkeypatch, "2026-08-17", tmp_path, journeys='{"a":1}\n')
    re_.score_day("2026-08-17", tmp_path, "8000001")
    outs = {Path(e["HARNESS_OUT"]).name for e in envs}
    assert outs == {"arrivals-live.jsonl", "arrivals-blind.jsonl",
                    "connections-live.jsonl", "connections-blind.jsonl",
                    "journeys-live.jsonl", "journeys-blind.jsonl"}
    blind = {Path(e["HARNESS_OUT"]).name for e in envs if "HARNESS_BLIND" in e}
    assert blind == {"arrivals-blind.jsonl", "connections-blind.jsonl",
                     "journeys-blind.jsonl"}
    assert all(e["HARNESS_DAY"] == "2026-08-17" for e in envs)


def test_a_day_with_no_journeys_skips_the_journey_harness(tmp_path, monkeypatch):
    """Every day before the second tier started being polled. Two gradle round
    trips per day to discover an empty file costs minutes across a week."""
    _, envs = stub_run(monkeypatch, "2026-08-17", tmp_path, journeys="")
    re_.score_day("2026-08-17", tmp_path, "8000001")
    assert not any("JOURNEY" in k for e in envs for k in e)


def test_the_journey_events_are_built_even_when_they_are_not_scored(
        tmp_path, monkeypatch):
    """The builder still runs: its count of journeys with no answer from one
    side or the other is the diagnostic that says why a day is empty."""
    calls, _ = stub_run(monkeypatch, "2026-08-17", tmp_path, journeys="")
    re_.score_day("2026-08-17", tmp_path, "8000001")
    assert any("journeys" in c for c in calls)


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


def test_the_first_cohort_is_scored_without_being_asked_for(tmp_path, monkeypatch):
    """It is what every published figure so far speaks for, so it stays the
    default and the flag is what selects anything else."""
    calls, _ = stub_run(monkeypatch, "2026-08-17", tmp_path, journeys='{"a":1}\n')
    re_.score_day("2026-08-17", tmp_path, "8000001")
    joined = " ".join(" ".join(c) for c in calls)
    assert "--cohort" not in joined
    assert "forecast_destinations.csv" in joined


def test_a_later_cohort_is_scored_with_its_own_far_ends(tmp_path, monkeypatch):
    """Its own, not the first cohort's: the far ends were derived from its own
    origins, and scoring it against another cohort's would find almost none."""
    calls, _ = stub_run(monkeypatch, "2026-08-17", tmp_path, journeys='{"a":1}\n')
    re_.score_day("2026-08-17", tmp_path, "8000001", cohort=2)
    joined = " ".join(" ".join(c) for c in calls)
    assert "--cohort 2" in joined
    assert "forecast_destinations_cohort2.csv" in joined


def test_the_truth_filter_covers_every_cohort() -> None:
    """A far end with no recorded arrival cannot end a scored journey, so every
    station any cohort might need truth for has to survive the archive trim."""
    got = re_.stations(TOOLS / "forecast_stations.csv",
                       TOOLS / "forecast_destinations.csv",
                       TOOLS / "forecast_stations_cohort2.csv",
                       TOOLS / "forecast_destinations_cohort2.csv").split(",")
    import collect_forecasts as cf
    assert set(got) == {s.eva for s in cf.station_set(TOOLS)}
    assert len(got) == len(set(got))


def test_the_skip_code_matches_the_fetcher():
    """Mirrored across environments, so drift would silently make it fatal."""
    source = (ROOT / "pipeline" / "fetch_raw_day.py").read_text(encoding="utf-8")
    line = next(l for l in source.splitlines() if l.startswith("NOT_PUBLISHED"))
    assert int(line.split("=")[1]) == re_.NOT_PUBLISHED


def test_an_unpublished_day_is_skipped_not_scored(tmp_path, monkeypatch):
    """The archive lags, and a day it has not finished is not an error."""
    calls, _ = stub_run(monkeypatch, "2026-08-26", tmp_path,
                        unpublished=("2026-08-26",))
    assert re_.score_day("2026-08-26", tmp_path, "8000001") is False
    joined = [" ".join(c) for c in calls]
    assert any("fetch_raw_day.py" in c for c in joined)
    assert not any("score_events.py" in c for c in joined), (
        "nothing may be built from an archive that was never fetched"
    )


def test_a_scored_day_reports_success(tmp_path, monkeypatch):
    stub_run(monkeypatch, "2026-08-27", tmp_path, journeys='{"a": 1}\n')
    assert re_.score_day("2026-08-27", tmp_path, "8000001") is True


def test_one_unready_day_does_not_cancel_the_others(tmp_path, monkeypatch, capsys):
    """The bug this guards: 2026-08-26 was incomplete and took 27-30 with it.

    The later days are the more likely to be ready, so aborting the run on the
    earliest unready one is exactly backwards.
    """
    days = ["2026-08-26", "2026-08-27", "2026-08-28"]
    scored_days = []

    def fake_score_day(day, scored, station_list, cohort=1):
        if day == "2026-08-26":
            return False
        scored_days.append(day)
        return True

    monkeypatch.setattr(re_, "score_day", fake_score_day)
    reported: list[list[str]] = []
    monkeypatch.setattr(re_, "run", lambda cmd, env=None, allow=(): reported.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", ["run_evaluation.py", *days,
                                      "--scored-dir", str(tmp_path)])
    re_.main()

    assert scored_days == ["2026-08-27", "2026-08-28"]
    out = capsys.readouterr().out
    assert "2026-08-26" in out and "not published" in out
    report = next(c for c in reported if "tools/report.py" in " ".join(c))
    assert "2026-08-26" not in report, (
        "a skipped day in --days makes the report describe an empty directory"
    )
    assert "2026-08-27" in report and "2026-08-28" in report


def test_a_run_with_nothing_scoreable_stops(tmp_path, monkeypatch):
    """Rendering a report over no days would publish an empty page."""
    monkeypatch.setattr(re_, "score_day", lambda *a, **k: False)
    monkeypatch.setattr(sys, "argv", ["run_evaluation.py", "2026-08-26",
                                      "--scored-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as excinfo:
        re_.main()
    assert "no day could be scored" in str(excinfo.value)
