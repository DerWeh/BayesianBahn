"""Tests for the data pipeline.

The pipeline has two code paths that run at very different rates: the daily one
(condense raw days, rebuild the recent overlays) and the monthly one (rebuild
the base and the synthetic boards, once a new monthly archive file appears).
The monthly path broke on a schema detail and nothing noticed for a week,
because the only thing exercising it was the monthly run itself. These tests
exercise both paths on synthetic fixtures on every push.

Fixtures mirror the two sources faithfully, including the details that differ
between them: the archive publishes nanosecond timestamps and zero-padded EVA
numbers, build_recent.py writes microsecond timestamps and unpadded ones.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import sys
import zipfile
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_boards  # noqa: E402
import build_recent  # noqa: E402
import build_shards  # noqa: E402
import fetch_raw_day  # noqa: E402

# The schema of piebro/deutsche-bahn-data's monthly_processed_data files. Kept
# here so a fixture cannot silently drift from the real input;
# test_monthly_fixture_matches_archive checks it against a real file when one
# is available locally.
MONTHLY_SCHEMA = {
    "station_name": pl.String,
    "xml_station_name": pl.String,
    "eva": pl.String,
    "train_number": pl.String,
    "line_number": pl.String,
    "final_destination_station": pl.String,
    "delay_in_min": pl.Int32,
    "time": pl.Datetime("ns"),
    "is_canceled": pl.Boolean,
    "train_type": pl.String,
    "train_line_ride_id": pl.String,
    "train_line_station_num": pl.Int32,
    "arrival_planned_time": pl.Datetime("ns"),
    "arrival_change_time": pl.Datetime("ns"),
    "departure_planned_time": pl.Datetime("ns"),
    "departure_change_time": pl.Datetime("ns"),
    "id": pl.String,
}
RECENT_SCHEMA = build_recent.SCHEMA

ULM, AUGSBURG = "8000170", "8000013"


def _stop(schema, day, *, eva, station, ttype, number, line, minute, delay=0,
          station_num=1, canceled=False):
    """One stop of one train on one day, in whichever source schema."""
    arr = dt.datetime.combine(day, dt.time(8, 0)) + dt.timedelta(minutes=minute)
    dep = arr + dt.timedelta(minutes=2)
    ride = f"{number}-{day:%y%m%d}"
    row = {
        # The archive zero-pads EVA numbers; IRIS (and so build_recent.py) does not.
        "eva": f"0{eva}" if schema is MONTHLY_SCHEMA else eva,
        "station_name": station,
        "train_type": ttype,
        "train_number": number,
        "line_number": line,
        "train_line_ride_id": ride,
        "train_line_station_num": station_num,
        "arrival_planned_time": arr,
        "arrival_change_time": arr + dt.timedelta(minutes=delay),
        "departure_planned_time": dep,
        "departure_change_time": dep + dt.timedelta(minutes=delay),
        "is_canceled": canceled,
    }
    if schema is MONTHLY_SCHEMA:
        row |= {
            "xml_station_name": station,
            "final_destination_station": "Muenchen Hbf",
            "delay_in_min": delay,
            "time": arr,
            "id": f"{ride}-{station_num}",
        }
    return row


def _write(path: Path, schema, rows) -> Path:
    df = pl.DataFrame(rows).select([pl.col(c).cast(t) for c, t in schema.items()])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def _days(start: dt.date, count: int) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range(count)]


@pytest.fixture
def mixed_dir(tmp_path: Path) -> Path:
    """A data dir as the monthly rebuild sees it: monthly files plus cached days.

    RE 4711 calls at Ulm on six June days (monthly file); RB 5000 calls at
    Augsburg on four July days (daily files). Both must survive into the output.
    """
    data = tmp_path / "data"
    june = _days(dt.date(2026, 6, 25), 6)
    _write(
        data / "data-2026-06.parquet",
        MONTHLY_SCHEMA,
        [
            _stop(MONTHLY_SCHEMA, day, eva=ULM, station="Ulm Hbf", ttype="RE",
                  number="4711", line="9", minute=5, delay=i % 4)
            for i, day in enumerate(june)
        ],
    )
    for day in _days(dt.date(2026, 7, 1), 4):
        _write(
            data / f"data-recent-{day:%Y-%m-%d}.parquet",
            RECENT_SCHEMA,
            [
                _stop(RECENT_SCHEMA, day, eva=AUGSBURG, station="Augsburg Hbf",
                      ttype="RB", number="5000", line="7", minute=30, delay=1)
            ],
        )
    return data


def _run(module, monkeypatch, data_dir, out_dir: Path, *extra: str) -> None:
    dirs = [data_dir] if isinstance(data_dir, (str, Path)) else data_dir
    monkeypatch.setattr(sys, "argv", [
        module.__name__,
        "--data-dir", *[str(d) for d in dirs],
        "--out-dir", str(out_dir),
        *extra,
    ])
    module.main()


def _board(out_dir: Path, eva: str) -> dict:
    return json.loads(gzip.decompress((out_dir / "boards" / f"{eva}.jgz").read_bytes()))


@pytest.mark.skipif(
    not sorted(Path(__file__).resolve().parents[1].glob("data/data-2*.parquet")),
    reason="no local copy of the monthly archive to compare against",
)
def test_monthly_fixture_matches_archive() -> None:
    """The fixture is only worth anything if it still matches the real input."""
    real = sorted(Path(__file__).resolve().parents[1].glob("data/data-2*.parquet"))[-1]
    assert dict(pl.scan_parquet(real).collect_schema()) == MONTHLY_SCHEMA


def test_nightly_job_runs_the_environment_these_tests_ran_in() -> None:
    """Otherwise these tests vouch for an environment production never installs.

    The job used to `pip install "polars>=1.0"`, so the nightly data build could
    silently be a different polars from the one anything had been tested with.
    pixi.lock is now the single definition of both.
    """
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/update-data.yml"
    ).read_text()
    assert "prefix-dev/setup-pixi" in workflow
    assert re.search(r"activate-environment:\s*pipeline", workflow), (
        "the pipeline environment must be on PATH, or the steps run the runner's "
        "bare python"
    )
    assert not re.search(r"^\s*run:.*pip install", workflow, re.M), (
        "installing dependencies outside pixi.lock reintroduces the drift this "
        "test exists to prevent"
    )


def test_boards_reads_monthly_and_recent_together(mixed_dir, tmp_path, monkeypatch) -> None:
    """Regression: nanosecond archive files and microsecond daily files in one concat.

    This is what broke the monthly rebuild — polars refuses to concatenate
    frames whose datetime units differ, so the step died with exit code 1 every
    night once a new monthly file triggered a rebuild.
    """
    out = tmp_path / "out"
    _run(build_boards, monkeypatch, mixed_dir, out)

    ulm, augsburg = _board(out, ULM), _board(out, AUGSBURG)
    assert ulm["name"] == "Ulm Hbf"
    assert augsburg["name"] == "Augsburg Hbf"
    # Board keys are unpadded EVA numbers even though the archive pads them.
    assert not (out / "boards" / f"0{ULM}.jgz").exists()
    assert [t[:3] for t in ulm["trains"]] == [["RE", "4711", "9"]]
    assert [t[:3] for t in augsburg["trains"]] == [["RB", "5000", "7"]]


def test_boards_weekday_mask_and_times(mixed_dir, tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    _run(build_boards, monkeypatch, mixed_dir, out)

    ttype, number, line, arr_tod, dep_tod, mask, last = _board(out, ULM)["trains"][0]
    assert (ttype, number, line) == ("RE", "4711", "9")
    assert (arr_tod, dep_tod) == (8 * 60 + 5, 8 * 60 + 7)
    june = _days(dt.date(2026, 6, 25), 6)
    assert mask == sum(1 << (d.isoweekday() - 1) for d in june)
    assert last == june[-1].toordinal() - dt.date(1970, 1, 1).toordinal()


def test_boards_drop_rare_and_stale_entries(tmp_path, monkeypatch) -> None:
    """Below MIN_RUNS, or older than the lookback window, must not reach a board."""
    data = tmp_path / "data"
    newest = dt.date(2026, 6, 30)
    rows = [
        # Enough runs, inside the window: kept.
        *[_stop(MONTHLY_SCHEMA, d, eva=ULM, station="Ulm Hbf", ttype="RE",
                number="4711", line="9", minute=5)
          for d in _days(newest - dt.timedelta(days=4), 5)],
        # Only two runs: below MIN_RUNS.
        *[_stop(MONTHLY_SCHEMA, d, eva=ULM, station="Ulm Hbf", ttype="RE",
                number="4712", line="9", minute=45)
          for d in _days(newest - dt.timedelta(days=1), 2)],
        # Plenty of runs, but before the last timetable change.
        *[_stop(MONTHLY_SCHEMA, d, eva=ULM, station="Ulm Hbf", ttype="RE",
                number="4713", line="9", minute=15)
          for d in _days(newest - dt.timedelta(days=build_boards.LOOKBACK_DAYS + 10), 5)],
    ]
    _write(data / "data-2026-06.parquet", MONTHLY_SCHEMA, rows)

    out = tmp_path / "out"
    _run(build_boards, monkeypatch, data, out)
    assert {t[1] for t in _board(out, ULM)["trains"]} == {"4711"}


def test_boards_reads_several_data_dirs(tmp_path, monkeypatch) -> None:
    """The monthly archive and the daily cache are read where they lie."""
    monthly, recent = tmp_path / "data", tmp_path / "recent-cache"
    june = _days(dt.date(2026, 6, 25), 4)
    _write(monthly / "data-2026-06.parquet", MONTHLY_SCHEMA, [
        _stop(MONTHLY_SCHEMA, d, eva=ULM, station="Ulm Hbf", ttype="RE",
              number="4711", line="9", minute=5) for d in june
    ])
    for day in _days(dt.date(2026, 7, 1), 3):
        _write(recent / f"data-recent-{day:%Y-%m-%d}.parquet", RECENT_SCHEMA, [
            _stop(RECENT_SCHEMA, day, eva=AUGSBURG, station="Augsburg Hbf",
                  ttype="RB", number="5000", line="7", minute=30)
        ])

    out = tmp_path / "out"
    _run(build_boards, monkeypatch, [monthly, recent], out)
    assert _board(out, ULM)["trains"] and _board(out, AUGSBURG)["trains"]


def test_boards_tolerate_an_empty_or_missing_cache(tmp_path, monkeypatch) -> None:
    """Regression: the nightly job crashed with a bare `FileNotFoundError`.

    A new monthly archive file supersedes everything in the daily cache, so the
    cache is legitimately empty for a few days afterwards — which is precisely
    when the monthly rebuild runs.
    """
    monthly, missing = tmp_path / "data", tmp_path / "recent-cache"
    _write(monthly / "data-2026-06.parquet", MONTHLY_SCHEMA, [
        _stop(MONTHLY_SCHEMA, d, eva=ULM, station="Ulm Hbf", ttype="RE",
              number="4711", line="9", minute=5)
        for d in _days(dt.date(2026, 6, 25), 4)
    ])
    assert not missing.exists()
    _run(build_boards, monkeypatch, [monthly, missing], tmp_path / "out-missing")
    assert _board(tmp_path / "out-missing", ULM)["trains"]

    missing.mkdir()
    _run(build_boards, monkeypatch, [monthly, missing], tmp_path / "out-empty")
    assert _board(tmp_path / "out-empty", ULM)["trains"]


def test_boards_refuse_to_run_on_no_data(tmp_path, monkeypatch) -> None:
    """Every directory empty is a broken job, not an empty board set."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        _run(build_boards, monkeypatch, [empty], tmp_path / "out")
    assert "nothing" in str(excinfo.value)


def test_shards_reads_monthly_and_recent_together(mixed_dir, tmp_path, monkeypatch) -> None:
    """build_recent.py promises its output is consumable alongside monthly files."""
    out = tmp_path / "out"
    _run(build_shards, monkeypatch, mixed_dir, out)

    index = json.loads((out / "index.json").read_text())
    assert set(index) == {"RE_4711", "RB_5000"}

    shard = json.loads(gzip.decompress((out / "shards" / "RE_4711.jgz").read_bytes()))
    station = shard[ULM] if isinstance(shard, dict) and ULM in shard else shard
    assert station, "the Ulm stop must survive into the shard"

    meta = json.loads((out / "meta.json").read_text())
    assert meta["months"] == ["2026-06"]
    assert (meta["recent_from"], meta["recent_through"]) == ("2026-07-01", "2026-07-04")

    with zipfile.ZipFile(out / "history.zip") as zf:
        assert {"index.json", "meta.json"} <= set(zf.namelist())


def test_shards_station_filter_keeps_whole_runs(mixed_dir, tmp_path, monkeypatch) -> None:
    """--stations selects trains by identity, so the regional zip stays coherent."""
    out = tmp_path / "out"
    _run(build_shards, monkeypatch, mixed_dir, out, "--stations", AUGSBURG)
    assert set(json.loads((out / "index.json").read_text())) == {"RB_5000"}


def _plan_xml(station: str, sid: str, arr: str, dep: str, line: str = "9") -> str:
    return (
        f'<timetable station="{station}"><s id="{sid}">'
        f'<tl c="RE" n="4711"/>'
        f'<ar pt="{arr}" l="{line}"/><dp pt="{dep}" l="{line}"/>'
        f"</s></timetable>"
    )


def _hour_file(path: Path, docs: list[tuple[str, str, str]]) -> Path:
    """docs: (api_name, url, xml) in chronological order."""
    pl.DataFrame(
        {
            "timestamp": list(range(len(docs))),
            "api_name": [d[0] for d in docs],
            "url": [d[1] for d in docs],
            "response_data": [d[2] for d in docs],
        }
    ).write_parquet(path)
    return path


def test_recent_parses_iris_xml(tmp_path: Path) -> None:
    """plan gives the schedule, fchg the realised times; the last fchg wins."""
    raw = tmp_path / "raw"
    raw.mkdir()
    plan_url = f"https://x/timetables/v1/plan/0{ULM}/260706/12"
    fchg_url = f"https://x/timetables/v1/fchg/0{ULM}"
    _hour_file(raw / "hour_00.parquet", [
        ("timetables/v1/plan", plan_url,
         _plan_xml("Ulm Hbf", "4711-2607061200-3", "2607061205", "2607061207")),
        ("timetables/v1/fchg", fchg_url,
         '<timetable station="Ulm Hbf"><s id="4711-2607061200-3">'
         '<ar ct="2607061210"/><dp ct="2607061212"/></s></timetable>'),
        # A later fchg supersedes the earlier one.
        ("timetables/v1/fchg", fchg_url,
         '<timetable station="Ulm Hbf"><s id="4711-2607061200-3">'
         '<ar ct="2607061215"/><dp ct="2607061217"/></s></timetable>'),
    ])

    df = build_recent.process_day(sorted(raw.glob("hour_*.parquet")), None)
    assert df.schema == RECENT_SCHEMA
    row = df.row(0, named=True)
    assert row["eva"] == ULM  # unpadded, taken from the request URL
    assert row["station_name"] == "Ulm Hbf"
    assert (row["train_type"], row["train_number"], row["line_number"]) == ("RE", "4711", "9")
    # The stop id splits into a daily run and a position on the route.
    assert row["train_line_ride_id"] == "4711-2607061200"
    assert row["train_line_station_num"] == 3
    assert row["arrival_planned_time"] == dt.datetime(2026, 7, 6, 12, 5)
    assert row["arrival_change_time"] == dt.datetime(2026, 7, 6, 12, 15)
    assert row["departure_change_time"] == dt.datetime(2026, 7, 6, 12, 17)
    assert row["is_canceled"] is False


def test_recent_without_fchg_counts_as_on_time(tmp_path: Path) -> None:
    """Matches the monthly files' semantics: no change entry means no delay."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _hour_file(raw / "hour_00.parquet", [
        ("timetables/v1/plan", f"https://x/timetables/v1/plan/0{ULM}/260706/12",
         _plan_xml("Ulm Hbf", "4711-2607061200-3", "2607061205", "2607061207")),
    ])
    row = build_recent.process_day(sorted(raw.glob("hour_*.parquet")), None).row(0, named=True)
    assert row["arrival_change_time"] == row["arrival_planned_time"]
    assert row["departure_change_time"] == row["departure_planned_time"]


def test_recent_marks_cancellations(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _hour_file(raw / "hour_00.parquet", [
        ("timetables/v1/plan", f"https://x/timetables/v1/plan/0{ULM}/260706/12",
         _plan_xml("Ulm Hbf", "4711-2607061200-3", "2607061205", "2607061207")),
        ("timetables/v1/fchg", f"https://x/timetables/v1/fchg/0{ULM}",
         '<timetable station="Ulm Hbf"><s id="4711-2607061200-3">'
         '<ar cs="c"/></s></timetable>'),
    ])
    assert build_recent.process_day(sorted(raw.glob("hour_*.parquet")), None).row(0, named=True)[
        "is_canceled"
    ] is True


# Real listings from the archive, before and after it changed naming mid-day on
# 2026-07-26. The old workflow asked for four fixed names and got 404 for each.
LEGACY_LISTING = [
    "hour_00_01_02_03_04_05.parquet",
    "hour_06_07_08_09_10_11.parquet",
    "hour_12_13_14_15_16_17.parquet",
    "hour_18_19_20_21_22_23.parquet",
]
CURRENT_LISTING = [
    "date_2026-08-07_hour_00_01_02_03_04_05.parquet",
    "date_2026-08-07_hour_06_07_08_09.parquet",
    "date_2026-08-07_hour_10_11_12_13_14.parquet",
    "date_2026-08-07_hour_15_16_17_18_19_20.parquet",
    "date_2026-08-07_hour_21_22_23.parquet",
    "date_2026-08-08_hour_00_01.parquet",  # belongs to the next day
]


def test_select_files_current_layout() -> None:
    """Take this day's files, leave the next day's, take the spill-over from
    the previous partition — a day's 00-02 hours are published there."""
    day = dt.date(2026, 8, 7)
    listings = {
        day - dt.timedelta(days=1): [
            "date_2026-08-06_hour_21_22_23.parquet",
            "date_2026-08-07_hour_00_01_02.parquet",
        ],
        day: CURRENT_LISTING,
    }
    chosen = fetch_raw_day.select_files(day, listings)
    assert [name for _, name in chosen] == [
        "date_2026-08-07_hour_00_01_02.parquet",
        "date_2026-08-07_hour_00_01_02_03_04_05.parquet",
        "date_2026-08-07_hour_06_07_08_09.parquet",
        "date_2026-08-07_hour_10_11_12_13_14.parquet",
        "date_2026-08-07_hour_15_16_17_18_19_20.parquet",
        "date_2026-08-07_hour_21_22_23.parquet",
    ]
    # The spill-over is fetched from the previous day's partition, not this one.
    assert dict(chosen)  # names unique across partitions
    assert next(p for p, n in chosen if n.endswith("hour_00_01_02.parquet")) == (
        day - dt.timedelta(days=1)
    )


def test_select_files_legacy_layout() -> None:
    """Days published before the rename must still be fetchable."""
    day = dt.date(2026, 7, 15)
    chosen = fetch_raw_day.select_files(day, {day: LEGACY_LISTING})
    assert [name for _, name in chosen] == LEGACY_LISTING
    assert {part for part, _ in chosen} == {day}


def test_select_files_partial_legacy_day() -> None:
    day = dt.date(2026, 7, 19)
    chosen = fetch_raw_day.select_files(day, {day: LEGACY_LISTING[1:]})
    assert [name for _, name in chosen] == LEGACY_LISTING[1:]


def test_hours_covered_spans_both_layouts() -> None:
    assert fetch_raw_day.hours_covered(LEGACY_LISTING) == set(range(24))
    # The current layout needs the previous partition's spill-over to be whole:
    # a day's own partition starts at hour 03.
    own = [
        "date_2026-07-28_hour_03_04_05_06_07_08_09_10.parquet",
        "date_2026-07-28_hour_11_12_13_14_15.parquet",
        "date_2026-07-28_hour_16_17_18_19_20.parquet",
        "date_2026-07-28_hour_21_22_23.parquet",
    ]
    assert fetch_raw_day.hours_covered(own) == set(range(3, 24))
    spill = ["date_2026-07-28_hour_00_01_02.parquet"]
    assert fetch_raw_day.hours_covered(own + spill) == set(range(24))


def test_fetch_raw_day_waits_for_a_partly_published_day(tmp_path, monkeypatch) -> None:
    """A day is condensed once and cached forever, so half a day must not be
    fetched — it would silently bias every prediction that uses it."""
    day = dt.date(2026, 7, 19)
    monkeypatch.setattr(
        fetch_raw_day, "list_partition",
        lambda d: LEGACY_LISTING[1:] if d == day else [],
    )
    monkeypatch.setattr(fetch_raw_day, "_get", lambda *a, **k: pytest.fail("must not download"))
    monkeypatch.setattr(sys, "argv", [
        "fetch_raw_day.py", "--date", day.isoformat(), "--out-dir", str(tmp_path / "raw"),
    ])
    with pytest.raises(SystemExit) as excinfo:
        fetch_raw_day.main()
    assert excinfo.value.code == fetch_raw_day.NOT_PUBLISHED


def test_select_files_reports_an_unknown_layout() -> None:
    """The next rename must surface as a red job, not as a silently skipped day."""
    day = dt.date(2026, 9, 1)
    assert fetch_raw_day.select_files(day, {day: ["chunk-0.parquet", "chunk-1.parquet"]}) == []


def test_fetch_raw_day_exits_2_when_unpublished(tmp_path, monkeypatch) -> None:
    """Distinguishing "not yet" from "changed shape" is the whole point."""
    monkeypatch.setattr(fetch_raw_day, "list_partition", lambda day: [])
    monkeypatch.setattr(sys, "argv", [
        "fetch_raw_day.py", "--date", "2026-08-07", "--out-dir", str(tmp_path / "raw"),
    ])
    with pytest.raises(SystemExit) as excinfo:
        fetch_raw_day.main()
    assert excinfo.value.code == fetch_raw_day.NOT_PUBLISHED


def test_fetch_raw_day_fails_loudly_on_unknown_layout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fetch_raw_day, "list_partition", lambda day: ["chunk-0.parquet"])
    monkeypatch.setattr(sys, "argv", [
        "fetch_raw_day.py", "--date", "2026-08-07", "--out-dir", str(tmp_path / "raw"),
    ])
    with pytest.raises(SystemExit) as excinfo:
        fetch_raw_day.main()
    assert excinfo.value.code != 0
    assert excinfo.value.code != fetch_raw_day.NOT_PUBLISHED
    assert "naming changed" in str(excinfo.value)


def test_fetch_raw_day_downloads_under_upstream_names(tmp_path, monkeypatch) -> None:
    day = dt.date(2026, 8, 7)
    monkeypatch.setattr(
        fetch_raw_day, "list_partition",
        lambda d: CURRENT_LISTING if d == day else [],
    )
    fetched: list[str] = []

    def fake_get(url: str, tries: int = 3) -> bytes:
        fetched.append(url)
        return b"payload"

    monkeypatch.setattr(fetch_raw_day, "_get", fake_get)
    out = tmp_path / "raw"
    monkeypatch.setattr(sys, "argv", [
        "fetch_raw_day.py", "--date", day.isoformat(), "--out-dir", str(out),
    ])
    fetch_raw_day.main()

    names = sorted(f.name for f in out.glob("*.parquet"))
    assert names == sorted(n for n in CURRENT_LISTING if "2026-08-07" in n)
    # Unpadded month/day in the partition path, as the archive publishes it.
    assert all("year=2026/month=8/day=7/" in url for url in fetched)


def test_recent_reads_date_prefixed_files(tmp_path, monkeypatch) -> None:
    """build_recent must accept whatever fetch_raw_day.py saved, under any name."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _hour_file(raw / "date_2026-07-06_hour_10_11_12.parquet", [
        ("timetables/v1/plan", f"https://x/timetables/v1/plan/0{ULM}/260706/12",
         _plan_xml("Ulm Hbf", "4711-2607061200-3", "2607061205", "2607061207")),
    ])
    out = tmp_path / "day.parquet"
    monkeypatch.setattr(sys, "argv", [
        "build_recent.py", "--date", "2026-07-06",
        "--raw-dir", str(raw), "--out", str(out),
    ])
    build_recent.main()
    assert pl.read_parquet(out).height == 1


def test_recent_station_filter_keeps_the_whole_run(tmp_path: Path) -> None:
    """Filtering by station must keep the run's other stops, or the prev-stop
    feature loses exactly the value it exists for."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _hour_file(raw / "hour_00.parquet", [
        ("timetables/v1/plan", f"https://x/timetables/v1/plan/0{ULM}/260706/12",
         _plan_xml("Ulm Hbf", "4711-2607061200-3", "2607061205", "2607061207")),
        ("timetables/v1/plan", f"https://x/timetables/v1/plan/0{AUGSBURG}/260706/12",
         _plan_xml("Augsburg Hbf", "4711-2607061200-4", "2607061245", "2607061247")),
        ("timetables/v1/plan", f"https://x/timetables/v1/plan/0{AUGSBURG}/260706/12",
         _plan_xml("Augsburg Hbf", "9999-2607061200-1", "2607061250", "2607061252")),
    ])
    df = build_recent.process_day(sorted(raw.glob("hour_*.parquet")), [ULM])
    assert sorted(df["train_line_station_num"]) == [3, 4]
    assert set(df["train_line_ride_id"]) == {"4711-2607061200"}


# --- backtest: the knobs the model review used -------------------------------
#
# These exist to answer "is the shipped distribution the right shape". Two of
# the three answers turned out to be no, which is only worth anything if the
# machinery that produced them is right — an experiment that silently measures
# something other than what it claims is worse than no experiment. The trap hit
# during the review: `sharpen` compressed one side while a sibling variant
# compressed both, so the two were not comparable and the decomposition did not
# add up.

def _bt():
    """backtest.py, imported the way the other pipeline modules are.

    Not at module scope: it pulls in numpy and the holiday table, and the
    fixtures above have no need of either.
    """
    import backtest
    return backtest


def test_sharpening_each_side_is_independent() -> None:
    bt = _bt()
    import numpy as np
    x = np.array([-4.0, 0.0, 4.0])
    w = np.array([1.0, 1.0, 1.0])
    high = bt.Variant("h", sharpen_high=0.5)
    low = bt.Variant("l", sharpen_low=0.5)
    assert list(bt.sharpen(high, x, w, False)) == [-4.0, 0.0, 2.0]
    assert list(bt.sharpen(low, x, w, False)) == [-2.0, 0.0, 4.0]


def test_sharpening_leaves_the_support_alone_by_default() -> None:
    bt = _bt()
    import numpy as np
    x = np.array([-4.0, 0.0, 9.0])
    assert list(bt.sharpen(bt.Variant("n"), x, np.ones(3), True)) == list(x)


def test_sharpening_can_be_limited_to_one_of_the_two_distributions() -> None:
    """The decomposition the review turned on: applying a change to the
    conditional distribution, the unconditional one, or both, gives three
    different answers, and on this data they had opposite signs."""
    bt = _bt()
    import numpy as np
    x, w = np.array([0.0, 10.0]), np.ones(2)
    only_cond = bt.Variant("c", sharpen_high=0.5, sharpen_when="conditioned")
    only_uncond = bt.Variant("u", sharpen_high=0.5, sharpen_when="unconditioned")
    assert list(bt.sharpen(only_cond, x, w, True)) == [0.0, 5.0]
    assert list(bt.sharpen(only_cond, x, w, False)) == [0.0, 10.0]
    assert list(bt.sharpen(only_uncond, x, w, False)) == [0.0, 5.0]
    assert list(bt.sharpen(only_uncond, x, w, True)) == [0.0, 10.0]


def test_shrinkage_leans_on_the_population_only_when_history_is_thin() -> None:
    bt = _bt()
    import numpy as np
    pooled = (np.array([100.0]), np.array([1.0]))
    thin = bt.Variant("t", shrink_kappa=4.0)
    x, w = np.array([0.0]), np.array([1.0])          # one run: N_eff = 1
    _, weights = bt.shrink_to_pooled(thin, x, w, pooled)
    assert weights[-1] == pytest.approx(0.8)          # 1 - 1/(1+4)
    x, w = np.zeros(16), np.ones(16)                  # sixteen: N_eff = 16
    _, weights = bt.shrink_to_pooled(thin, x, w, pooled)
    assert weights[-1] == pytest.approx(0.2)          # 1 - 16/(16+4)


def test_shrinkage_returns_a_distribution_whatever_went_in() -> None:
    bt = _bt()
    import numpy as np
    pooled = (np.array([5.0, 6.0]), np.array([0.5, 0.5]))
    x, w = np.array([0.0, 1.0]), np.array([7.0, 3.0])   # unnormalised
    _, weights = bt.shrink_to_pooled(bt.Variant("s", shrink_kappa=2.0), x, w, pooled)
    assert weights.sum() == pytest.approx(1.0)


def test_shrinkage_is_off_unless_asked_for() -> None:
    bt = _bt()
    import numpy as np
    x, w = np.array([0.0]), np.array([1.0])
    got_x, got_w = bt.shrink_to_pooled(bt.Variant("n"), x, w,
                                       (np.array([9.0]), np.array([1.0])))
    assert list(got_x) == [0.0] and list(got_w) == [1.0]


def test_the_predictive_says_whether_it_used_the_live_report() -> None:
    """Without the flag a change cannot be aimed at one of the two
    distributions, and aiming it wrongly is what made the first decomposition
    disagree with itself."""
    bt = _bt()
    import numpy as np
    variant = bt.Variant("d", live_bandwidth=0.3, live_mode="delta")
    delay = np.arange(20.0)
    prev = np.arange(20.0)
    w = np.ones(20)
    _, _, used = bt.predictive_points(variant, delay, prev, w, 5.0)
    assert used is True
    _, _, used = bt.predictive_points(variant, delay, prev, w, None)
    assert used is False, "no report, so nothing was conditioned on"
    _, _, used = bt.predictive_points(variant, delay[:3], prev[:3], w[:3], 5.0)
    assert used is False, "too few runs with a known previous stop"
