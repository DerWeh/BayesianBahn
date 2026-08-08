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


def _run(module, monkeypatch, data_dir: Path, out_dir: Path, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", [
        module.__name__, "--data-dir", str(data_dir), "--out-dir", str(out_dir), *extra,
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


def test_workflow_pins_the_polars_these_tests_ran_against() -> None:
    """Otherwise these tests vouch for a version the nightly job never installs."""
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/update-data.yml"
    ).read_text()
    pinned = re.search(r"polars==([0-9][0-9a-z.]*)", workflow)
    assert pinned, "update-data.yml must pin polars to an exact version"
    assert pinned.group(1) == pl.__version__, (
        f"update-data.yml pins polars {pinned.group(1)} but the tests run "
        f"{pl.__version__}; bump pixi.toml/pixi.lock and the workflow together"
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
