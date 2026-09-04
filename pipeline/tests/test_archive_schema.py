"""Every archive reader must speak both spellings of the cancellation column.

Upstream replaced `is_canceled` with `arrival_is_canceled` and
`departure_is_canceled` on 2026-09-03, in every monthly file rather than only in
new ones, and the nightly rebuild failed on months it had processed for a year.
Nothing here caught it because every fixture in the suite wrote the old
spelling, so the whole suite agreed with the code about a fact neither of them
owned.

So the readers are listed by name below and each is run twice over the same
data, once in each spelling, and required to produce the same answer. A fifth
reader added without being added here is a visible omission rather than a silent
one.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))

import archive  # noqa: E402
import backtest  # noqa: E402
import build_shards  # noqa: E402
import route_bench as rb  # noqa: E402
import score_events as se  # noqa: E402

DAY = dt.date(2026, 8, 17)
ULM = "8000170"

SCHEMA = {
    "station_name": pl.String,
    "eva": pl.String,
    "train_type": pl.String,
    "train_number": pl.String,
    "line_number": pl.String,
    "train_line_ride_id": pl.String,
    "train_line_station_num": pl.Int32,
    "arrival_planned_time": pl.Datetime("us"),
    "arrival_change_time": pl.Datetime("us"),
    "departure_planned_time": pl.Datetime("us"),
    "departure_change_time": pl.Datetime("us"),
    "is_canceled": pl.Boolean,
}


def rows() -> list[dict]:
    """A day of one train at one station, one run of it cancelled."""
    out = []
    for seq, minute in enumerate((0, 30)):
        arrives = dt.datetime.combine(DAY, dt.time(8)) + dt.timedelta(minutes=minute)
        departs = arrives + dt.timedelta(minutes=2)
        out.append({
            "station_name": "Ulm Hbf", "eva": f"0{ULM}", "train_type": "RE",
            "train_number": "42", "line_number": "9",
            "train_line_ride_id": "ride-a", "train_line_station_num": seq,
            "arrival_planned_time": arrives,
            "arrival_change_time": arrives + dt.timedelta(minutes=4),
            "departure_planned_time": departs,
            "departure_change_time": departs + dt.timedelta(minutes=4),
            "is_canceled": False,
        })
    out.append({**out[0], "train_line_ride_id": "ride-b", "train_number": "43",
                "is_canceled": True})
    return out


def write(directory: Path, spelling: str, name: str = "data-2026-08.parquet") -> Path:
    """The same data, written the way the archive wrote it before or after."""
    df = pl.DataFrame(rows(), schema=SCHEMA)
    if spelling == "split":
        df = df.drop("is_canceled").with_columns(
            arrival_is_canceled=pl.Series(r["is_canceled"] for r in rows()),
            departure_is_canceled=pl.Series(r["is_canceled"] for r in rows()))
    directory.mkdir(parents=True, exist_ok=True)
    df.write_parquet(directory / name)
    return directory / name


SPELLINGS = ["single", "split"]


# --- the module itself -------------------------------------------------------

def test_the_old_column_stands_for_both_ends(tmp_path):
    """It was the disjunction, so a reader of it meant both and could tell neither."""
    path = write(tmp_path, "single")
    arrival, departure = archive.cancellation(path)
    got = pl.scan_parquet(path).select(a=arrival, d=departure).collect()
    assert got["a"].to_list() == got["d"].to_list() == [False, False, True]


def test_the_split_columns_are_read_apart(tmp_path):
    path = write(tmp_path, "split")
    arrival, departure = archive.cancellation(path)
    frame = pl.scan_parquet(path).with_columns(
        arrival_is_canceled=pl.Series([True, False, False]),
        departure_is_canceled=pl.Series([False, True, False]))
    got = frame.select(a=arrival, d=departure, e=archive.either(path)).collect()
    assert got["a"].to_list() == [True, False, False]
    assert got["d"].to_list() == [False, True, False]
    assert got["e"].to_list() == [True, True, False], "either means either end"


def test_a_file_with_neither_spelling_says_so(tmp_path):
    path = tmp_path / "odd.parquet"
    pl.DataFrame({"eva": ["1"]}).write_parquet(path)
    with pytest.raises(SystemExit) as raised:
        archive.cancellation(path)
    assert "is_canceled" in str(raised.value)
    assert "arrival_is_canceled" in str(raised.value)


def test_require_names_the_column_the_file_and_what_it_does_have(tmp_path):
    """The message is the point: the real failure named a frame, not a column."""
    path = write(tmp_path, "single")
    with pytest.raises(SystemExit) as raised:
        archive.require(path, ["eva", "platform_of_departure"])
    message = str(raised.value)
    assert "platform_of_departure" in message
    assert "eva" not in message.split("missing")[1].split(";")[0]
    assert str(path) in message
    assert "archive.py" in message, "the message should say where to fix it"


# --- the readers -------------------------------------------------------------

@pytest.mark.parametrize("spelling", SPELLINGS)
def test_build_shards_reads_either_spelling(tmp_path, spelling):
    path = write(tmp_path / spelling, spelling)
    df = build_shards.prepare_month(path, {ULM}, None, "identity")
    assert df.height == 3
    assert df["is_canceled"].sum() == 1


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_backtest_reads_either_spelling(tmp_path, spelling):
    path = write(tmp_path / spelling, spelling)
    df = backtest.load_month(path, [ULM])
    # The cancelled run is dropped; the two stops of the other survive.
    assert df.height == 2
    assert set(df["train_number"]) == {"42"}


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_route_bench_reads_either_spelling(tmp_path, spelling):
    directory = tmp_path / spelling
    write(directory, spelling)
    out = tmp_path / f"{spelling}.parquet"
    rb.snapshot([directory], DAY, out)
    got = pl.read_parquet(out)
    assert got.height == 2, "the cancelled ride is excluded"
    assert set(got["eva"]) == {ULM}


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_score_events_reads_either_spelling(tmp_path, spelling):
    directory = tmp_path / spelling
    write(directory, spelling)
    truth = se.load_truth([directory], DAY, {ULM})
    planned = [k for k in truth if k[0] == ULM and k[2] == "42"]
    assert planned, "the uncancelled train must join"
    assert all(not truth[k]["cancelled"] for k in planned)
    assert any(v["cancelled"] for v in truth.values()), "and the cancelled one must too"


# --- the two spellings must agree --------------------------------------------

def test_the_readers_do_not_merely_run_but_agree(tmp_path):
    """Same data, both spellings, identical answers — not just no exception."""
    single = build_shards.prepare_month(write(tmp_path / "a", "single"),
                                        {ULM}, None, "identity")
    split = build_shards.prepare_month(write(tmp_path / "b", "split"),
                                       {ULM}, None, "identity")
    assert single.sort("train_number", "tod").equals(split.sort("train_number", "tod"))


def test_a_run_holding_both_spellings_at_once_still_reads(tmp_path):
    """The realistic case: monthly files renamed upstream, our own cache not.

    `build_recent.py` writes the retired spelling and its output is cached
    across runs, so a rebuild routinely concatenates one of each. That is what
    broke `route_bench.snapshot`, which concatenated before selecting.
    """
    directory = tmp_path / "mixed"
    write(directory, "split", "data-2026-08.parquet")
    write(directory, "single", "data-recent-2026-08-17.parquet")
    out = tmp_path / "mixed.parquet"
    rb.snapshot([directory], DAY, out)
    assert pl.read_parquet(out).height == 2, "deduplicated across the two files"
