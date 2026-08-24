"""Tests for the event builder (Phase A of the DB comparison).

The two errors that would quietly invalidate the whole comparison are both
here. First, taking events from what was observed rather than from the plan:
`fchg` reports a stop in four shapes and three of them mean "on time", so an
observation-driven universe keeps the trains DB flagged and drops the ones it
got right. Second, the clock boundary — journal timestamps are real epoch
seconds, IRIS times are wall clock stored as if UTC, and comparing them
directly shifts every lead time by the German UTC offset.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import collect_forecasts as cf  # noqa: E402
import score_events as se  # noqa: E402

ROOT_APP = TOOLS.parent / "app/src/main/java/io/github/derweh/bayesianbahn"

DAY = dt.date(2026, 8, 17)
# 18:00 German wall clock on that day, in both bases.
PLANNED = cf.iris_time("2608171800")
EPOCH_1800 = se.wall_to_epoch(PLANNED)


def write(tmp_path: Path, records: list[dict]) -> Path:
    journal = cf.Journal(tmp_path / f"forecasts-{DAY}.jsonl")
    for r in records:
        journal.append(r)
    journal.close()
    return tmp_path


def plan_rec(at, *, trip="t1", eva="8000001", par=PLANNED, cat="RE", num="1"):
    return {"t": "plan", "at": int(at), "eva": eva, "id": trip, "cat": cat,
            "num": num, "line": None, "par": par, "pdp": None, "ppth": []}


def obs_rec(at, ar, *, trip="t1", eva="8000001", arc=False):
    return {"t": "obs", "at": int(at), "eva": eva, "id": trip,
            "ar": ar, "dp": None, "arc": arc, "dpc": False}


def poll_rec(at, *, eva="8000001", ok=True):
    return {"t": "poll", "at": int(at), "eva": eva, "ok": ok, "stops": 1}


# --- the clock boundary ------------------------------------------------------


def test_wall_clock_converts_to_the_right_instant_in_summer() -> None:
    """18:00 German summer time is 16:00 UTC; a naive comparison loses 2 hours."""
    got = dt.datetime.fromtimestamp(se.wall_to_epoch(cf.iris_time("2608171800")),
                                    dt.timezone.utc)
    assert (got.hour, got.minute) == (16, 0)


def test_wall_clock_converts_to_the_right_instant_in_winter() -> None:
    got = dt.datetime.fromtimestamp(se.wall_to_epoch(cf.iris_time("2601171800")),
                                    dt.timezone.utc)
    assert (got.hour, got.minute) == (17, 0), "one hour offset outside DST"


# --- the event universe ------------------------------------------------------


def test_a_train_db_never_mentions_still_produces_events(tmp_path: Path) -> None:
    """The bug this file exists to prevent: absence from fchg means on time,
    and those are exactly the trains DB gets right."""
    out = write(tmp_path, [poll_rec(EPOCH_1800 - 7200), plan_rec(EPOCH_1800 - 7200)])
    stops, polls = se.read_day(out, DAY)
    events = se.build_events(stops, polls, horizons=(60,))
    assert len(events) == 1
    assert events[0]["db"] == 0 and events[0]["db_explicit"] is False


def test_an_entry_without_a_changed_time_means_on_time(tmp_path: Path) -> None:
    """Seen live: EUR 9459 had an fchg entry carrying messages but no `ct`."""
    out = write(tmp_path, [poll_rec(EPOCH_1800 - 7200), plan_rec(EPOCH_1800 - 7200),
                           obs_rec(EPOCH_1800 - 7200, None)])
    stops, polls = se.read_day(out, DAY)
    events = se.build_events(stops, polls, horizons=(60,))
    assert events[0]["db"] == 0, "no ct is not missing data"


def test_a_changed_time_becomes_a_delay(tmp_path: Path) -> None:
    out = write(tmp_path, [poll_rec(EPOCH_1800 - 7200), plan_rec(EPOCH_1800 - 7200),
                           obs_rec(EPOCH_1800 - 7200, PLANNED + 33)])
    stops, polls = se.read_day(out, DAY)
    assert se.build_events(stops, polls, horizons=(60,))[0]["db"] == 33


def test_a_departure_only_stop_is_not_an_arrival_event(tmp_path: Path) -> None:
    out = write(tmp_path, [poll_rec(EPOCH_1800), plan_rec(EPOCH_1800, par=None)])
    stops, _ = se.read_day(out, DAY)
    assert stops == {}


# --- the trajectory ----------------------------------------------------------


def test_the_forecast_is_the_last_one_before_the_lead_time(tmp_path: Path) -> None:
    """DB revises; an event must use what was knowable then, not afterwards."""
    out = write(tmp_path, [
        poll_rec(EPOCH_1800 - 7200), plan_rec(EPOCH_1800 - 7200),
        obs_rec(EPOCH_1800 - 7200, PLANNED + 2),      # 2h before: +2
        obs_rec(EPOCH_1800 - 1800, PLANNED + 20),     # 30m before: +20
    ])
    stops, polls = se.read_day(out, DAY)
    got = {e["tau"]: e["db"] for e in se.build_events(stops, polls, horizons=(60, 15))}
    assert got == {60: 2, 15: 20}


def test_events_are_skipped_where_we_were_not_yet_collecting(tmp_path: Path) -> None:
    """A gap in our collection must not be recorded as DB predicting on time."""
    out = write(tmp_path, [poll_rec(EPOCH_1800 - 600), plan_rec(EPOCH_1800 - 600)])
    stops, polls = se.read_day(out, DAY)
    taus = {e["tau"] for e in se.build_events(stops, polls, horizons=(240, 60, 5))}
    assert taus == {5}, "only the lead time we were actually watching"


def test_a_failed_poll_does_not_count_as_coverage(tmp_path: Path) -> None:
    out = write(tmp_path, [poll_rec(EPOCH_1800 - 7200, ok=False),
                           plan_rec(EPOCH_1800 - 7200)])
    stops, polls = se.read_day(out, DAY)
    assert se.build_events(stops, polls, horizons=(60,)) == []


def test_settled_truth_is_taken_well_after_the_train_was_due(tmp_path: Path) -> None:
    late = EPOCH_1800 + 60 * (20 + se.SETTLED_AFTER) + 600
    out = write(tmp_path, [
        poll_rec(EPOCH_1800 - 7200), plan_rec(EPOCH_1800 - 7200),
        obs_rec(EPOCH_1800 - 1800, PLANNED + 20),
        obs_rec(EPOCH_1800 + 60 * 25, PLANNED + 26),
        poll_rec(late),
    ])
    stops, polls = se.read_day(out, DAY)
    assert list(stops.values())[0].settled(polls["8000001"][-1]) == 26


def test_a_forecast_still_in_flight_has_no_truth_yet(tmp_path: Path) -> None:
    """Otherwise "truth" is the very forecast being scored, and DB is compared
    against itself — which is what made the first Phase A run report MAE 0.0."""
    out = write(tmp_path, [
        poll_rec(EPOCH_1800 - 3600), plan_rec(EPOCH_1800 - 3600),
        obs_rec(EPOCH_1800 - 1800, PLANNED + 20),
        poll_rec(EPOCH_1800 - 300),
    ])
    stops, polls = se.read_day(out, DAY)
    assert list(stops.values())[0].settled(polls["8000001"][-1]) is None


def test_settling_waits_for_the_delay_not_just_the_timetable(tmp_path: Path) -> None:
    """A train forecast 40 min late has not arrived 30 min after its plan."""
    out = write(tmp_path, [
        poll_rec(EPOCH_1800 - 3600), plan_rec(EPOCH_1800 - 3600),
        obs_rec(EPOCH_1800 - 1800, PLANNED + 40),
        poll_rec(EPOCH_1800 + 60 * 35),
    ])
    stops, polls = se.read_day(out, DAY)
    assert list(stops.values())[0].settled(polls["8000001"][-1]) is None


def test_a_cancellation_is_flagged_not_scored_as_a_delay(tmp_path: Path) -> None:
    out = write(tmp_path, [
        poll_rec(EPOCH_1800 - 7200), plan_rec(EPOCH_1800 - 7200),
        obs_rec(EPOCH_1800 - 3600, PLANNED, arc=True),
    ])
    stops, polls = se.read_day(out, DAY)
    assert se.build_events(stops, polls, horizons=(30,))[0]["cancelled"] is True


def test_a_stop_never_polled_after_arrival_yields_no_truth(tmp_path: Path) -> None:
    out = write(tmp_path, [poll_rec(EPOCH_1800 - 7200), plan_rec(EPOCH_1800 - 7200)])
    stops, polls = se.read_day(out, DAY)
    assert all(e["settled"] is None
               for e in se.build_events(stops, polls, horizons=(60,)))


def test_an_observation_with_no_plan_record_is_dropped(tmp_path: Path) -> None:
    """Stops beyond the plan horizon have no planned time, so no delay."""
    out = write(tmp_path, [poll_rec(EPOCH_1800), obs_rec(EPOCH_1800, PLANNED + 5)])
    stops, _ = se.read_day(out, DAY)
    assert stops == {}


def test_the_recorded_lead_time_is_the_real_one_not_the_nominal(tmp_path: Path) -> None:
    """With a 10-minute cadence a "5 minute" forecast is 5-15 minutes old, and
    labelling it 5 would move the crossover point we are trying to find."""
    out = write(tmp_path, [
        poll_rec(EPOCH_1800 - 3600), plan_rec(EPOCH_1800 - 3600),
        poll_rec(EPOCH_1800 - 780),          # 13 minutes before arrival
        obs_rec(EPOCH_1800 - 780, PLANNED + 4),
    ])
    stops, polls = se.read_day(out, DAY)
    event = se.build_events(stops, polls, horizons=(5,))[0]
    assert event["tau"] == 5
    assert event["lead"] == 13.0, "the freshest reading we hold is 13 min old"


def test_lead_time_runs_from_the_poll_even_with_no_observation(tmp_path: Path) -> None:
    """A stop absent from fchg says "on time as of that poll", not as of now."""
    out = write(tmp_path, [poll_rec(EPOCH_1800 - 3600), plan_rec(EPOCH_1800 - 3600),
                           poll_rec(EPOCH_1800 - 900)])
    stops, polls = se.read_day(out, DAY)
    event = se.build_events(stops, polls, horizons=(5,))[0]
    assert event["db"] == 0 and event["lead"] == 15.0


# --- the archive join --------------------------------------------------------


def archive_file(path: Path, rows: list[dict]) -> Path:
    import polars as pl
    schema = {"eva": pl.String, "train_type": pl.String, "train_number": pl.String,
              "arrival_planned_time": pl.Datetime("ns"),
              "arrival_change_time": pl.Datetime("ns"),
              "departure_planned_time": pl.Datetime("ns"),
              "departure_change_time": pl.Datetime("ns"),
              "is_canceled": pl.Boolean}
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema).write_parquet(path / "data-2026-08.parquet")
    return path


def arch_row(minute_delay, *, eva="08000001", cat="RE", num="1", cancelled=False,
             dep_delay=None):
    planned = dt.datetime(2026, 8, 17, 18, 0)
    departs = planned + dt.timedelta(minutes=2)
    shift = dep_delay if dep_delay is not None else minute_delay
    return {"eva": eva, "train_type": cat, "train_number": num,
            "arrival_planned_time": planned,
            "arrival_change_time": (planned + dt.timedelta(minutes=minute_delay)
                                    if minute_delay is not None else None),
            "departure_planned_time": departs,
            "departure_change_time": (departs + dt.timedelta(minutes=shift)
                                      if shift is not None else None),
            "is_canceled": cancelled}


def test_archive_truth_joins_on_the_journal_key(tmp_path: Path) -> None:
    data = archive_file(tmp_path / "arch", [arch_row(7)])
    truth = se.load_truth([data], DAY, {"8000001"})
    assert truth[("8000001", "RE", "1", PLANNED)]["delay"] == 7


def test_archive_padding_does_not_break_the_join(tmp_path: Path) -> None:
    """The archive zero-pads EVAs, IRIS does not."""
    data = archive_file(tmp_path / "arch", [arch_row(3, eva="08000001")])
    assert ("8000001", "RE", "1", PLANNED) in se.load_truth([data], DAY, {"8000001"})


def test_no_change_time_in_the_archive_means_on_time(tmp_path: Path) -> None:
    data = archive_file(tmp_path / "arch", [arch_row(None)])
    assert se.load_truth([data], DAY, {"8000001"})[
        ("8000001", "RE", "1", PLANNED)]["delay"] == 0


def test_join_rate_is_reported_per_train_type() -> None:
    """A whole operator failing to join would vanish from the comparison
    instead of failing it — as TR nearly did against a two-month-old archive."""
    events = [{"eva": "8000001", "cat": "RE", "num": "1", "planned": PLANNED,
               "cancelled": False},
              {"eva": "8000310", "cat": "TR", "num": "33622", "planned": PLANNED,
               "cancelled": False}]
    truth = {("8000001", "RE", "1", PLANNED):
             {"delay": 4, "dep_delay": 4, "cancelled": False}}
    rate = se.attach_truth(events, truth)
    assert rate == {"RE": (1, 1), "TR": (0, 1)}
    assert events[0]["archive"] == 4 and "archive" not in events[1]


def test_a_cancellation_in_the_archive_marks_the_event(tmp_path: Path) -> None:
    data = archive_file(tmp_path / "arch", [arch_row(None, cancelled=True)])
    events = [{"eva": "8000001", "cat": "RE", "num": "1", "planned": PLANNED,
               "cancelled": False}]
    se.attach_truth(events, se.load_truth([data], DAY, {"8000001"}))
    assert events[0]["cancelled"] is True


def test_a_day_the_archive_does_not_cover_fails_loudly(tmp_path: Path) -> None:
    import pytest
    (tmp_path / "arch").mkdir()
    with pytest.raises(SystemExit, match="no archive file"):
        se.load_truth([tmp_path / "arch"], DAY, {"8000001"})


# --- shard keys --------------------------------------------------------------


def test_shard_keys_mirror_the_repository() -> None:
    """The harness reads shards by the key the app derives; a divergence would
    silently give the model no history and score the prior instead."""
    import fetch_shards as fs
    source = (ROOT_APP / "data/HistoryRepository.kt").read_text(encoding="utf-8")
    assert 'replace(Regex("[^A-Za-z0-9]+"), "_").trim(\'_\').uppercase()' in source, (
        "shardKey changed; the Python mirror in fetch_shards.py must follow"
    )
    assert fs.shard_key("ICE 512") == "ICE_512"
    assert fs.shard_key("  RE  9  ") == "RE_9"
    assert fs.shard_key("RB26") == "RB26"


def test_candidate_keys_try_the_number_then_the_line() -> None:
    import fetch_shards as fs
    assert fs.candidate_keys("RE", "10924", None) == ["RE_10924"]
    assert fs.candidate_keys("RB", "25441", "RB26") == ["RB_25441", "RB26"]
    # A line not already prefixed by the category gets it prepended.
    assert fs.candidate_keys("S", "42687", "1") == ["S_42687", "S_1"]
    assert fs.candidate_keys("RE", "", "RE9") == ["RE9"]


# --- the binning matrix ------------------------------------------------------


def test_the_actual_anchor_moves_a_late_train_to_a_longer_lead() -> None:
    """A train 30 min late has its "5 min before planned arrival" reading taken
    35 min before it actually arrives; binning by the plan hides that."""
    event = {"planned": PLANNED, "planned_dep": PLANNED + 2, "archive": 30,
             "archive_dep": 30}
    assert se.anchor_minutes(event, "planned_arrival") == PLANNED
    assert se.anchor_minutes(event, "actual_arrival") == PLANNED + 30
    assert se.anchor_minutes(event, "planned_departure") == PLANNED + 2
    assert se.anchor_minutes(event, "actual_departure") == PLANNED + 32


def test_an_anchor_without_the_data_it_needs_is_skipped() -> None:
    """A terminus has no departure, and an unjoined event has no actual time."""
    assert se.anchor_minutes({"planned": PLANNED, "planned_dep": None},
                             "planned_departure") is None
    assert se.anchor_minutes({"planned": PLANNED}, "actual_arrival") is None


def test_buckets_cover_the_lead_times_without_gaps() -> None:
    assert se.bucket_of(0) == "<10m"
    assert se.bucket_of(9.9) == "<10m"
    assert se.bucket_of(10) == "10-20m"
    assert se.bucket_of(44) == "20-45m"
    assert se.bucket_of(200) == ">3h"
    assert se.bucket_of(-1) is None, "a reading taken after the event is not a lead"


def test_departure_delay_is_read_from_the_archive(tmp_path: Path) -> None:
    data = archive_file(tmp_path / "arch", [arch_row(7, dep_delay=9)])
    got = se.load_truth([data], DAY, {"8000001"})[("8000001", "RE", "1", PLANNED)]
    assert got["delay"] == 7 and got["dep_delay"] == 9


# --- one-change journeys -----------------------------------------------------


def conn_plan(at, *, trip, par=None, pdp=None, num="1", cat="RE", eva="8000001"):
    return {"t": "plan", "at": int(at), "eva": eva, "id": trip, "cat": cat,
            "num": num, "line": None, "par": par, "pdp": pdp, "ppth": []}


def test_trip_start_comes_from_the_trip_id() -> None:
    stop = se.Stop(eva="1", trip="-297098023597558982-2608171611-32", cat="RE",
                   num="1", line=None, planned=PLANNED)
    assert stop.trip_start() == cf.iris_time("2608171611")


def test_a_trip_id_without_a_timestamp_has_no_start() -> None:
    stop = se.Stop(eva="1", trip="odd-id", cat="RE", num="1", line=None,
                   planned=PLANNED)
    assert stop.trip_start() is None


def connection_day(tmp_path: Path, *, conn_dep_offset: int, feeder_delay: int,
                   conn_delay: int):
    """A feeder arriving at 18:00 and a connecting train leaving soon after."""
    start = "2608171700"            # the feeder set off at 17:00
    feeder = f"-1-{start}-4"
    onward = "-2-2608171730-9"
    boarded = se.wall_to_epoch(cf.iris_time(start))
    out = write(tmp_path, [
        poll_rec(boarded - 3600),
        conn_plan(boarded - 3600, trip=feeder, par=PLANNED, pdp=PLANNED + 1),
        conn_plan(boarded - 3600, trip=onward, pdp=PLANNED + conn_dep_offset, num="2"),
        obs_rec(boarded - 3600, PLANNED, trip=feeder),
    ])
    stops, polls = se.read_day(out, DAY)
    truth = {
        ("8000001", "RE", "1", PLANNED):
            {"delay": feeder_delay, "dep_delay": feeder_delay, "cancelled": False},
        ("dep", "8000001", "RE", "2", PLANNED + conn_dep_offset):
            {"delay": conn_delay, "dep_delay": conn_delay, "cancelled": False},
    }
    return se.build_connections(stops, polls, truth)


def test_a_connection_is_caught_when_the_feeder_arrives_in_time(tmp_path: Path) -> None:
    got = connection_day(tmp_path, conn_dep_offset=15, feeder_delay=0, conn_delay=0)
    assert len(got) == 1
    assert got[0]["caught"] is True
    assert got[0]["slack"] == 15 - se.TRANSFER_MINUTES


def test_a_late_feeder_misses_it(tmp_path: Path) -> None:
    got = connection_day(tmp_path, conn_dep_offset=15, feeder_delay=20, conn_delay=0)
    assert got[0]["caught"] is False


def test_a_connection_that_waits_is_still_caught(tmp_path: Path) -> None:
    """The change works if the onward train is late too — which is exactly the
    case a yes/no answer computed from the timetable gets wrong."""
    got = connection_day(tmp_path, conn_dep_offset=15, feeder_delay=20,
                         conn_delay=25)
    assert got[0]["caught"] is True


def test_too_little_slack_is_not_a_connection_anyone_would_plan(tmp_path: Path) -> None:
    assert connection_day(tmp_path, conn_dep_offset=6, feeder_delay=0,
                          conn_delay=0) == []


def test_too_much_slack_is_not_worth_scoring(tmp_path: Path) -> None:
    assert connection_day(tmp_path, conn_dep_offset=120, feeder_delay=0,
                          conn_delay=0) == []


def test_the_reading_is_taken_before_the_feeder_set_off(tmp_path: Path) -> None:
    """Once aboard the passenger has committed, so a later reading is useless."""
    got = connection_day(tmp_path, conn_dep_offset=15, feeder_delay=0, conn_delay=0)
    start = se.wall_to_epoch(cf.iris_time("2608171700"))
    assert got[0]["read_at"] <= start


def test_a_train_is_not_a_connection_to_itself(tmp_path: Path) -> None:
    boarded = se.wall_to_epoch(cf.iris_time("2608171700"))
    trip = "-1-2608171700-4"
    out = write(tmp_path, [
        poll_rec(boarded - 3600),
        conn_plan(boarded - 3600, trip=trip, par=PLANNED, pdp=PLANNED + 10),
    ])
    stops, polls = se.read_day(out, DAY)
    truth = {("8000001", "RE", "1", PLANNED):
             {"delay": 0, "dep_delay": 0, "cancelled": False},
             ("dep", "8000001", "RE", "1", PLANNED + 10):
             {"delay": 0, "dep_delay": 0, "cancelled": False}}
    assert se.build_connections(stops, polls, truth) == []


# --- the second tier must not become a second set of origins -----------------
#
# The collector polls the far ends of changes so DB's forecast for the second
# leg can be read. Those stations were picked after the data started arriving,
# from what the timetable happened to serve — the opposite of pre-registered.
# If they reach `build_events` or `build_connections` the comparison silently
# becomes a different one, and no number in the report would show it.


def tier_journal(tmp_path: Path) -> Path:
    """One origin and one far end, each with a plan, an observation and a poll."""
    return write(tmp_path, [
        {"t": "poll", "at": EPOCH_1800 - 3600, "eva": "8000001", "tier": 1,
         "ok": True, "stops": 1},
        {"t": "poll", "at": EPOCH_1800 - 3600, "eva": "8000041", "tier": 2,
         "ok": True, "stops": 1},
        plan_rec(EPOCH_1800 - 3600, trip="t1", eva="8000001"),
        plan_rec(EPOCH_1800 - 3600, trip="t2", eva="8000041"),
        obs_rec(EPOCH_1800 - 3600, PLANNED + 4, trip="t1", eva="8000001"),
        obs_rec(EPOCH_1800 - 3600, PLANNED + 4, trip="t2", eva="8000041"),
    ])


def test_a_far_end_station_is_not_read_as_an_origin(tmp_path: Path) -> None:
    stops, polls = se.read_day(tier_journal(tmp_path), DAY)
    assert {eva for eva, _ in stops} == {"8000001"}
    assert set(polls) == {"8000001"}


def test_the_far_end_is_still_there_when_it_is_asked_for(tmp_path: Path) -> None:
    """Dropped from the comparison, not from the data — scoring a two-leg
    journey needs exactly these stops."""
    stops, polls = se.read_day(tier_journal(tmp_path), DAY, tiers=(1, 2))
    assert {eva for eva, _ in stops} == {"8000001", "8000041"}
    assert set(polls) == {"8000001", "8000041"}


def test_a_journal_written_before_tiers_existed_is_all_origins(tmp_path: Path) -> None:
    """The first days were collected without the field, and every station in
    them was pre-registered. Treating a missing tier as anything else would
    silently empty those days."""
    path = write(tmp_path, [poll_rec(EPOCH_1800 - 3600),
                            plan_rec(EPOCH_1800 - 3600),
                            obs_rec(EPOCH_1800 - 3600, PLANNED + 4)])
    stops, polls = se.read_day(path, DAY)
    assert len(stops) == 1 and set(polls) == {"8000001"}


def test_a_far_end_cannot_originate_a_connection(tmp_path: Path) -> None:
    """The tighter guard: `build_connections` pairs any two stops at the same
    station, so a leaked tier-2 station would invent journeys nobody registered."""
    stops, polls = se.read_day(tier_journal(tmp_path), DAY)
    assert all(stop.eva == "8000001" for stop in stops.values())
