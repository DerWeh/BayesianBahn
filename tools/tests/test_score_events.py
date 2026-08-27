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


# --- and the cohorts must not be pooled either -------------------------------
#
# The second cohort samples by the kind of line, deliberately over-representing
# rare structures, and it started collecting weeks after the first. Pooling the
# two produces a number that describes neither sample — and the first is the one
# every published figure so far rests on.


def cohort_journal(tmp_path: Path) -> Path:
    return write(tmp_path, [
        {"t": "poll", "at": EPOCH_1800 - 3600, "eva": "8000001", "tier": 1,
         "cohort": 1, "ok": True, "stops": 1},
        {"t": "poll", "at": EPOCH_1800 - 3600, "eva": "8000025", "tier": 1,
         "cohort": 2, "ok": True, "stops": 1},
        plan_rec(EPOCH_1800 - 3600, trip="t1", eva="8000001"),
        plan_rec(EPOCH_1800 - 3600, trip="t2", eva="8000025"),
        obs_rec(EPOCH_1800 - 3600, PLANNED + 4, trip="t1", eva="8000001"),
        obs_rec(EPOCH_1800 - 3600, PLANNED + 4, trip="t2", eva="8000025"),
    ])


def test_a_second_cohort_origin_is_not_scored_by_default(tmp_path: Path) -> None:
    stops, polls = se.read_day(cohort_journal(tmp_path), DAY)
    assert set(polls) == {"8000001"}, "cohort 1 is what the headline speaks for"
    assert {eva for eva, _ in stops} == {"8000001"}


def test_the_second_cohort_is_there_when_it_is_asked_for(tmp_path: Path) -> None:
    stops, polls = se.read_day(cohort_journal(tmp_path), DAY, cohorts=(2,))
    assert set(polls) == {"8000025"}
    assert {eva for eva, _ in stops} == {"8000025"}


def test_the_two_filters_are_independent(tmp_path: Path) -> None:
    """A far end in cohort 2 is excluded twice over; asking for cohort 2 alone
    must not also let its far ends originate."""
    path = write(tmp_path, [
        {"t": "poll", "at": EPOCH_1800 - 3600, "eva": "8000025", "tier": 1,
         "cohort": 2, "ok": True, "stops": 1},
        {"t": "poll", "at": EPOCH_1800 - 3600, "eva": "8000041", "tier": 2,
         "cohort": 2, "ok": True, "stops": 1},
        plan_rec(EPOCH_1800 - 3600, trip="t1", eva="8000025"),
        plan_rec(EPOCH_1800 - 3600, trip="t2", eva="8000041"),
    ])
    stops, polls = se.read_day(path, DAY, cohorts=(2,))
    assert set(polls) == {"8000025"}


# --- two-leg journeys --------------------------------------------------------
#
# No collected day yet holds both a far-end forecast and an archive truth for
# it, so these fixtures are the only check this code has until one does. They
# are written against the behaviours that decide whether the comparison is fair
# rather than merely non-crashing: both forecasters answer from the same moment,
# over the same candidates, and neither is handed a fact the other was denied.

DEST = {"Musterstadt": "8000999"}
# The far end answered from long before anything is read, so the only
# reason a reading can be missing in these fixtures is the stop itself.
FAR_POLLS = {"8000999": [0.0]}


def far_stop(planned, *, trip="c1", eva="8000999", cat="RE", num="2",
             obs=()):
    stop = se.Stop(eva=eva, trip=trip, cat=cat, num=num, line=None, planned=planned)
    stop.obs = list(obs)
    return stop


def leg(planned_dep, *, trip="c1", cat="RE", num="2", eva="8000001",
        ppth=("Musterstadt",), obs=()):
    stop = se.Stop(eva=eva, trip=trip, cat=cat, num=num, line=None, planned=None,
                   planned_dep=planned_dep, ppth=list(ppth))
    stop.obs = list(obs)
    return stop


def feeder_stop(planned, *, trip="f1-2608171200-3", eva="8000001", cat="RE", num="1"):
    return se.Stop(eva=eva, trip=trip, cat=cat, num=num, line=None, planned=planned)


def journey_fixture(*, feeder_delay=0, feeder_db=0, cand_dep_truth=0,
                    cand_dep_live=0, cand_arr_truth=0, cand_arr_db=0,
                    slack=10):
    """One feeder, one candidate, both forecasters holding a reading."""
    arrive = 12 * 60          # feeder due at 12:00 wall-clock minutes
    depart = arrive + se.TRANSFER_MINUTES + slack
    reach = depart + 60
    read_at = se.wall_to_epoch(arrive) - 3600
    feeder = feeder_stop(arrive)
    feeder.obs = [(read_at - 60, arrive + feeder_db, None, False)]
    candidate = leg(depart, obs=[(read_at - 60, None, depart + cand_dep_live, False)])
    far = {("8000999", se.run_key("c1")):
           far_stop(reach, obs=[(read_at - 60, reach + cand_arr_db, None, False)])}
    truth = {
        ("8000001", "RE", "1", arrive): {"delay": feeder_delay, "dep_delay": 0,
                                         "cancelled": False},
        ("dep", "8000001", "RE", "2", depart): {"delay": 0,
                                                "dep_delay": cand_dep_truth,
                                                "cancelled": False},
        ("8000999", "RE", "2", reach): {"delay": cand_arr_truth, "dep_delay": 0,
                                        "cancelled": False},
    }
    stops = {("8000001", feeder.trip): feeder, ("8000001", "c1"): candidate}
    polls = {"8000001": [read_at]}
    return stops, far, polls, truth, {"arrive": arrive, "reach": reach}


def test_a_journey_is_built_from_a_feeder_and_a_reachable_destination():
    stops, far, polls, truth, _ = journey_fixture()
    got = se.build_journeys(stops, far, polls, FAR_POLLS, truth, DEST)
    assert len(got) == 1
    assert got[0]["dest_eva"] == "8000999" and got[0]["dest"] == "Musterstadt"
    assert [c["id"] for c in got[0]["candidates"]] == ["c1"]


def test_the_truth_is_the_arrival_of_the_train_actually_caught():
    stops, far, polls, truth, when = journey_fixture(cand_arr_truth=7)
    got = se.build_journeys(stops, far, polls, FAR_POLLS, truth, DEST)[0]
    assert got["truth_arrival"] == when["reach"] + 7
    assert got["caught_id"] == "c1"


def test_db_answers_with_the_train_its_own_forecasts_name():
    """Not the one that was actually caught: DB is scored on what it said."""
    stops, far, polls, truth, when = journey_fixture(cand_arr_db=4, cand_arr_truth=9)
    got = se.build_journeys(stops, far, polls, FAR_POLLS, truth, DEST)[0]
    assert got["db_arrival"] == when["reach"] + 4
    assert got["truth_arrival"] == when["reach"] + 9


def test_a_connection_missed_in_truth_falls_through_to_the_next_train():
    """The outcome the catch scorer can only call a failure has an arrival time,
    and it is the one the passenger actually experiences."""
    arrive = 12 * 60
    read_at = se.wall_to_epoch(arrive) - 3600
    feeder = feeder_stop(arrive)
    feeder.obs = [(read_at - 60, arrive, None, False)]
    early = leg(arrive + 7, trip="c1", num="2",
                obs=[(read_at - 60, None, arrive + 7, False)])
    late = leg(arrive + 40, trip="c2", num="3",
               obs=[(read_at - 60, None, arrive + 40, False)])
    far = {("8000999", se.run_key("c1")): far_stop(arrive + 70, trip="c1", num="2",
                                                   obs=[(read_at - 60, arrive + 70,
                                                         None, False)]),
           ("8000999", se.run_key("c2")): far_stop(arrive + 100, trip="c2", num="3",
                                                   obs=[(read_at - 60, arrive + 100,
                                                         None, False)])}
    truth = {
        ("8000001", "RE", "1", arrive): {"delay": 20, "dep_delay": 0,
                                         "cancelled": False},
        ("dep", "8000001", "RE", "2", arrive + 7): {"delay": 0, "dep_delay": 0,
                                                    "cancelled": False},
        ("dep", "8000001", "RE", "3", arrive + 40): {"delay": 0, "dep_delay": 0,
                                                     "cancelled": False},
        ("8000999", "RE", "2", arrive + 70): {"delay": 0, "dep_delay": 0,
                                              "cancelled": False},
        ("8000999", "RE", "3", arrive + 100): {"delay": 0, "dep_delay": 0,
                                               "cancelled": False},
    }
    stops = {("8000001", feeder.trip): feeder, ("8000001", "c1"): early,
             ("8000001", "c2"): late}
    got = se.build_journeys(stops, far, {"8000001": [read_at]}, FAR_POLLS, truth, DEST)[0]
    assert got["caught_id"] == "c2", "20 minutes late misses the 7-minute change"
    assert got["truth_arrival"] == arrive + 100
    assert got["db_id"] == "c1", "DB predicted the feeder on time"


def test_a_cancelled_candidate_is_not_boarded():
    stops, far, polls, truth, when = journey_fixture()
    key = ("dep", "8000001", "RE", "2", when["arrive"] + se.TRANSFER_MINUTES + 10)
    truth[key] = {**truth[key], "cancelled": True}
    got = se.build_journeys(stops, far, polls, FAR_POLLS, truth, DEST)[0]
    assert got["caught_id"] is None and got["truth_arrival"] is None


def test_the_feeder_is_never_its_own_connection():
    arrive = 12 * 60
    read_at = se.wall_to_epoch(arrive) - 3600
    feeder = feeder_stop(arrive, trip="c1")
    feeder.obs = [(read_at - 60, arrive, None, False)]
    feeder.planned_dep = arrive + 2
    feeder.ppth = ["Musterstadt"]
    stops = {("8000001", "c1"): feeder}
    far = {("8000999", se.run_key("c1")): far_stop(arrive + 60)}
    truth = {("8000001", "RE", "1", arrive): {"delay": 0, "dep_delay": 0,
                                              "cancelled": False}}
    assert se.build_journeys(stops, far, {"8000001": [read_at]}, FAR_POLLS, truth, DEST) == []


def test_a_destination_outside_the_second_tier_is_not_offered():
    """There is no forecast to read at a station nobody polls, so a journey
    ending there could only ever be scored against silence."""
    stops, far, polls, truth, _ = journey_fixture()
    assert se.build_journeys(stops, far, polls, FAR_POLLS, truth, {}) == []


def test_a_candidate_with_nothing_collected_at_the_far_end_is_dropped():
    stops, _, polls, truth, _ = journey_fixture()
    assert se.build_journeys(stops, {}, polls, FAR_POLLS, truth, DEST) == []


def test_a_journey_is_read_before_the_feeder_sets_off():
    """The last moment the answer can still change a decision — the same rule
    the connection scorer uses, so the two are read from the same instant."""
    stops, far, polls, truth, _ = journey_fixture()
    got = se.build_journeys(stops, far, polls, FAR_POLLS, truth, DEST)[0]
    assert got["read_at"] <= se.wall_to_epoch(cf.iris_time("2608171200"))


def test_a_feeder_we_were_not_yet_polling_for_is_skipped():
    stops, far, _, truth, _ = journey_fixture()
    assert se.build_journeys(stops, far, {"8000001": []}, FAR_POLLS, truth, DEST) == []


def test_the_candidate_list_is_capped_the_way_the_app_caps_it():
    """The model is given MAX_CANDIDATES trains; scoring it over more would
    score something the app never runs."""
    arrive = 12 * 60
    read_at = se.wall_to_epoch(arrive) - 3600
    feeder = feeder_stop(arrive)
    feeder.obs = [(read_at - 60, arrive, None, False)]
    stops = {("8000001", feeder.trip): feeder}
    far, truth = {}, {("8000001", "RE", "1", arrive): {"delay": 0, "dep_delay": 0,
                                                       "cancelled": False}}
    for i in range(se.MAX_CANDIDATES + 4):
        trip = f"c{i}"
        depart = arrive + se.TRANSFER_MINUTES + 3 + i
        stops[("8000001", trip)] = leg(depart, trip=trip, num=str(100 + i),
                                       obs=[(read_at - 60, None, depart, False)])
        far[("8000999", se.run_key(trip))] = far_stop(
            depart + 60, trip=trip, num=str(100 + i),
            obs=[(read_at - 60, depart + 60, None, False)])
        truth[("dep", "8000001", "RE", str(100 + i), depart)] = {
            "delay": 0, "dep_delay": 0, "cancelled": False}
        truth[("8000999", "RE", str(100 + i), depart + 60)] = {
            "delay": 0, "dep_delay": 0, "cancelled": False}
    got = se.build_journeys(stops, far, {"8000001": [read_at]}, FAR_POLLS, truth, DEST)[0]
    assert len(got["candidates"]) == se.MAX_CANDIDATES


def test_a_train_leaving_before_the_feeder_is_due_is_still_a_candidate():
    """A delayed earlier train is sometimes exactly the connection that works,
    and the app offers it — so the evaluation has to as well."""
    arrive = 12 * 60
    read_at = se.wall_to_epoch(arrive) - 3600
    feeder = feeder_stop(arrive)
    feeder.obs = [(read_at - 60, arrive, None, False)]
    early = leg(arrive - 10, trip="c0", num="9",
                obs=[(read_at - 60, None, arrive - 10, False)])
    booked = leg(arrive + 15, trip="c1", num="2",
                 obs=[(read_at - 60, None, arrive + 15, False)])
    stops = {("8000001", feeder.trip): feeder, ("8000001", "c0"): early,
             ("8000001", "c1"): booked}
    far = {("8000999", se.run_key("c0")): far_stop(arrive + 50, trip="c0", num="9",
                                                   obs=[(read_at - 60, arrive + 50,
                                                         None, False)]),
           ("8000999", se.run_key("c1")): far_stop(arrive + 75, trip="c1", num="2",
                                                   obs=[(read_at - 60, arrive + 75,
                                                         None, False)])}
    truth = {("8000001", "RE", "1", arrive): {"delay": 0, "dep_delay": 0,
                                              "cancelled": False}}
    for num, dep, arr in (("9", arrive - 10, arrive + 50), ("2", arrive + 15,
                                                            arrive + 75)):
        truth[("dep", "8000001", "RE", num, dep)] = {"delay": 0, "dep_delay": 0,
                                                     "cancelled": False}
        truth[("8000999", "RE", num, arr)] = {"delay": 0, "dep_delay": 0,
                                              "cancelled": False}
    got = se.build_journeys(stops, far, {"8000001": [read_at]}, FAR_POLLS, truth, DEST)[0]
    assert [c["id"] for c in got["candidates"]] == ["c0", "c1"]


def test_neither_forecaster_sees_the_others_information():
    """The fairness the whole comparison rests on: DB boards the train its own
    forecasts point at, truth boards the train the realised times point at, and
    with different delays those are different trains."""
    got = se.boarded(720, 20, 0, [
        {"id": "early", "planned_dep": 730, "planned_arr": 790, "live_dep": 0,
         "truth_dep": 0, "truth_arr": 0, "cancelled": False, "cancelled_live": False,
         "db_arr": 0},
        {"id": "later", "planned_dep": 760, "planned_arr": 820, "live_dep": 0,
         "truth_dep": 0, "truth_arr": 0, "cancelled": False, "cancelled_live": False,
         "db_arr": 0},
    ])
    assert got["db_id"] == "early" and got["caught_id"] == "later"


def _cand(name, dep, arr, *, cancelled=False):
    return {"id": name, "planned_dep": dep, "planned_arr": arr, "live_dep": 0,
            "truth_dep": 0, "truth_arr": 0, "cancelled": cancelled,
            "cancelled_live": False, "db_arr": 0}


def test_truth_boards_a_train_past_the_end_of_the_apps_own_list():
    """The app shows six trains. A passenger who misses all six waits for the
    seventh, and the forecast is then wrong by however long that took. Stopping
    at six instead dropped the journey — which excused exactly the answers that
    were most wrong."""
    shown = [_cand(f"c{i}", 700 + i, 760 + i) for i in range(se.MAX_CANDIDATES)]
    rest = [_cand("rescue", 900, 960)]
    got = se.boarded(720, 60, 0, shown, shown + rest)
    assert got["caught_id"] == "rescue"
    assert got["truth_arrival"] == 960
    assert got["caught_rank"] == se.MAX_CANDIDATES


def test_the_forecasters_still_answer_over_the_apps_own_list():
    """Truth walking further must not hand DB a train the model never saw —
    that would make the two forecasters answer different questions."""
    shown = [_cand(f"c{i}", 700 + i, 760 + i) for i in range(se.MAX_CANDIDATES)]
    rest = [_cand("rescue", 900, 960)]
    got = se.boarded(720, 60, 60, shown, shown + rest)
    assert got["db_arrival"] is None, "DB is not rescued by a train it never saw"
    assert got["truth_arrival"] == 960


def test_the_walk_steps_over_a_cancelled_train_to_the_next_that_ran():
    """The passenger's own rule: wait for the next train that really runs."""
    shown = [_cand("c0", 730, 790, cancelled=True)]
    rest = shown + [_cand("c1", 760, 820, cancelled=True),
                    _cand("c2", 800, 860)]
    got = se.boarded(720, 0, 0, shown, rest)
    assert got["caught_id"] == "c2" and got["caught_rank"] == 2


def test_a_passenger_left_behind_by_the_whole_day_is_still_dropped():
    """Walking the day is not the same as inventing a train. When nothing ran,
    there is no arrival to score against and the journey is counted, not
    guessed at."""
    shown = [_cand("c0", 730, 790, cancelled=True)]
    got = se.boarded(720, 0, 0, shown, shown)
    assert got["truth_arrival"] is None and got["caught_id"] is None


def test_boarded_without_a_full_list_walks_what_it_was_given():
    """`every` is optional: the old two-argument call still scores the six."""
    shown = [_cand("c0", 730, 790)]
    assert se.boarded(720, 0, 0, shown)["caught_id"] == "c0"


def test_a_journey_db_does_not_believe_in_has_no_db_answer():
    got = se.boarded(720, 0, 60, [
        {"id": "only", "planned_dep": 730, "planned_arr": 790, "live_dep": 0,
         "truth_dep": 0, "truth_arr": 0, "cancelled": False, "cancelled_live": False,
         "db_arr": 0},
    ])
    assert got["db_arrival"] is None and got["truth_arrival"] == 790


# --- the two halves of the journey scorer have to agree ----------------------
#
# Python decides which train the passenger caught; Kotlin decides what the model
# predicted. They are scoring the same journey only if they assume the same
# transfer time and the same candidate cap. Nothing in the output would show a
# divergence: both halves would keep producing plausible numbers about slightly
# different journeys.

def kotlin(path: str) -> str:
    return (TOOLS.parent / "app/src" / path).read_text(encoding="utf-8")


def test_the_transfer_time_matches_the_harness() -> None:
    source = kotlin("test/java/io/github/derweh/bayesianbahn/JourneyHarness.kt")
    assert f"const val TRANSFER_MINUTES = {se.TRANSFER_MINUTES}" in source


def test_the_candidate_cap_matches_the_app() -> None:
    """Scoring the model over more candidates than the app ever gives it would
    score something that does not ship."""
    source = kotlin("main/java/io/github/derweh/bayesianbahn/data/ConnectionPlanner.kt")
    assert f"const val MAX_CANDIDATES = {se.MAX_CANDIDATES}" in source


def test_the_candidate_window_matches_the_app() -> None:
    source = kotlin("main/java/io/github/derweh/bayesianbahn/data/ConnectionPlanner.kt")
    assert f"const val LOOK_BACK_MINUTES = {se.CANDIDATE_WINDOW_BEFORE}" in source


def test_the_already_gone_cap_matches_the_app() -> None:
    """The rule that stops six missed trains crowding out every catchable one
    lives in two languages; a change to one has to reach the other."""
    source = kotlin("main/java/io/github/derweh/bayesianbahn/data/ConnectionPlanner.kt")
    assert f"const val MAX_ALREADY_GONE = {se.MAX_ALREADY_GONE}" in source


def test_the_apps_list_is_not_all_trains_that_already_left():
    """The bug this rule exists for: at a station with a service every few
    minutes, taking the first six by departure time filled the list with trains
    that had gone before the feeder arrived — six impossible ones and no
    possible one. On one collected day that was 31% of journeys with a change."""
    feeder_planned = 600
    resolved = [{"id": f"gone{i}", "planned_dep": 570 + i * 5} for i in range(6)]
    resolved += [{"id": f"next{i}", "planned_dep": 605 + i * 10} for i in range(4)]
    got = se.pick_candidates(resolved, feeder_planned)
    assert [c["id"] for c in got] == ["gone4", "gone5", "next0", "next1",
                                      "next2", "next3"]


def test_a_just_missed_train_is_still_offered():
    """It is usually missed, but a delayed one is sometimes the connection that
    works — which is why the list keeps room for two of them."""
    resolved = [{"id": "just-missed", "planned_dep": 595},
                {"id": "next", "planned_dep": 610}]
    got = se.pick_candidates(resolved, 600)
    assert [c["id"] for c in got] == ["just-missed", "next"]


def test_the_last_trains_of_the_day_still_fill_the_list():
    """With only one train ahead, showing more of the missed ones beats showing
    a shorter list."""
    resolved = [{"id": f"gone{i}", "planned_dep": 580 + i} for i in range(6)]
    resolved += [{"id": "last", "planned_dep": 640}]
    got = se.pick_candidates(resolved, 600)
    assert len(got) == se.MAX_CANDIDATES
    assert got[-1]["id"] == "last"


def test_the_harness_builds_candidates_with_the_apps_own_code() -> None:
    """Not a description of it. The Python mirror of the arrival model drifted
    exactly this way, and kept producing plausible numbers while it did."""
    source = kotlin("test/java/io/github/derweh/bayesianbahn/JourneyHarness.kt")
    assert "CandidateBuilder.build(" in source
    assert "ConnectionModel.propagate(" in source


# --- what "no reading" means ------------------------------------------------
#
# Found by running the builder over a real collected day rather than by
# reasoning about it: 60% of candidates had no far-end reading and 52% no
# departure reading, which would have been read as "DB declines to answer" and
# dropped. It is not. DB states a stop in four shapes and three of them mean on
# time — a stop it lists no change for is one of them. Only a station we had not
# begun polling is genuinely silent, and the poll log is what tells them apart.


def test_a_stop_db_lists_no_change_for_is_db_saying_on_time():
    stops, far, polls, truth, when = journey_fixture()
    stops[("8000001", "c1")].obs = []                     # nothing at the transfer
    far[("8000999", se.run_key("c1"))].obs = []           # nothing at the far end
    got = se.build_journeys(stops, far, polls, FAR_POLLS, truth, DEST)[0]
    assert got["candidates"][0]["live_dep"] == 0
    assert got["candidates"][0]["db_arr"] == 0
    assert got["db_arrival"] == when["reach"], "DB does answer, and says on time"


def test_a_far_end_we_had_not_begun_polling_is_genuinely_silent():
    """The one case where nothing can be read: no poll at all by then. Scoring
    DB as if it had said 'on time' there would credit it with a prediction it
    never made."""
    stops, far, polls, truth, _ = journey_fixture()
    read_at = polls["8000001"][0]
    assert se.build_journeys(stops, far, polls, {"8000999": [read_at + 1]},
                             truth, DEST) == []


def test_an_unjoined_candidate_stops_the_walk_rather_than_being_stepped_over():
    """Stepping over it hands the passenger a later train than they may have
    taken, and the error lands in the tail — the part being measured."""
    got = se.boarded(720, 0, 0, [
        {"id": "unknown", "planned_dep": 730, "planned_arr": 790, "live_dep": 0,
         "truth_dep": None, "truth_arr": None, "cancelled": None,
         "cancelled_live": False, "db_arr": 0},
        {"id": "later", "planned_dep": 760, "planned_arr": 820, "live_dep": 0,
         "truth_dep": 0, "truth_arr": 0, "cancelled": False,
         "cancelled_live": False, "db_arr": 0},
    ])
    assert got["truth_arrival"] is None and got["caught_id"] is None
    assert got["db_id"] == "unknown", "DB's own side resolves regardless"


def test_a_candidate_the_archive_never_joined_is_not_read_as_cancelled():
    """It used to default to cancelled, which skipped straight past it."""
    stops, far, polls, truth, _ = journey_fixture()
    del truth[("dep", "8000001", "RE", "2", 12 * 60 + se.TRANSFER_MINUTES + 10)]
    got = se.build_journeys(stops, far, polls, FAR_POLLS, truth, DEST)[0]
    assert got["candidates"][0]["cancelled"] is None
    assert got["truth_arrival"] is None, "unknown, so the journey is not scored"


def test_a_boarded_candidate_with_no_recorded_arrival_is_not_scored():
    got = se.boarded(720, 0, 0, [
        {"id": "only", "planned_dep": 730, "planned_arr": 790, "live_dep": 0,
         "truth_dep": 0, "truth_arr": None, "cancelled": False,
         "cancelled_live": False, "db_arr": 0},
    ])
    assert got["truth_arrival"] is None
