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
