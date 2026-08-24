"""Tests for the forecast collector.

This is the one part of the evaluation that cannot be recomputed. The archive
records what a train did, never what DB predicted beforehand, so a lost hour of
collection is lost for good — which makes the recovery behaviour, not the
parsing, the thing worth testing hardest.

The failure modes tested here are the ones a multi-day run on a small machine
actually meets: killed mid-write, restarted, network gone, DB unreachable, and
the day rolling over underneath the process.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import collect_forecasts as cf  # noqa: E402
import route_bench as rb  # noqa: E402

PLAN = """<timetable station="Ulm Hbf" eva="8000170">
  <s id="trip-1"><tl f="N" t="p" o="80" c="RE" n="4230" l="9"/>
    <ar pt="2606101210" pp="2"/>
    <dp pt="2606101212" pp="2" ppth="Neu-Ulm|Senden"/>
  </s>
  <s id="trip-2"><tl f="N" t="p" o="80" c="ICE" n="599"/>
    <ar pt="2606101230" pp="1"/>
  </s>
</timetable>"""

FCHG = """<timetable station="Ulm Hbf" eva="8000170">
  <s id="trip-1"><ar ct="2606101216"/><dp ct="2606101218"/></s>
  <s id="trip-2"><ar ct="2606101245" cs="c"/></s>
</timetable>"""

FCHG_LATER = FCHG.replace('ct="2606101216"', 'ct="2606101223"')


def collector(tmp_path: Path, responses: dict[str, str], clock: list[float]):
    """A collector wired to canned responses and a clock the test controls."""
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        for fragment, body in responses.items():
            if fragment in url:
                if isinstance(body, Exception):
                    raise body
                return body
        raise OSError("no route to host")

    got = cf.Collector([cf.Station("8000170", "Ulm Hbf")], out=tmp_path,
                       fetch=fetch, now=lambda: clock[0])
    got.calls = calls
    return got


# --- parsing -----------------------------------------------------------------


def test_plan_gives_planned_times_and_identity() -> None:
    got = cf.parse_plan(PLAN)
    assert got["trip-1"]["cat"] == "RE" and got["trip-1"]["num"] == "4230"
    assert got["trip-1"]["par"] == cf.iris_time("2606101210")
    assert got["trip-2"]["pdp"] is None, "an arrival-only stop has no departure"


def test_changes_read_the_forecast_and_the_cancellation() -> None:
    got = cf.parse_changes(FCHG)
    assert got["trip-1"]["ar"] == cf.iris_time("2606101216")
    assert got["trip-1"]["arc"] is False
    assert got["trip-2"]["arc"] is True, 'cs="c" is a cancellation'


def test_time_convention_matches_the_benchmark() -> None:
    """Both read IRIS times as German wall clock; a divergence would silently
    shift every comparison by the UTC offset."""
    assert cf.iris_time("2606101210") == rb.wall_minutes(dt.datetime(2026, 6, 10, 12, 10))


def test_a_malformed_time_is_dropped_not_guessed() -> None:
    assert cf.iris_time(None) is None
    assert cf.iris_time("") is None
    assert cf.iris_time("26061012") is None


# --- the journal -------------------------------------------------------------


def test_every_record_is_its_own_line_and_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    journal = cf.Journal(path)
    journal.append({"t": "poll", "at": 1})
    journal.append({"t": "obs", "at": 2})
    journal.close()
    cf.Journal(path).append({"t": "obs", "at": 3})
    records, torn = cf.Journal.read(path)
    assert [r["at"] for r in records] == [1, 2, 3] and torn == 0


def test_a_torn_final_line_costs_only_that_record(tmp_path: Path) -> None:
    """What a power cut leaves behind: everything before it must still read."""
    path = tmp_path / "j.jsonl"
    journal = cf.Journal(path)
    for i in range(3):
        journal.append({"t": "obs", "at": i})
    journal.close()
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"t": "obs", "at": 3, "ev')  # killed mid-write
    records, torn = cf.Journal.read(path)
    assert [r["at"] for r in records] == [0, 1, 2]
    assert torn == 1


def test_reading_a_journal_that_does_not_exist_is_not_an_error(tmp_path: Path) -> None:
    assert cf.Journal.read(tmp_path / "nope.jsonl") == ([], 0)


# --- collection --------------------------------------------------------------


def test_a_poll_records_plan_forecast_and_the_attempt(tmp_path: Path) -> None:
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.poll_station(got.stations[0])
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    kinds = [r["t"] for r in records]
    assert kinds.count("poll") == 1 and kinds.count("plan") == 2
    assert kinds.count("obs") == 2
    assert [r for r in records if r["t"] == "poll"][0]["ok"] is True


def test_an_unchanged_forecast_is_not_written_twice(tmp_path: Path) -> None:
    """The trajectory is the moves; most stops do not move between polls."""
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.poll_station(got.stations[0])
    clock[0] += 600
    got.poll_station(got.stations[0])
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    assert sum(1 for r in records if r["t"] == "obs") == 2, "no repeats"
    assert sum(1 for r in records if r["t"] == "poll") == 2, "but both polls logged"


def test_a_changed_forecast_is_appended(tmp_path: Path) -> None:
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.poll_station(got.stations[0])
    got._fetch = lambda url: PLAN if "/plan/" in url else FCHG_LATER
    clock[0] += 600
    got.poll_station(got.stations[0])
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    moves = [r for r in records if r["t"] == "obs" and r["id"] == "trip-1"]
    assert [m["ar"] for m in moves] == [cf.iris_time("2606101216"),
                                        cf.iris_time("2606101223")]


def test_a_restart_resumes_without_duplicating(tmp_path: Path) -> None:
    """The whole point: an interrupted run must not re-report what it knows."""
    clock = [1781000000.0]
    first = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    first.poll_station(first.stations[0])
    first.journal.close()

    clock[0] += 600
    second = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    second.restore(second._today())
    second.poll_station(second.stations[0])

    records, _ = cf.Journal.read(tmp_path / f"forecasts-{second._today()}.jsonl")
    assert sum(1 for r in records if r["t"] == "obs") == 2, "state was rebuilt"
    assert sum(1 for r in records if r["t"] == "plan") == 2, "plan not re-announced"


def test_a_restart_after_a_torn_line_still_resumes(tmp_path: Path) -> None:
    clock = [1781000000.0]
    first = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    first.poll_station(first.stations[0])
    first.journal.close()
    path = tmp_path / f"forecasts-{first._today()}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"t": "obs", "at": 999, "eva": "8000')

    second = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    assert second.restore(second._today()) == 1
    second.poll_station(second.stations[0])
    records, torn = cf.Journal.read(path)
    assert torn == 1
    assert sum(1 for r in records if r["t"] == "obs") == 2


def test_a_failed_poll_is_recorded_so_a_gap_is_not_read_as_silence(tmp_path: Path) -> None:
    """An outage must be distinguishable from DB not changing its mind."""
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": OSError("down")}, clock)
    got.poll_station(got.stations[0])
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    polls = [r for r in records if r["t"] == "poll"]
    assert len(polls) == 1 and polls[0]["ok"] is False
    assert not [r for r in records if r["t"] == "obs"]


def test_malformed_xml_fails_the_poll_rather_than_the_run(tmp_path: Path) -> None:
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": "<timetable><s id="}, clock)
    got.poll_station(got.stations[0])
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    assert [r["ok"] for r in records if r["t"] == "poll"] == [False]


def test_plan_documents_are_fetched_once_and_cached(tmp_path: Path) -> None:
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.poll_station(got.stations[0])
    first = sum(1 for c in got.calls if "/plan/" in c)
    got.poll_station(got.stations[0])
    assert sum(1 for c in got.calls if "/plan/" in c) == first, "served from disk"
    assert not list((tmp_path / "plan").glob("*.part")), "no half-written cache"


def test_a_plan_fetch_that_fails_does_not_poison_the_cache(tmp_path: Path) -> None:
    clock = [1781000000.0]
    got = collector(tmp_path, {"/fchg/": FCHG}, clock)  # /plan/ raises
    got.poll_station(got.stations[0])
    assert not list((tmp_path / "plan").glob("*.xml"))
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    # The forecast is still worth keeping; only the planned time is missing.
    assert sum(1 for r in records if r["t"] == "obs") == 2
    assert not [r for r in records if r["t"] == "plan"]


def test_the_journal_rolls_over_at_midnight(tmp_path: Path) -> None:
    before = dt.datetime(2026, 6, 10, 23, 55).timestamp()
    clock = [before]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.poll_station(got.stations[0])
    clock[0] = dt.datetime(2026, 6, 11, 0, 5).timestamp()
    got.poll_station(got.stations[0])
    assert (tmp_path / "forecasts-2026-06-10.jsonl").exists()
    assert (tmp_path / "forecasts-2026-06-11.jsonl").exists()


# --- the schedule ------------------------------------------------------------


def test_polls_are_aligned_to_the_clock_so_restarts_land_in_step() -> None:
    at = dt.datetime(2026, 6, 10, 12, 3, 20).timestamp()
    assert cf.slot_start(at, cadence=10) == dt.datetime(2026, 6, 10, 12, 10).timestamp()
    exact = dt.datetime(2026, 6, 10, 12, 10).timestamp()
    assert cf.slot_start(exact, cadence=10) == dt.datetime(2026, 6, 10, 12, 20).timestamp()


def test_run_stops_when_asked(tmp_path: Path) -> None:
    """SIGTERM sets the flag; the loop must not start another station."""
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.stopping = True
    got.run(minutes=60, sleep=lambda _s: None)
    assert not (tmp_path / f"forecasts-{got._today()}.jsonl").exists()


def test_run_polls_each_slot_until_the_deadline(tmp_path: Path) -> None:
    clock = [dt.datetime(2026, 6, 10, 12, 0).timestamp()]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.run(minutes=25, sleep=lambda s: clock.__setitem__(0, clock[0] + max(s, 1)))
    records, _ = cf.Journal.read(tmp_path / "forecasts-2026-06-10.jsonl")
    assert sum(1 for r in records if r["t"] == "poll") == 2, "12:10 and 12:20"


# --- the pre-registered station set ------------------------------------------


def test_the_station_set_is_stratified_and_committed() -> None:
    stations = cf.load_stations(TOOLS / "forecast_stations.csv")
    assert len(stations) == 20
    by_eva = rb.stations()
    weights = sorted(by_eva[s.eva].weight for s in stations)
    assert weights[0] < 40, "the thin cases are in on purpose"
    assert weights[-1] >= 250, "and the busy ones, where most journeys start"
    assert len({s.eva for s in stations}) == 20


def test_hafas_backs_off_while_the_proxy_is_down(tmp_path: Path) -> None:
    """It answered 503 on every data endpoint the day this was written; a
    cross-check must not turn into a retry storm against a struggling service."""
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    base = got.hafas_interval()
    for expected in (2, 4, 8, 8):
        got.cross_check_hafas(__import__("random").Random(1))
        assert got.hafas_interval() == base * expected
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    assert all(r["ok"] is False for r in records if r["t"] == "hafas")


def test_hafas_recovers_on_the_first_success(tmp_path: Path) -> None:
    clock = [1781000000.0]
    payload = json.dumps({"departures": [
        {"line": {"name": "RE 1"}, "when": "2026-06-10T12:16:00+02:00",
         "plannedWhen": "2026-06-10T12:10:00+02:00", "delay": 360},
    ]})
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG,
                               "transport.rest": payload}, clock)
    got.hafas_failures = 3
    got.cross_check_hafas(__import__("random").Random(1))
    assert got.hafas_failures == 0
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    rows = [r for r in records if r["t"] == "hafas" and r["ok"]]
    assert rows and rows[0]["rows"][0]["delay"] == 360


def poll_records(slots: list[int], evas: list[str], *, ok=True, cadence=10) -> list[dict]:
    return [{"t": "poll", "at": s * cadence * 60 + 1, "eva": e,
             "ok": ok, "stops": 100}
            for s in slots for e in evas]


def test_health_counts_rounds_and_stations() -> None:
    got = cf.health(poll_records([100, 101, 102], ["a", "b"]), expected_stations=2)
    assert got["rounds"] == 3 and got["missed_slots"] == 0
    assert got["stations_seen"] == 2 and got["stops"] == 600


def test_health_notices_a_slot_that_produced_nothing() -> None:
    """A suspended laptop or a crash-and-restart leaves a hole; `ps` cannot see it."""
    got = cf.health(poll_records([100, 101, 104], ["a"]), expected_stations=1)
    assert got["rounds"] == 3 and got["missed_slots"] == 2


def test_health_notices_a_station_that_never_answers() -> None:
    records = poll_records([100, 101], ["a"]) + poll_records([100], ["b"], ok=False)
    got = cf.health(records, expected_stations=3)
    assert got["stations_seen"] == 2 and got["stations_expected"] == 3
    assert got["failed"] == 1
    assert got["stops"] == 200, "a failed poll contributes no stops"


def test_health_of_an_empty_journal_is_not_an_error() -> None:
    got = cf.health([], expected_stations=20)
    assert got["rounds"] == 0 and got["missed_slots"] == 0 and got["last_at"] is None


def test_polls_are_jittered_inside_their_slot(tmp_path: Path) -> None:
    """Sampling on the exact grid would fix our phase against DB's update cycle
    and against the clock-friendly minutes trains are scheduled on."""
    import random
    seen = set()
    for seed in range(6):
        clock = [dt.datetime(2026, 6, 10, 12, 0).timestamp()]
        got = collector(tmp_path / f"s{seed}", {"/plan/": PLAN, "/fchg/": FCHG}, clock)
        # Long enough that the jittered first slot always fits inside it.
        got.run(minutes=20, sleep=lambda s: clock.__setitem__(0, clock[0] + max(s, 1)),
                rng=random.Random(seed))
        records, _ = cf.Journal.read(
            tmp_path / f"s{seed}" / "forecasts-2026-06-10.jsonl")
        polls = [r["at"] for r in records if r["t"] == "poll"]
        assert polls
        seen.add(min(polls) % (cf.CADENCE_MINUTES * 60))
    assert len(seen) > 1, "different seeds must land at different offsets"
    assert all(0 <= s <= cf.JITTER_SECONDS + 60 for s in seen), "but inside the slot"


def test_jitter_does_not_move_a_poll_out_of_its_slot() -> None:
    """Health accounting buckets by slot; the offset must stay smaller than one."""
    assert cf.JITTER_SECONDS < cf.CADENCE_MINUTES * 60


# --- the second tier: the far end of a change --------------------------------
#
# The failure this guards is not a crash. If a destination station leaks into
# tier 1 it becomes an *origin*, and the comparison quietly stops being the one
# that was pre-registered — with nothing in the output to say so.


def test_the_plan_keeps_the_whole_onward_path() -> None:
    """The terminus is the far end of the second leg; only the next stop was
    kept before, which made the two-leg journey unscoreable after the fact."""
    plan = cf.parse_plan(PLAN)
    assert plan["trip-1"]["ppth"] == ["Neu-Ulm", "Senden"]


def test_a_stop_with_no_onward_path_keeps_an_empty_one() -> None:
    plan = cf.parse_plan(PLAN)
    assert plan["trip-2"]["ppth"] == []


def write_stations(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text("# a comment\n"
                    + "".join(f"{eva};{name};1\n" for eva, name in rows),
                    encoding="utf-8")
    return path


ONE = (1, "o1.csv", "d1.csv")
TWO = (2, "o2.csv", "d2.csv")


def roles(stations) -> list[tuple[str, int, int]]:
    return [(s.eva, s.tier, s.cohort) for s in stations]


def test_destinations_are_loaded_as_a_second_tier(tmp_path: Path) -> None:
    write_stations(tmp_path / "o1.csv", [("8000001", "Aachen Hbf")])
    write_stations(tmp_path / "d1.csv", [("8000041", "Bochum Hbf")])
    assert roles(cf.station_set(tmp_path, (ONE,))) == [
        ("8000001", 1, 1), ("8000041", 2, 1)]


def test_a_station_in_both_files_is_polled_once_as_an_origin(tmp_path: Path) -> None:
    """Three of the twenty origins are also termini. Polling them twice a round
    would double their requests, and tier 2 must not shadow tier 1."""
    write_stations(tmp_path / "o1.csv", [("8000310", "Remagen")])
    write_stations(tmp_path / "d1.csv",
                   [("8000310", "Remagen"), ("8000041", "Bochum Hbf")])
    assert roles(cf.station_set(tmp_path, (ONE,))) == [
        ("8000310", 1, 1), ("8000041", 2, 1)]


def test_a_missing_destinations_file_leaves_the_origins_alone(tmp_path: Path) -> None:
    write_stations(tmp_path / "o1.csv", [("8000001", "Aachen Hbf")])
    assert roles(cf.station_set(tmp_path, (ONE,))) == [("8000001", 1, 1)]


def test_a_later_cohort_cannot_take_a_station_from_an_earlier_one(tmp_path: Path
                                                                  ) -> None:
    """The collision rule that keeps a cohort's data interpretable: a station
    already polled for cohort 1 stays cohort 1, whatever a later file says. Data
    already collected under a registration cannot be re-labelled by a new one."""
    write_stations(tmp_path / "o1.csv", [("8000001", "Aachen Hbf")])
    write_stations(tmp_path / "d1.csv", [("8000041", "Bochum Hbf")])
    write_stations(tmp_path / "o2.csv",
                   [("8000041", "Bochum Hbf"), ("8000025", "Bamberg")])
    write_stations(tmp_path / "d2.csv", [("8000001", "Aachen Hbf")])
    assert roles(cf.station_set(tmp_path, (ONE, TWO))) == [
        ("8000001", 1, 1), ("8000041", 2, 1), ("8000025", 1, 2)]


def test_a_cohort_with_no_files_yet_is_simply_absent(tmp_path: Path) -> None:
    write_stations(tmp_path / "o1.csv", [("8000001", "Aachen Hbf")])
    assert roles(cf.station_set(tmp_path, (ONE, TWO))) == [("8000001", 1, 1)]


def test_a_poll_records_which_cohort_the_station_belongs_to(tmp_path: Path) -> None:
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.poll_station(cf.Station("8000170", "Ulm Hbf", 1, 2))
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    poll = next(r for r in records if r["t"] == "poll")
    assert (poll["tier"], poll["cohort"]) == (1, 2)


def test_a_journal_without_a_cohort_is_the_first_one() -> None:
    """Every day before cohort 2 existed. Reading them as anything else would
    empty the days the published comparison rests on."""
    assert cf.roles_of([{"t": "poll", "at": 1, "eva": "8000001", "ok": True}]) == {
        "8000001": (1, 1)}
    assert cf.roles_of([{"t": "poll", "at": 1, "eva": "8000041", "tier": 2,
                         "ok": True}]) == {"8000041": (2, 1)}


def test_a_poll_records_which_tier_the_station_was(tmp_path: Path) -> None:
    """Written into the journal, not looked up in the CSV: editing the station
    files later must not be able to re-label data already collected."""
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG}, clock)
    got.stations = [cf.Station("8000170", "Ulm Hbf", 2)]
    got.poll_station(got.stations[0])
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    polls = [r for r in records if r["t"] == "poll"]
    assert [r["tier"] for r in polls] == [2]


def test_a_failed_poll_still_records_its_tier(tmp_path: Path) -> None:
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN}, clock)   # no /fchg/ route
    got.poll_station(cf.Station("8000170", "Ulm Hbf", 2))
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    poll = next(r for r in records if r["t"] == "poll")
    assert poll["ok"] is False and poll["tier"] == 2


def test_the_hafas_cross_check_stays_on_the_comparisons_own_stations(
        tmp_path: Path) -> None:
    """It checks the feed the comparison is built on, and the proxy's budget is
    small enough that spending it on the far ends would empty it."""
    clock = [1781000000.0]
    got = collector(tmp_path, {"/plan/": PLAN, "/fchg/": FCHG,
                               "/stops/": '{"departures": []}'}, clock)
    got.stations = [cf.Station("8000170", "Ulm Hbf", 1),
                    cf.Station("8000041", "Bochum Hbf", 2),
                    cf.Station("8000049", "Braunschweig Hbf", 2)]
    got.cross_check_hafas(__import__("random").Random(0))
    records, _ = cf.Journal.read(tmp_path / f"forecasts-{got._today()}.jsonl")
    assert {r["eva"] for r in records if r["t"] == "hafas"} == {"8000170"}


def test_the_destination_set_is_derived_and_committed() -> None:
    """It is a function of the timetable, so it must be reproducible from the
    file alone: a real eva per row, no duplicates, and no origin smuggled in."""
    ends = cf.load_stations(TOOLS / "forecast_destinations.csv", 2)
    assert len(ends) > 40, "a second tier this thin would score almost nothing"
    assert all(s.eva.isdigit() and s.name for s in ends)
    assert len({s.eva for s in ends}) == len(ends)
    assert all(s.tier == 2 for s in ends)


def test_polling_every_cohort_still_fits_inside_a_slot() -> None:
    """A round that overruns [CADENCE_MINUTES] drops slots, and the whole
    comparison rests on having a reading close to the moment being scored.

    This is also the rate limit. Growing the station set is how the comparison
    widens, and this is the check that says how far it may grow."""
    stations = cf.station_set(TOOLS)
    slot = cf.CADENCE_MINUTES * 60
    assert len(stations) * cf.SECONDS_PER_STATION < slot * cf.MAX_SLOT_FRACTION, (
        f"{len(stations)} stations overruns the slot")


def test_the_station_set_stays_inside_the_self_imposed_request_rate() -> None:
    """One fchg per station per round, plus one plan document per station per
    hour — the horizon is cached after the first round of each hour."""
    stations = cf.station_set(TOOLS)
    rounds_per_hour = 60 / cf.CADENCE_MINUTES
    per_hour = len(stations) * rounds_per_hour + len(stations)
    assert per_hour / 3600 < cf.MAX_REQUESTS_PER_SECOND, f"{per_hour / 3600:.2f}/s"


def test_the_tier_comes_from_the_journal_not_the_station_files() -> None:
    records = [{"t": "poll", "at": 1, "eva": "8000041", "tier": 2, "ok": True},
               {"t": "poll", "at": 2, "eva": "8000041", "tier": 2, "ok": True},
               {"t": "poll", "at": 1, "eva": "8000001", "ok": True}]
    assert cf.tiers_of(records) == {"8000041": 2, "8000001": 1}


def test_status_judges_a_day_by_the_tiers_it_was_collected_under(
        tmp_path: Path, capsys) -> None:
    """The days collected before the second tier existed are complete with
    twenty stations; measuring them against today's 101 would flag every one."""
    journal = cf.Journal(tmp_path / "forecasts-2026-08-18.jsonl")
    for eva in [s.eva for s in cf.load_stations(TOOLS / "forecast_stations.csv")]:
        journal.append({"t": "poll", "at": 1787000000, "eva": eva, "ok": True,
                        "stops": 1})
    journal.close()
    cf.status(tmp_path, TOOLS, now=lambda: 1787000060.0)
    assert "stations" not in capsys.readouterr().out.split("rounds")[1]
