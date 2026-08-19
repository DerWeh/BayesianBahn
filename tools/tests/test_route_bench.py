"""Tests for the routing benchmarks.

A benchmark that measures the wrong algorithm is worse than none: it produces
numbers that look like evidence. Two such bugs are the reason this file exists.

  * `journey_bench.py` filtered feeders through a hand-written category
    allow-list, {RE, RB, S, IRE, RS}, while the app filters through
    `DeutschlandTicket.covers()` — which excludes only long-distance trains and
    therefore keeps the private regional operators. In their regions those are
    the only feeders there are, so the harness was measuring a strictly weaker
    search than the one that ships.
  * Its ground truth was built by the same board-walk the search uses, so it
    could only ever propose journeys that mechanism can see. That is why it
    reported 98% recall where an exhaustive ground truth reports 77%.

Both classes of bug are silent, so the mirrored constants are checked against
the Kotlin they mirror, and the exhaustive scan is checked against hand-built
timetables small enough to verify by eye.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
import time
from pathlib import Path

import polars as pl
import pytest

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import journey_bench  # noqa: E402
import route_bench as rb  # noqa: E402

APP = ROOT / "app/src/main/java/io/github/derweh/bayesianbahn"

# Aligned to an hour so the IRIS bucket boundaries are exactly where the
# arithmetic in the tests says they are.
BASE = 29_000_040 // 60 * 60


def kotlin(path: str) -> str:
    return (APP / path).read_text(encoding="utf-8")


def const(source: str, name: str) -> float:
    """The value of a `const val NAME = <number>` in Kotlin."""
    m = re.search(rf"const val {name} = ([0-9.]+)", source)
    assert m, f"{name} is no longer a const val — the mirror cannot be checked"
    return float(m.group(1))


def tt_of(*rides: tuple[str, str, list[tuple[str, int | None, int | None]]]) -> rb.Timetable:
    """A timetable from explicit rides: (category, number, [(eva, arr, dep)])."""
    return rb.Timetable({i: (cat, num, tuple(stops))
                         for i, (cat, num, stops) in enumerate(rides)})


def station(eva: str, weight: int, lat: float, lon: float = 10.0) -> rb.Station:
    return rb.Station(eva, f"station-{eva}", weight, lat, lon)


# --- the mirrors -------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "ORIGIN_HOURS", "MAX_TRANSFER_SCAN", "MAX_TRANSFER_RESULTS",
    "MAX_TRANSFER_ATTEMPTS", "TRANSFERS_PER_FEEDER", "MIN_TRANSFER_WEIGHT",
    "DETOUR_TOLERANCE", "MAX_DIRECT",
])
def test_constants_mirror_the_planner(name: str) -> None:
    assert getattr(rb, name) == const(kotlin("data/JourneyPlanner.kt"), name), (
        f"{name} drifted from JourneyPlanner.kt; the benchmark would measure "
        "an app that does not exist"
    )


def test_transfer_board_window_mirrors_the_connection_planner() -> None:
    # ConnectionPlanner passes hours = 4 literally rather than via a constant.
    source = kotlin("data/ConnectionPlanner.kt")
    hours = {int(h) for h in re.findall(r"board\([^)]*?hours = (\d+)", source, re.S)}
    assert hours == {rb.TRANSFER_HOURS}


def app_long_distance() -> set[str]:
    """The app's own long-distance boundary, read from the source of truth."""
    block = re.search(r"LONG_DISTANCE_CATEGORIES = setOf\((.*?)\)",
                      kotlin("model/DelayModel.kt"), re.S)
    assert block, "LONG_DISTANCE_CATEGORIES moved; the mirrors cannot be checked"
    return set(re.findall(r'"([^"]+)"', block.group(1)))


def python_long_distance() -> dict[str, set[str]]:
    """Every Python copy of that set, found rather than listed.

    Listing them by hand is how the drift below survived: `backtest.py` grew a
    copy, the test knew about two others, and nobody noticed the third was a
    category short. Discovery means a fourth copy cannot be added without being
    checked.
    """
    found = {}
    for path in sorted(ROOT.glob("tools/*.py")) + sorted(ROOT.glob("pipeline/*.py")):
        block = re.search(r"^LONG_DISTANCE = \{(.*?)\}", path.read_text(encoding="utf-8"),
                          re.S | re.M)
        if block:
            found[str(path.relative_to(ROOT))] = set(re.findall(r'"([^"]+)"', block.group(1)))
    return found


def test_every_python_copy_of_the_long_distance_set_matches_the_app() -> None:
    """`backtest.py` was missing "WB" — every Westbahn service was scored
    against the regional prior, and the parameters it selected were chosen
    under a classification the app does not use."""
    copies = python_long_distance()
    assert copies, "the discovery pattern found no copies at all; it has stopped working"
    expected = app_long_distance()
    for where, value in copies.items():
        assert value == expected, (
            f"{where} drifted from DelayModel.kt: "
            f"missing {sorted(expected - value)}, extra {sorted(value - expected)}"
        )


def test_the_known_mirrors_are_among_the_discovered_ones() -> None:
    """A rename that hides a copy from discovery would make the test above pass
    for the wrong reason."""
    assert {"tools/route_bench.py", "tools/journey_bench.py",
            "pipeline/backtest.py"} <= set(python_long_distance())


def test_the_imported_modules_agree_with_their_source() -> None:
    expected = app_long_distance()
    assert rb.LONG_DISTANCE == expected
    assert journey_bench.LONG_DISTANCE == expected


@pytest.mark.parametrize("category", ["HLB", "NWB", "ARV", "AVG", "ag", "MEX", "BRB",
                                      "RE", "RB", "S", "IRE"])
def test_private_regional_operators_are_usable_feeders(category: str) -> None:
    """The bug this file's docstring opens with: these were excluded."""
    assert rb.covers(category)
    assert journey_bench.covers(category)


@pytest.mark.parametrize("category", ["ICE", "IC", "EC", "FLX", "NJ", "TGV"])
def test_long_distance_is_not_covered(category: str) -> None:
    assert not rb.covers(category)
    assert not journey_bench.covers(category)


# --- the timetable -----------------------------------------------------------


def test_board_serves_whole_hours_like_iris() -> None:
    # A departure 3h59 after the hour is inside a 4-hour board; 4h01 is not.
    tt = tt_of(("RE", "1", [("A", None, BASE + 239)]),
               ("RE", "2", [("A", None, BASE + 241)]))
    assert len(tt.board("A", BASE, 4)) == 1
    assert len(tt.board("A", BASE, 5)) == 2


def test_board_starts_at_the_hour_not_the_request() -> None:
    """IRIS serves the whole hour, so a stop 10 minutes earlier is visible."""
    tt = tt_of(("RE", "1", [("A", None, BASE + 10)]))
    assert len(tt.board("A", BASE + 20, 1) or []) == 1


def test_board_reports_the_onward_path_only() -> None:
    tt = tt_of(("RE", "1", [("A", None, BASE), ("B", BASE + 20, BASE + 22),
                            ("C", BASE + 40, None)]))
    stop = tt.board("A", BASE, 1)[0]
    assert stop.path == (("B", BASE + 20), ("C", BASE + 40))
    assert tt.board("B", BASE, 2)[0].path == (("C", BASE + 40),)


def test_board_is_empty_for_an_unknown_station() -> None:
    assert tt_of(("RE", "1", [("A", None, BASE)])).board("Z", BASE, 3) == []


# --- exhaustive ground truth -------------------------------------------------


def one_change_network() -> rb.Timetable:
    """O --RE1--> V --RB2--> D, plus a direct RE3 from O to X."""
    return tt_of(
        ("RE", "1", [("O", None, BASE + 5), ("V", BASE + 30, BASE + 32)]),
        ("RB", "2", [("V", None, BASE + 45), ("D", BASE + 70, None)]),
        ("RE", "3", [("O", None, BASE + 10), ("X", BASE + 25, None)]),
    )


def test_reachable_separates_direct_from_one_change() -> None:
    direct, witnesses = rb.reachable(one_change_network(), "O", BASE)
    assert direct == {"V", "X"}
    assert witnesses == {"D": {"V"}}


def test_reachable_excludes_what_is_already_direct() -> None:
    # A second train reaching D without a change removes D from the witnesses:
    # a journey the app solves directly cannot measure the transfer search.
    tt = tt_of(
        ("RE", "1", [("O", None, BASE + 5), ("V", BASE + 30, BASE + 32)]),
        ("RB", "2", [("V", None, BASE + 45), ("D", BASE + 70, None)]),
        ("RE", "9", [("O", None, BASE + 8), ("D", BASE + 50, None)]),
    )
    direct, witnesses = rb.reachable(tt, "O", BASE)
    assert "D" in direct
    assert witnesses == {}


def test_reachable_ignores_staying_on_the_same_train() -> None:
    """Boarding the train you are already on is not a change."""
    tt = tt_of(("RE", "1", [("O", None, BASE + 5), ("V", BASE + 30, BASE + 32),
                            ("D", BASE + 70, None)]))
    direct, witnesses = rb.reachable(tt, "O", BASE)
    assert direct == {"V", "D"}
    assert witnesses == {}


def test_reachable_needs_the_transfer_minutes() -> None:
    """A connection leaving less than TRANSFER_MINUTES after arrival is unusable."""
    def net(onward_dep: int) -> rb.Timetable:
        return tt_of(
            ("RE", "1", [("O", None, BASE + 5), ("V", BASE + 30, BASE + 32)]),
            ("RB", "2", [("V", None, onward_dep), ("D", onward_dep + 20, None)]),
        )
    tight = BASE + 30 + rb.TRANSFER_MINUTES - 1
    assert rb.reachable(net(tight), "O", BASE)[1] == {}
    assert rb.reachable(net(tight + 1), "O", BASE)[1] == {"D": {"V"}}


def test_reachable_drops_long_distance_feeders() -> None:
    tt = tt_of(("ICE", "1", [("O", None, BASE + 5), ("V", BASE + 30, BASE + 32)]),
               ("RB", "2", [("V", None, BASE + 45), ("D", BASE + 70, None)]))
    assert rb.reachable(tt, "O", BASE) == (set(), {})


# --- candidate ranking -------------------------------------------------------


def ranking_setup() -> tuple[dict[str, rb.Station], rb.Station, rb.Station, tuple]:
    """Destination 1 degree of latitude south; a hub twice that far north."""
    origin, dest = station("O", 500, 49.0), station("D", 60, 48.0)
    by_eva = {
        "O": origin, "D": dest,
        "HUB": station("HUB", 1000, 51.0),   # big, and clearly the wrong way
        "NEAR": station("NEAR", 50, 48.2),   # small, but nearly there
        "MID": station("MID", 300, 48.6),    # decent size, right direction
        "HALT": station("HALT", 5, 48.1),    # below the weight floor
    }
    path = tuple((eva, BASE + 20) for eva in ("HUB", "NEAR", "MID", "HALT", "D"))
    return by_eva, origin, dest, path


def test_candidates_drop_the_destination_and_the_light_halts() -> None:
    by_eva, origin, dest, path = ranking_setup()
    names = [s.eva for s in rb.candidates(path, origin, dest, set(), by_eva, rb.Config())]
    assert "D" not in names, "changing at the destination is not a transfer"
    assert "HALT" not in names, "below MIN_TRANSFER_WEIGHT"


def test_distance_ranking_drops_the_wrong_direction_and_sorts_by_km() -> None:
    by_eva, origin, dest, path = ranking_setup()
    got = rb.candidates(path, origin, dest, set(), by_eva, rb.Config())
    assert [s.eva for s in got] == ["NEAR", "MID"]


def test_weight_ranking_keeps_the_wrong_direction_first() -> None:
    """Pre-0.1.2 behaviour, and the reason Ulm → Türkheim went via Stuttgart."""
    by_eva, origin, dest, path = ranking_setup()
    cfg = rb.Config(ranking="weight")
    assert [s.eva for s in rb.candidates(path, origin, dest, set(), by_eva, cfg)][0] == "HUB"


def test_hybrid_filters_by_detour_then_prefers_the_junction() -> None:
    by_eva, origin, dest, path = ranking_setup()
    cfg = rb.Config(ranking="hybrid")
    got = [s.eva for s in rb.candidates(path, origin, dest, set(), by_eva, cfg)]
    assert got == ["MID", "NEAR"], "the wrong way is dropped, then size decides"


def test_candidates_skip_stations_already_tried() -> None:
    by_eva, origin, dest, path = ranking_setup()
    tried = {by_eva["NEAR"].name}
    got = rb.candidates(path, origin, dest, tried, by_eva, rb.Config())
    assert [s.eva for s in got] == ["MID"]


# --- the replay --------------------------------------------------------------


def test_search_finds_the_connection_and_reports_its_attempt() -> None:
    by_eva, origin, dest, _ = ranking_setup()
    tt = tt_of(
        ("RE", "1", [("O", None, BASE + 5), ("MID", BASE + 30, None)]),
        ("RB", "2", [("MID", None, BASE + 45), ("D", BASE + 70, None)]),
    )
    got = rb.search(tt, origin, dest, BASE, by_eva, rb.Config(), budget=8)
    assert got["first_hit"] == 1 and got["direct"] == 0
    assert got["first_arrival"] == BASE + 70


def test_search_reports_the_arrival_of_the_first_catchable_train() -> None:
    """Two onward trains: the itinerary is the earlier one, not the best one."""
    by_eva, origin, dest, _ = ranking_setup()
    tt = tt_of(
        ("RE", "1", [("O", None, BASE + 5), ("MID", BASE + 30, None)]),
        ("RB", "2", [("MID", None, BASE + 45), ("D", BASE + 90, None)]),
        ("RB", "3", [("MID", None, BASE + 50), ("D", BASE + 70, None)]),
    )
    got = rb.search(tt, origin, dest, BASE, by_eva, rb.Config(), budget=8)
    assert got["first_arrival"] == BASE + 90, "boarding the first train that runs"


def test_search_counts_the_direct_trains_separately() -> None:
    by_eva, origin, dest, _ = ranking_setup()
    tt = tt_of(("RE", "9", [("O", None, BASE + 5), ("D", BASE + 50, None)]))
    got = rb.search(tt, origin, dest, BASE, by_eva, rb.Config(), budget=8)
    assert got == {"direct": 1, "first_hit": None, "attempts": 0,
                   "first_arrival": None, "feeders": 1}


def test_search_stops_at_the_budget() -> None:
    """Ten feeders, each offering one useless transfer; the last one works."""
    by_eva, origin, dest, _ = ranking_setup()
    by_eva |= {f"U{i}": station(f"U{i}", 100, 48.5, lon=10.0 + i * 0.01)
               for i in range(10)}
    rides = [("RE", str(i), [("O", None, BASE + i), (f"U{i}", BASE + 20, None)])
             for i in range(10)]
    # Only the last transfer has an onward train to the destination.
    rides.append(("RB", "99", [("U9", None, BASE + 30), ("D", BASE + 60, None)]))
    tt = tt_of(*rides)
    assert rb.search(tt, origin, dest, BASE, by_eva, rb.Config(), budget=3)["first_hit"] is None
    assert rb.search(tt, origin, dest, BASE, by_eva, rb.Config(), budget=32)["first_hit"] == 10


def test_search_needs_the_feeder_to_reach_the_transfer_in_the_window() -> None:
    """A feeder arriving after the 4-hour transfer board is a dead attempt."""
    by_eva, origin, dest, _ = ranking_setup()
    late = BASE + rb.TRANSFER_HOURS * 60 + 30
    tt = tt_of(
        ("RE", "1", [("O", None, BASE + 5), ("MID", late, late + 2)]),
        ("RB", "2", [("MID", None, late + 20), ("D", late + 60, None)]),
    )
    got = rb.search(tt, origin, dest, BASE, by_eva, rb.Config(), budget=8)
    assert got["attempts"] == 1 and got["first_hit"] is None


# --- miss diagnosis ----------------------------------------------------------


def test_diagnose_names_the_binding_constraint() -> None:
    by_eva, origin, dest, _ = ranking_setup()
    base = {"from": "O", "to": "D"}
    # "GONE" is one of the 3% of archive stations missing from the app's list.
    assert "not in station list" in rb.diagnose(base | {"vias": ["GONE"]}, by_eva, rb.Config())
    assert "weight" in rb.diagnose(base | {"vias": ["HALT"]}, by_eva, rb.Config())
    assert "detour" in rb.diagnose(base | {"vias": ["HUB"]}, by_eva, rb.Config())
    assert "never ranked" in rb.diagnose(base | {"vias": ["MID"]}, by_eva, rb.Config())


# --- the snapshot ------------------------------------------------------------


def test_snapshot_reconstructs_routes_and_drops_cancellations(tmp_path: Path) -> None:
    day = dt.date(2026, 6, 10)
    noon = dt.datetime.combine(day, dt.time(12))
    rows = []
    for seq, (eva, minute) in enumerate([("08000170", 0), ("08000013", 30)]):
        rows.append({
            "eva": eva, "train_type": "RE", "train_number": "42",
            "train_line_ride_id": "ride-a", "train_line_station_num": seq,
            "arrival_planned_time": noon + dt.timedelta(minutes=minute) if seq else None,
            "departure_planned_time": noon + dt.timedelta(minutes=minute + 2),
            "is_canceled": False,
        })
    rows.append({**rows[0], "train_line_ride_id": "ride-b", "is_canceled": True})
    # A stop on the day before must not leak into the snapshot.
    rows.append({**rows[0], "train_line_ride_id": "ride-c",
                 "arrival_planned_time": noon - dt.timedelta(days=1),
                 "departure_planned_time": noon - dt.timedelta(days=1)})
    schema = {"eva": pl.String, "train_type": pl.String, "train_number": pl.String,
              "train_line_ride_id": pl.String, "train_line_station_num": pl.Int32,
              "arrival_planned_time": pl.Datetime("ns"),
              "departure_planned_time": pl.Datetime("ns"), "is_canceled": pl.Boolean}
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(rows, schema=schema).write_parquet(data / "data-2026-06.parquet")

    out = tmp_path / "snap.parquet"
    rb.snapshot([data], day, out)
    tt = rb.load(out)

    assert len(tt.rides) == 1, "the cancelled and the previous-day ride are gone"
    (cat, num, stops), = tt.rides.values()
    assert (cat, num) == ("RE", "42")
    assert [s[0] for s in stops] == ["8000170", "8000013"], "EVAs lose their padding"
    stop = tt.board("8000170", rb.wall_minutes(noon), 1)[0]
    assert stop.path == (("8000013", rb.wall_minutes(noon + dt.timedelta(minutes=30))),)


@pytest.mark.parametrize("tz", ["Europe/Berlin", "UTC", "America/New_York"])
def test_query_times_are_wall_clock_whatever_the_machine_thinks(tz, monkeypatch) -> None:
    """A local-timezone conversion shifted every benchmarked departure by the
    UTC offset, so `--times 08:00` measured 06:00 in summer."""
    monkeypatch.setenv("TZ", tz)
    if hasattr(time, "tzset"):
        time.tzset()
    eight = rb.wall_minutes(dt.datetime(2026, 6, 10, 8, 0))
    assert eight % 60 == 0
    # 08:00 must be exactly eight hours after the day's own midnight.
    assert eight - rb.wall_minutes(dt.datetime(2026, 6, 10)) == 8 * 60


def test_snapshot_refuses_a_day_outside_the_archive(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    schema = {"eva": pl.String, "train_type": pl.String, "train_number": pl.String,
              "train_line_ride_id": pl.String, "train_line_station_num": pl.Int32,
              "arrival_planned_time": pl.Datetime("ns"),
              "departure_planned_time": pl.Datetime("ns"), "is_canceled": pl.Boolean}
    pl.DataFrame([], schema=schema).write_parquet(data / "data-2026-06.parquet")
    with pytest.raises(SystemExit):
        rb.snapshot([data], dt.date(1999, 1, 1), tmp_path / "out.parquet")


def test_shipped_station_list_parses() -> None:
    by_eva = rb.stations()
    assert by_eva["8000170"].name == "Ulm Hbf"
    assert by_eva["8000144"].weight >= rb.MIN_TRANSFER_WEIGHT
    assert 40 < by_eva["8000170"].lat < 55


def test_only_the_relevant_archive_files_are_opened(tmp_path: Path) -> None:
    """Extracting one day used to scan every monthly file — ~5 GB, and enough
    to get the process killed."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("data-2025-11.parquet", "data-2026-03.parquet",
                 "data-2026-06.parquet", "data-recent-2026-06-11.parquet"):
        (data / name).touch()
    got = {f.name for f in rb.candidate_files([data], dt.date(2026, 3, 11))}
    assert got == {"data-2026-03.parquet", "data-recent-2026-06-11.parquet"}, (
        "the day's own month plus the daily cache, and nothing else"
    )


def test_a_day_no_archive_file_can_cover_is_refused(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "data-2026-06.parquet").touch()
    with pytest.raises(SystemExit, match="covering"):
        rb.snapshot([data], dt.date(2011, 1, 1), tmp_path / "out.parquet")


def test_recall_is_reported_per_origin_size() -> None:
    """The aggregate hides the structure: a headline number mixes village halts
    (~89% solved) with the big hubs (~52%), and most journeys start at hubs."""
    by_eva = {"S": station("S", 10, 48.0), "B": station("B", 900, 48.0),
              "D": station("D", 60, 48.5)}
    rows = [
        {"query": {"from": "S", "to": "D"}, "first_hit": 2, "feeders": 4},
        {"query": {"from": "S", "to": "D"}, "first_hit": 30, "feeders": 6},
        {"query": {"from": "B", "to": "D"}, "first_hit": None, "feeders": 40},
    ]
    got = rb.by_origin_size(rows, by_eva, budget=rb.MAX_TRANSFER_ATTEMPTS)
    assert [(label, n) for label, n, *_ in got] == [("0-40", 2), (">=250", 1)]
    small, big = got
    assert small[2] == 0.5, "one of the two is found within the budget"
    assert small[3] == 1.0, "but both are found eventually"
    assert big[2] == 0.0 and big[4] == 40, "the hub's haystack is reported"
