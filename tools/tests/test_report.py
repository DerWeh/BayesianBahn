"""Tests for the numbers the published report states.

report.py was written as presentation and went untested, which is backwards: it
is the file that turns scored events into claims, and a claim is exactly the kind
of thing that should not be able to change silently. The statistics here earn
their tests twice over, because a first pass at this evaluation read a single
evening's 189 missed connections as a finding — the mistake an interval exists to
prevent.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report as R  # noqa: E402
from score_events import wall_to_epoch  # noqa: E402


def arrival(*, cat="RE", num="1", crps=1.0, db=2, truth=2, q10=-1.0, q90=5.0,
            lead_minutes=30.0, planned_dep=8 * 60):
    """A scored arrival, with `read_at` derived so the lead time is exact."""
    return {
        "cat": cat, "num": num, "crps": crps, "db": db, "truth": truth,
        "q10": q10, "q90": q90,
        "planned": planned_dep + 5, "planned_dep": planned_dep,
        "archive": truth, "archive_dep": truth,
        "read_at": wall_to_epoch(planned_dep) - lead_minutes * 60,
    }


def connection(*, cat="RE", num="1", p_catch=0.5, db_catch=1, caught=True, slack=10):
    return {
        "cat": cat, "num": num, "p_catch": p_catch, "db_catch_p": db_catch,
        "caught": caught, "slack": slack, "crps": 0.0, "db": 0, "truth": 0,
        "q10": 0.0, "q90": 0.0, "planned": 0, "planned_dep": 0,
        "read_at": wall_to_epoch(0),
    }


# --- the bootstrap -----------------------------------------------------------


def test_the_point_estimate_is_just_the_mean():
    rows = [arrival(num=str(i), crps=float(i)) for i in range(10)]
    point, _, _ = R.cluster_ci(rows, lambda r: r["crps"])
    assert point == pytest.approx(sum(range(10)) / 10)


def test_an_interval_over_identical_values_collapses():
    rows = [arrival(num=str(i), crps=3.0) for i in range(20)]
    point, lo, hi = R.cluster_ci(rows, lambda r: r["crps"], draws=200)
    assert (point, lo, hi) == (3.0, 3.0, 3.0)


def test_the_same_seed_gives_the_same_interval():
    rows = [arrival(num=str(i), crps=float(i % 7)) for i in range(40)]
    assert R.cluster_ci(rows, lambda r: r["crps"], draws=200) == \
        R.cluster_ci(rows, lambda r: r["crps"], draws=200)


def test_clustering_by_train_widens_the_interval():
    """The whole reason the bootstrap is clustered.

    The same 60 values, first as 60 independent trains and then as 3 trains of
    20 correlated predictions each. Treating the correlated case as independent
    is what produces a falsely confident claim, so it must come out wider.
    """
    values = [0.0, 5.0, 10.0] * 20
    spread = [arrival(num=str(i), crps=v) for i, v in enumerate(values)]
    clumped = [arrival(num=str(i % 3), crps=v) for i, v in enumerate(values)]

    _, lo_spread, hi_spread = R.cluster_ci(spread, lambda r: r["crps"], draws=500)
    _, lo_clumped, hi_clumped = R.cluster_ci(clumped, lambda r: r["crps"], draws=500)
    assert hi_clumped - lo_clumped > hi_spread - lo_spread


def test_a_bracketing_interval_contains_the_mean():
    rows = [arrival(num=str(i), crps=float(i % 5)) for i in range(50)]
    point, lo, hi = R.cluster_ci(rows, lambda r: r["crps"], draws=500)
    assert lo <= point <= hi


# --- sign conventions -------------------------------------------------------
#
# Getting these backwards would invert the report's conclusion while leaving
# every number looking plausible.


def test_a_better_forecast_gives_a_negative_crps_gap():
    # We score 1.0; DB is 3 minutes out. Ours minus DB must be negative.
    rows = [arrival(num=str(i), crps=1.0, db=3, truth=0) for i in range(10)]
    point, _, _ = R.crps_gap(rows)
    assert point == pytest.approx(1.0 - 3.0)


def test_a_worse_forecast_gives_a_positive_crps_gap():
    rows = [arrival(num=str(i), crps=4.0, db=1, truth=0) for i in range(10)]
    assert R.crps_gap(rows)[0] > 0


def test_db_being_right_and_us_hedging_is_a_positive_brier_gap():
    # DB says "yes" and the connection was caught: DB scores 0, we score 0.25.
    rows = [connection(num=str(i), p_catch=0.5, db_catch=1, caught=True)
            for i in range(10)]
    assert R.brier_gap(rows)[0] == pytest.approx(0.25)


def test_db_being_wrong_and_us_hedging_is_a_negative_brier_gap():
    # DB says "yes" and the connection was missed: DB scores 1, we score 0.25.
    rows = [connection(num=str(i), p_catch=0.5, db_catch=1, caught=False)
            for i in range(10)]
    assert R.brier_gap(rows)[0] == pytest.approx(0.25 - 1.0)


def test_brier_of_a_confident_wrong_answer_is_one():
    assert R.brier([connection(p_catch=0.0, caught=True)], "p_catch") == 1.0


# --- the headline table -----------------------------------------------------


def test_a_gap_that_straddles_zero_is_not_claimed():
    """The guard that a first pass at this evaluation lacked."""
    live = [arrival(num=str(i), crps=2.0 + (i % 2) * 6, db=5, truth=0)
            for i in range(30)]
    conn = [connection(num=str(i), p_catch=0.5, db_catch=i % 2, caught=bool(i % 2))
            for i in range(30)]
    rows = R.headline(live, live, conn, conn)
    for row in rows:
        if row["lo"] <= 0 <= row["hi"]:
            assert row["verdict"] == "not separated"
        elif row["hi"] < 0:
            assert row["verdict"] == "we are better"
        else:
            assert row["verdict"] == "DB is better"


def test_the_headline_covers_both_variants_and_the_missed_subset():
    live = [arrival(num=str(i)) for i in range(5)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(5)]
    rows = R.headline(live, live, conn, conn)
    assert [r["what"] for r in rows] == [
        "Arrival time, as shipped", "Arrival time, history only",
        "Every connection, as shipped", "Every connection, history only",
        "Missed connections, as shipped", "Missed connections, history only",
    ]
    # The missed rows must see only the failures, not all five.
    assert {r["n"] for r in rows if "Missed" in r["what"]} == {1}


# --- per-day replication ----------------------------------------------------


def tagged(day, rows):
    """The rows as `main` hands them over: each carrying the day it came from."""
    return [{**r, "day": day} for r in rows]


def a_day(day, arrivals, connections):
    """The four lists for one day, in per_day's argument order."""
    return (tagged(day, arrivals), tagged(day, arrivals),
            tagged(day, connections), tagged(day, connections))


def days(*collected):
    """Concatenate several days' worth of a_day() output, column by column."""
    return [sum(lists, []) for lists in zip(*collected)]


def test_each_collected_day_gets_its_own_row():
    rows = R.per_day(["2026-08-17", "2026-08-18"],
                     *days(a_day("2026-08-17", [arrival(crps=1.0)],
                                 [connection(caught=False)]),
                           a_day("2026-08-18", [arrival(crps=2.0)],
                                 [connection(caught=False)])))
    assert [r["day"] for r in rows] == ["2026-08-17", "2026-08-18"]
    assert [r["live"] for r in rows] == [1.0, 2.0]


def test_a_later_day_is_not_filtered_away_by_an_earlier_one():
    """per_day narrows the pooled rows once per day. Assigning the result back
    to the same names would leave every day after the first empty."""
    rows = R.per_day(["2026-08-17", "2026-08-18", "2026-08-19"],
                     *days(a_day("2026-08-17", [arrival()], [connection()]),
                           a_day("2026-08-18", [arrival()], [connection()]),
                           a_day("2026-08-19", [arrival()], [connection()])))
    assert len(rows) == 3


def test_a_day_with_no_scored_events_is_skipped_not_zeroed():
    """A missing day must not appear as a day where everything scored zero."""
    rows = R.per_day(["2026-08-17", "2026-08-18"],
                     *days(a_day("2026-08-17", [arrival()], [connection()]),
                           a_day("2026-08-18", [], [])))
    assert [r["day"] for r in rows] == ["2026-08-17"]


def test_coverage_counts_the_truth_inside_the_stated_range():
    row = R.per_day(["2026-08-17"], *a_day("2026-08-17", [
        arrival(num="1", truth=0, q10=-1.0, q90=1.0),    # inside
        arrival(num="2", truth=9, q10=-1.0, q90=1.0),    # outside
    ], [connection()]))[0]
    assert row["live_cover"] == pytest.approx(0.5)


def test_a_day_with_no_missed_connections_reports_no_brier():
    row = R.per_day(["2026-08-17"],
                    *a_day("2026-08-17", [arrival()], [connection(caught=True)]))[0]
    assert row["missed"] == 0
    assert math.isnan(row["db_missed"])


# --- the pooled tables ------------------------------------------------------


def test_db_column_is_dbs_absolute_error():
    rows = [arrival(num=str(i), db=4, truth=1, lead_minutes=30.0) for i in range(4)]
    table = R.arrivals_table(rows, rows)
    assert len(table) == 1
    assert table[0]["db"] == pytest.approx(3.0)
    assert table[0]["db_bias"] == pytest.approx(3.0)


def test_lead_time_is_measured_from_the_planned_departure():
    """Binning against arrival instead of departure would silently reshuffle
    every row of the headline chart."""
    early = [arrival(num="1", lead_minutes=5.0)]
    late = [arrival(num="2", lead_minutes=120.0)]
    distant = [arrival(num="3", lead_minutes=400.0)]
    assert R.arrivals_table(early, early)[0]["bucket"] == "<10m"
    assert R.arrivals_table(late, late)[0]["bucket"] == "1.5-3h"
    assert R.arrivals_table(distant, distant)[0]["bucket"] == ">3h"


def test_a_surprise_is_an_arrival_later_than_db_promised():
    calm = [arrival(num="1", db=0, truth=1)]
    surprising = [arrival(num="2", db=0, truth=30)]
    assert R.arrivals_table(calm, calm)[0]["db_surprise"] == 0.0
    assert R.arrivals_table(surprising, surprising)[0]["db_surprise"] == 1.0


def test_the_outcome_split_separates_caught_from_missed():
    rows = [connection(num="1", caught=True), connection(num="2", caught=False),
            connection(num="3", caught=False)]
    split = R.outcome_split(rows, rows)
    assert [(r["outcome"], r["n"]) for r in split] == [
        ("Connection was caught", 1), ("Connection was missed", 2),
    ]


# --- rendering --------------------------------------------------------------


def test_the_page_renders_with_its_definitions_and_percent_signs(tmp_path):
    """The report body is a %-formatted string, so a literal percent in the
    prose has to be escaped; getting that wrong raises at render time."""
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-17", "2026-08-18"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn),
             daily=[{"day": "2026-08-17", "n": 2, "db": 1.0, "blind": 1.0,
                     "live": 1.0, "live_cover": 0.8, "blind_cover": 0.8,
                     "missed": 1, "db_missed": 1.0, "blind_missed": 0.5,
                     "live_missed": 0.5},
                    {"day": "2026-08-18", "n": 2, "db": 1.0, "blind": 1.0,
                     "live": 1.0, "live_cover": 0.8, "blind_cover": 0.8,
                     "missed": 1, "db_missed": 1.0, "blind_missed": 0.5,
                     "live_missed": 0.5}])
    page = out.read_text(encoding="utf-8")
    assert "72%" in page                      # the escaped literal
    assert "<dt>CRPS" in page
    assert "Does it hold from one day to the next?" in page
    assert "The answer, with its uncertainty" in page
    assert "%d" not in page and "%s" not in page


def test_a_single_day_hides_the_replication_section(tmp_path):
    """With one day there is nothing to replicate, and a one-row table would
    imply otherwise."""
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn),
             daily=[{"day": "2026-08-17", "n": 2, "db": 1.0, "blind": 1.0,
                     "live": 1.0, "live_cover": 0.8, "blind_cover": 0.8,
                     "missed": 1, "db_missed": 1.0, "blind_missed": 0.5,
                     "live_missed": 0.5}])
    assert "Does it hold from one day to the next?" not in out.read_text(encoding="utf-8")


# --- the hour-of-day curve ------------------------------------------------
#
# The chart these feed makes a causal claim (delay accumulates over the day),
# and the first draft of it read a two-train 02:00 as the worst hour on the
# network. Every guard below exists because of a way this section can lie.

def clock_row(*, day="2026-08-18", hour=12, truth=0, eva="1", num="1"):
    """One scored arrival whose planned arrival lands in `hour`, Berlin time."""
    return {**arrival(num=num, truth=truth), "day": day, "eva": eva,
            "planned": hour * 60 + 30}


def a_full_day(day, hour_means, *, per_hour=R.MIN_HOUR_EVENTS):
    """Enough rows to clear the sample floor, at the given mean per hour."""
    rows = []
    for hour, mean in hour_means.items():
        for i in range(per_hour):
            rows.append(clock_row(day=day, hour=hour, truth=mean,
                                  eva=str(hour), num=f"{hour}-{i}"))
    return rows


def test_repeated_polls_of_one_train_count_once():
    """A train sits in the collector's window for as long as its lead time, so
    counting polls would weight the curve by lead time, not by lateness."""
    hours = {h: 0 for h in range(24)}
    rows = a_full_day("2026-08-18", hours)
    # One train polled thirty more times, very late, in a single hour.
    rows += [{**clock_row(day="2026-08-18", hour=12, truth=60, eva="9", num="X")}
             for _ in range(30)]
    twelve = next(r for r in R.hourly(rows) if r["hour"] == 12)
    assert twelve["n"] == R.MIN_HOUR_EVENTS + 1
    assert twelve["mean"] == pytest.approx(60 / (R.MIN_HOUR_EVENTS + 1))


def test_an_hour_with_too_few_trains_is_dropped():
    """Two trains at 02:00 produced the largest mean on the first chart."""
    rows = a_full_day("2026-08-18", {h: 1 for h in range(24)})
    rows = [r for r in rows if R.hour_of(r) != 2][:]
    rows += [clock_row(day="2026-08-18", hour=2, truth=90, num=f"n{i}", eva="2")
             for i in range(2)]
    assert 2 not in {r["hour"] for r in R.hourly(rows)}


def test_an_hour_exactly_at_the_floor_is_kept():
    rows = a_full_day("2026-08-18", {h: 1 for h in range(24)})
    assert len(R.hourly(rows)) == 24


def test_a_partial_day_is_excluded_from_the_curve():
    """The first collected day starts in the evening; averaging it in would put
    a busy evening against a full day's whole clock."""
    full = a_full_day("2026-08-18", {h: 2 for h in range(24)})
    evening = a_full_day("2026-08-17", {h: 20 for h in range(19, 24)})
    rows = R.hourly(full + evening)
    assert all(r["mean"] == pytest.approx(2) for r in rows)


def test_no_full_day_means_no_curve_at_all():
    """Better an absent section than one drawn from a single evening."""
    assert R.hourly(a_full_day("2026-08-17", {h: 2 for h in range(19, 24)})) == []


def test_the_late_share_counts_arrivals_past_the_surprise_threshold():
    rows = a_full_day("2026-08-18", {h: 0 for h in range(24)})
    rows += [clock_row(day="2026-08-18", hour=9, truth=R.SURPRISE_MINUTES + 1,
                       num=f"late{i}", eva="9") for i in range(R.MIN_HOUR_EVENTS)]
    nine = next(r for r in R.hourly(rows) if r["hour"] == 9)
    assert nine["late"] == pytest.approx(0.5)


def test_band_spread_is_the_distance_between_the_hourly_extremes():
    rows = [{"hour": 9, "mean": 2.0}, {"hour": 10, "mean": 5.0},
            {"hour": 11, "mean": 3.0}]
    band = R.band_spread(rows, {"B": ("09-11", range(9, 12))})[0]
    assert (band["lo"], band["hi"], band["spread"]) == (2.0, 5.0, 3.0)


def test_a_band_whose_hours_were_all_dropped_is_skipped():
    """The night band can lose every one of its hours to the sample floor."""
    assert R.band_spread([{"hour": 9, "mean": 2.0}],
                         {"NIGHT": ("02-03", range(2, 4))}) == []


def test_a_band_uses_only_the_hours_that_survived():
    rows = [{"hour": 20, "mean": 4.0}]
    band = R.band_spread(rows, {"NIGHT": ("19-05", tuple(range(19, 24)))})[0]
    assert band["spread"] == 0.0 and band["hours"] == "19-05"


def test_the_reported_bands_match_the_ones_the_app_uses():
    """SHIPPED_BANDS is a third mirror of the model, and the other two have both
    drifted before. The report must not describe a cut the app does not use."""
    source = (Path(__file__).resolve().parents[2] / "app/src/main/java/io/github"
              / "derweh/bayesianbahn/model/DelayModel.kt").read_text(encoding="utf-8")
    body = source.split("fun fromEpochMillis")[1].split("}")[0]
    from_kotlin = {}
    for lo, hi, name in re.findall(r"in (\d+)\.\.(\d+) -> (\w+)", body):
        from_kotlin[name] = set(range(int(lo), int(hi) + 1))
    else_band = re.search(r"else -> (\w+)", body).group(1)
    from_kotlin[else_band] = set(range(24)) - set().union(*from_kotlin.values())

    ours = {name: set(hours) for name, (_, hours) in R.SHIPPED_BANDS.items()}
    assert ours == from_kotlin


def test_the_page_carries_the_hour_of_day_section(tmp_path):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    clock = R.hourly(a_full_day("2026-08-18", {h: h / 4 for h in range(24)}))
    R.render(["2026-08-18"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn), clock=clock)
    page = out.read_text(encoding="utf-8")
    assert "Does the day pile up?" in page
    assert "MORNING_PEAK" in page
    assert "%d" not in page and "%s" not in page


def test_the_page_omits_the_hour_section_when_no_day_is_full(tmp_path):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn), clock=[])
    assert "Does the day pile up?" not in out.read_text(encoding="utf-8")
