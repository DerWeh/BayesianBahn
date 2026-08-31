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
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]

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
    point, _, _ = R.cluster_ci(rows, pl.col("crps"))
    assert point == pytest.approx(sum(range(10)) / 10)


def test_an_interval_over_identical_values_collapses():
    rows = [arrival(num=str(i), crps=3.0) for i in range(20)]
    point, lo, hi = R.cluster_ci(rows, pl.col("crps"), draws=200)
    assert (point, lo, hi) == (3.0, 3.0, 3.0)


def test_the_same_seed_gives_the_same_interval():
    rows = [arrival(num=str(i), crps=float(i % 7)) for i in range(40)]
    assert R.cluster_ci(rows, pl.col("crps"), draws=200) == \
        R.cluster_ci(rows, pl.col("crps"), draws=200)


def test_clustering_by_train_widens_the_interval():
    """The whole reason the bootstrap is clustered.

    The same 60 values, first as 60 independent trains and then as 3 trains of
    20 correlated predictions each. Treating the correlated case as independent
    is what produces a falsely confident claim, so it must come out wider.
    """
    values = [0.0, 5.0, 10.0] * 20
    spread = [arrival(num=str(i), crps=v) for i, v in enumerate(values)]
    clumped = [arrival(num=str(i % 3), crps=v) for i, v in enumerate(values)]

    _, lo_spread, hi_spread = R.cluster_ci(spread, pl.col("crps"), draws=500)
    _, lo_clumped, hi_clumped = R.cluster_ci(clumped, pl.col("crps"), draws=500)
    assert hi_clumped - lo_clumped > hi_spread - lo_spread


def test_a_bracketing_interval_contains_the_mean():
    rows = [arrival(num=str(i), crps=float(i % 5)) for i in range(50)]
    point, lo, hi = R.cluster_ci(rows, pl.col("crps"), draws=500)
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
            assert row["verdict"] == "BayesianBahn lower"
        else:
            assert row["verdict"] == "DB lower"


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
    assert "The comparison, with its uncertainty" in page
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
    rows = [r for r in rows if (r["planned"] // 60) % 24 != 2]
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
    assert "Delay through the day" in page
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
    assert "Delay through the day" not in out.read_text(encoding="utf-8")


# --- the spread section -----------------------------------------------------
#
# The mean CRPS is the number every other section reports, and on its own it
# tells the reader the wrong thing twice: the medians are a tie, so it overstates
# the everyday difference, and the tail is where the whole advantage sits, so it
# understates the difference that costs a passenger a connection.

def graded(*values, **kwargs):
    """Scored arrivals whose CRPS is exactly the given values."""
    return [arrival(num=str(i), crps=float(v), **kwargs) for i, v in enumerate(values)]


def test_the_spread_reports_quantiles_not_just_the_mean():
    s = R.score_spread(list(range(1, 101)))
    assert s["n"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["p50"] == pytest.approx(50.5)
    assert s["p90"] == pytest.approx(90.1)
    assert s["p99"] == pytest.approx(99.01)
    assert s["max"] == pytest.approx(100.0)


def test_the_quantiles_come_out_in_order():
    s = R.score_spread(list(range(1, 101)))
    ordered = [s[k] for k in ("p10", "p25", "p50", "p75", "p90", "p99", "max")]
    assert ordered == sorted(ordered)


def test_a_mean_can_hide_a_tail():
    """The case the section exists for: same mean, very different tails."""
    flat = R.score_spread([5.0] * 100)
    spiky = R.score_spread([0.0] * 90 + [50.0] * 10)
    assert flat["mean"] == pytest.approx(spiky["mean"])
    assert flat["awful"] == 0.0
    assert spiky["awful"] == pytest.approx(0.10)


def test_the_bad_shares_count_strictly_over_the_threshold():
    s = R.score_spread([R.BAD_MINUTES, R.BAD_MINUTES + 0.1, R.AWFUL_MINUTES, R.AWFUL_MINUTES + 0.1])
    assert s["bad"] == pytest.approx(0.75)      # three are over BAD_MINUTES
    assert s["awful"] == pytest.approx(0.25)    # one is over AWFUL_MINUTES


def test_the_spread_is_ordered_by_lead_time_not_by_dict_order():
    rows = (graded(1.0, lead_minutes=200.0) + graded(2.0, lead_minutes=5.0)
            + graded(3.0, lead_minutes=30.0))
    assert [r["bucket"] for r in R.error_spread(rows)] == ["<10m", "20-45m", ">3h"]


def test_the_spread_scores_db_by_its_absolute_error():
    rows = [arrival(db=10, truth=4, crps=1.0)]
    assert R.error_spread(rows)[0]["db"]["p50"] == pytest.approx(6.0)


def test_the_box_chart_draws_a_box_per_series_per_bucket():
    rows = R.error_spread(graded(1.0, 2.0, 3.0, 4.0))
    svg = R.box_chart(rows, ["db", "live"], y_label="minutes")
    assert svg.count("<rect") == 2 * len(rows)
    assert "</svg>" in svg


def test_the_box_chart_clamps_a_tail_that_would_flatten_it():
    """p99 is several times p90 here; letting it set the scale would squash
    every box into a line."""
    rows = R.error_spread(graded(*([1.0] * 99 + [400.0])))
    svg = R.box_chart(rows, ["db", "live"], y_label="minutes")
    heights = [float(h) for h in re.findall(r'<rect[^>]*height="([0-9.]+)"', svg)]
    assert all(h > 0 for h in heights)


def test_the_page_carries_the_spread_section(tmp_path):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i), crps=float(i)) for i in range(8)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-18"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 8, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn), spread=R.error_spread(rows))
    page = out.read_text(encoding="utf-8")
    assert "How the errors are distributed" in page
    assert f"over {R.AWFUL_MINUTES} min" in page
    assert "%d" not in page and "%s" not in page


def test_the_weekday_caveat_names_what_the_days_cover():
    """Traffic and the timetable differ at weekends, so a page drawn only from
    weekdays has to say so — the hour-of-day curve especially."""
    assert R.weekday_caveat(["2026-08-17", "2026-08-21"]) == "Weekdays only."     # Mon, Fri
    assert R.weekday_caveat(["2026-08-22", "2026-08-23"]) == "Weekend days only."  # Sat, Sun
    assert R.weekday_caveat(["2026-08-21", "2026-08-22"]).startswith("Weekdays and")


def test_the_weekday_caveat_reaches_the_page(tmp_path):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-17", "2026-08-18"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn))
    assert "Weekdays only." in out.read_text(encoding="utf-8")


# --- provenance -------------------------------------------------------------
#
# The five-day page was generated from a working tree whose model was in no
# released version, and it described that model as "the one that ships".

def test_provenance_reads_the_app_version_and_commit():
    prov = R.provenance()
    assert re.fullmatch(r"[0-9a-f]{40}", prov["commit"]), prov["commit"]
    assert prov["short"] == prov["commit"][:12]
    assert re.fullmatch(r"\d+\.\d+\.\d+", prov["version"]), prov["version"]
    assert prov["code"].isdigit()


def test_only_model_and_scoring_files_count_as_dirty():
    """An edited README does not change a number on the page; an edited
    Predictor.kt does."""
    source = (R.ROOT / "tools/report.py").read_text(encoding="utf-8")
    body = source.split("def provenance(")[1].split("\ndef ")[0]
    assert 'startswith(("app/src/main/", "tools/", "pipeline/"))' in body


def fake_prov(**over):
    """A complete provenance dict. One place to keep in step with the real one:
    `render` reads every key directly, so a test that builds its own by hand
    starts failing with a KeyError the day a key is added."""
    return {"commit": "a" * 40, "short": "a" * 12, "version": "0.1.4",
            "code": "5", "tag": "", "release": "", "dirty": [], **over}


def test_the_fake_provenance_has_the_keys_the_real_one_has():
    assert set(fake_prov()) == set(R.provenance())


def test_an_unreleased_commit_is_labelled_as_such(tmp_path, monkeypatch):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    monkeypatch.setattr(R, "provenance", lambda: fake_prov())
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn))
    page = out.read_text(encoding="utf-8")
    assert "not a released version" in page
    assert "a" * 12 in page and "0.1.4" in page
    assert "the one that ships" not in page


def test_a_released_commit_names_its_tag(tmp_path, monkeypatch):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    monkeypatch.setattr(R, "provenance", lambda: {
        "commit": "b" * 40, "short": "b" * 12, "version": "0.2.0",
        "code": "6", "tag": "v0.2.0", "dirty": [],
    })
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn))
    page = out.read_text(encoding="utf-8")
    assert "released tag" in page and "v0.2.0" in page
    assert "not a released version" not in page


def test_a_dirty_tree_says_the_figures_are_not_reproducible(tmp_path, monkeypatch):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    monkeypatch.setattr(R, "provenance", lambda: {
        "commit": "c" * 40, "short": "c" * 12, "version": "0.2.0",
        "code": "6", "tag": "v0.2.0", "dirty": ["app/src/main/java/X.kt"],
    })
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn))
    page = out.read_text(encoding="utf-8")
    assert "does not identify the code that ran" in page
    assert "released tag" not in page


def test_the_full_commit_is_in_the_footer(tmp_path, monkeypatch):
    """The short form is for reading; the full hash is what a reader checks out."""
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    monkeypatch.setattr(R, "provenance", lambda: fake_prov(
        commit="d" * 40, short="d" * 12, version="0.2.0", code="6"))
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn))
    assert "d" * 40 in out.read_text(encoding="utf-8")


def test_the_local_hour_matches_the_wall_clock_it_is_taken_from():
    """`planned` is naive German local time, so the hour is arithmetic. The
    row-wise version converted to epoch and back; these must agree, including
    across midnight."""
    import polars as pl
    from score_events import BERLIN, wall_to_epoch
    import datetime as dt

    minutes = list(range(0, 60 * 24 * 3, 37))
    direct = (pl.DataFrame({"planned": minutes})
              .select(R.LOCAL_HOUR.alias("h"))["h"].to_list())
    roundtrip = [dt.datetime.fromtimestamp(wall_to_epoch(m), BERLIN).hour for m in minutes]
    assert direct == roundtrip


# --- working days against the weekend ----------------------------------------
#
# 2026-08-21 is a Friday and 2026-08-22 a Saturday. polars numbers weekdays from
# 1 = Monday, so the boundary is off by one from the 0-based convention used
# elsewhere in this pipeline — and getting it wrong would file Friday, the
# busiest day in the data, under "weekend" without changing any total.


def test_the_week_is_split_at_the_right_day():
    rows = R.week_split(*days(a_day("2026-08-21", [arrival(crps=1.0)], [connection()]),
                              a_day("2026-08-22", [arrival(crps=2.0)], [connection()])))
    assert [(r["part"], r["n"]) for r in rows] == [
        ("Monday to Friday", 1), ("Saturday and Sunday", 1)]


def test_sunday_is_the_weekend_too():
    """The other end of the boundary: 2026-08-23 is a Sunday, and polars puts
    it at 7, past both `>= 6` and any 0-based reading of the same rule."""
    rows = R.week_split(*days(a_day("2026-08-23", [arrival()], [connection()])))
    assert [r["part"] for r in rows] == ["Saturday and Sunday"]


def test_monday_is_a_working_day():
    rows = R.week_split(*days(a_day("2026-08-17", [arrival()], [connection()])))
    assert [r["part"] for r in rows] == ["Monday to Friday"]


def test_a_week_with_only_one_kind_of_day_yields_one_row():
    """The section is rendered only when there are two; a single row would
    invite reading a weekday figure as if it covered the week."""
    rows = R.week_split(*days(a_day("2026-08-17", [arrival()], [connection()]),
                              a_day("2026-08-18", [arrival()], [connection()])))
    assert len(rows) == 1


def test_the_split_reports_dbs_own_error_not_the_gap():
    """The column exists to show how much there is to improve on, so it is DB's
    absolute error — not a difference, which would be near zero for both."""
    rows = R.week_split(*days(a_day("2026-08-17", [arrival(db=2, truth=6)],
                                    [connection()])))
    assert rows[0]["db"] == pytest.approx(4.0)


def test_each_part_is_scored_against_its_own_days_only():
    """Not a filter applied to an already-pooled gap: the weekend figure has to
    come from weekend rows, or the quieter half borrows the busier one's."""
    rows = R.week_split(*days(
        a_day("2026-08-17", [arrival(crps=1.0, db=2, truth=2)], [connection()]),
        a_day("2026-08-22", [arrival(crps=5.0, db=2, truth=2)], [connection()])))
    gaps = {r["part"]: r["live"][0] for r in rows}
    assert gaps["Monday to Friday"] == pytest.approx(1.0)
    assert gaps["Saturday and Sunday"] == pytest.approx(5.0)


def test_a_part_of_the_week_with_no_missed_connection_says_so():
    """A Brier gap over an empty set is not zero, and zero would read as a tie."""
    rows = R.week_split(*days(a_day("2026-08-17", [arrival()],
                                    [connection(caught=True)])))
    assert rows[0]["missed_n"] == 0 and rows[0]["missed"] is None


# --- the box plot has to be readable, not merely correct ---------------------
#
# DB answers in whole minutes, so its per-event error piles onto the integers:
# in the collected data p25 is 0.00 and p50 and p75 are both 1.00 in the <10m
# bucket. A median drawn only across the box then lands exactly on the box's own
# rounded top edge and disappears, leaving the reader with the boxes alone — and
# DB's box reaches further down (to zero, which only a point forecast can reach)
# while its median is the higher of the two. The chart then says the opposite of
# the table underneath it. Everything below guards that reading.


def median_lines(svg: str) -> list[tuple[float, float, float]]:
    """(x1, x2, y) of every mark drawn in ink — the medians."""
    out = []
    for m in re.finditer(r'<line x1="([0-9.]+)" x2="([0-9.]+)" y1="([0-9.]+)" '
                         r'y2="\3" stroke="var\(--ink\)"', svg):
        out.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def boxes(svg: str) -> list[tuple[float, float]]:
    """(x, width) of every box rect."""
    return [(float(x), float(w)) for x, w in
            re.findall(r'<rect x="([0-9.]+)"[^>]*width="([0-9.]+)"', svg)]


def tied_at_the_hinge():
    """A bucket where DB's median equals its own p75, as it does at <10m.

    A quarter exactly right, most one minute out, the rest four: enough ties on
    the integer 1 that both the 50th and the 75th percentile land on it.
    """
    def db_for(i):
        return 2 if i < 25 else (3 if i < 85 else 6)
    return R.error_spread([arrival(num=str(i), db=db_for(i), truth=2,
                                   crps=0.4 + i / 50)
                           for i in range(100)])


def test_the_median_is_drawn_even_when_it_lands_on_a_box_edge():
    rows = tied_at_the_hinge()
    assert rows[0]["db"]["p50"] == rows[0]["db"]["p75"], "the case being guarded"
    svg = R.box_chart(rows, ["db", "live"], y_label="minutes")
    assert len(median_lines(svg)) == 2 * len(rows)


def test_the_median_overhangs_its_box_on_both_sides():
    """The overhang is what survives the coincidence: a line the width of the
    box, sitting on the box's edge, is the edge."""
    svg = R.box_chart(tied_at_the_hinge(), ["db", "live"], y_label="minutes")
    for (x1, x2, _), (bx, bw) in zip(median_lines(svg), boxes(svg)):
        assert x1 < bx and x2 > bx + bw


def test_the_median_is_not_drawn_in_the_surface_colour():
    """It used to be, which reads against a fill and vanishes against the page
    — and the page is exactly what is behind it when it sits on an edge."""
    svg = R.box_chart(tied_at_the_hinge(), ["db", "live"], y_label="minutes")
    assert 'stroke="var(--panel)"' not in svg


def test_only_a_point_forecast_can_score_zero():
    """The asymmetry the caption now states: DB names one minute and can be
    exactly right; a distribution always pays for its spread."""
    assert R.score_spread([0.0, 0.0, 1.0, 2.0])["zeros"] == pytest.approx(0.5)
    assert R.score_spread([0.2, 0.5, 1.0])["zeros"] == 0.0


def test_the_spread_counts_how_often_db_scores_lower():
    """Slightly more than half, while its mean is higher — the sentence that
    stops the chart being read as a contradiction of the table."""
    rows = R.error_spread([arrival(num="1", db=2, truth=2, crps=1.0),
                           arrival(num="2", db=2, truth=2, crps=1.0),
                           arrival(num="3", db=9, truth=2, crps=1.0)])
    assert rows[0]["db_wins"] == pytest.approx(2 / 3)


def test_a_tie_counts_for_neither_forecast():
    rows = R.error_spread([arrival(num="1", db=3, truth=2, crps=1.0)])
    assert rows[0]["db_wins"] == 0.0


def test_the_page_states_the_zero_score_asymmetry(tmp_path):
    """Without it the bottom of the boxes is read as a like-for-like race."""
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i), db=2, truth=2 + (i > 5), crps=float(i) / 4)
            for i in range(8)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": len(rows), "connections": len(conn)}, out,
             gaps=R.headline(rows, rows, conn, conn),
             spread=R.error_spread(rows))
    page = out.read_text(encoding="utf-8")
    assert "A distribution cannot score zero" in page
    assert "DB lower on the prediction" in page


# --- provenance: the claim that a figure is traceable ------------------------
#
# This block is the page's only defence against publishing numbers nobody can
# reproduce, and both failures it had were silent. `git status --porcelain`
# puts the status in columns 1-2, so `.strip()` on the whole output shifted the
# first line left by one and dropped that file from the dirty list — with a
# single uncommitted file, the page announced a clean tree. And `git diff
# --quiet` reports "differs" through exit code 1, which the helper swallows
# into the same empty string it returns for "identical", so a check written
# with it passes whatever the trees hold.


def repo(tmp_path: Path, version: str = "0.2.0") -> Path:
    """A git repository shaped like this one: app code, a tag, a generator."""
    def run(*args):
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "app/src/main").mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    (tmp_path / "app/build.gradle.kts").write_text(
        f'versionCode = 6\nversionName = "{version}"\n', encoding="utf-8")
    (tmp_path / "app/src/main/Model.kt").write_text("fun predict() = 1\n",
                                                    encoding="utf-8")
    # Tracked from the start: git collapses a wholly untracked directory to
    # `?? tools/`, which would make the test agree with a bug it is not about.
    (tmp_path / "tools/report.py").write_text("# generator\n", encoding="utf-8")
    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "T")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "release")
    run("git", "tag", f"v{version}")
    return tmp_path


def commit_a_generator_change(path: Path) -> None:
    """A commit that moves HEAD past the tag without touching the model."""
    def run(*args):
        subprocess.run(args, cwd=path, check=True, capture_output=True)
    (path / "tools/report.py").write_text("# regenerated\n", encoding="utf-8")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "re-render")


def test_a_single_uncommitted_file_is_still_reported(tmp_path, monkeypatch):
    """The exact shape of the bug: one modified file, and it is the first line
    of `git status --porcelain`, which is where the strip did its damage."""
    monkeypatch.setattr(R, "ROOT", repo(tmp_path))
    (tmp_path / "app/src/main/Model.kt").write_text("fun predict() = 2\n",
                                                    encoding="utf-8")
    assert R.provenance()["dirty"] == ["app/src/main/Model.kt"]


def test_every_uncommitted_file_is_reported_not_all_but_one(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", repo(tmp_path))
    (tmp_path / "app/src/main/Model.kt").write_text("x\n", encoding="utf-8")
    (tmp_path / "tools/report.py").write_text("y\n", encoding="utf-8")
    assert R.provenance()["dirty"] == ["app/src/main/Model.kt", "tools/report.py"]


def test_a_clean_tree_reports_nothing_dirty(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", repo(tmp_path))
    assert R.provenance()["dirty"] == []


def test_the_tag_is_named_when_head_is_the_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ROOT", repo(tmp_path))
    assert R.provenance()["tag"] == "v0.2.0"


def test_a_page_only_commit_still_carries_the_released_model(tmp_path, monkeypatch):
    """Publishing the page moves HEAD past the tag. The model did not move, and
    calling it unreleased is the same error as calling it released."""
    path = repo(tmp_path)
    commit_a_generator_change(path)
    monkeypatch.setattr(R, "ROOT", path)
    prov = R.provenance()
    assert prov["tag"] == "", "HEAD is no longer the tag"
    assert prov["release"] == "v0.2.0"


def test_a_changed_model_is_not_the_released_one(tmp_path, monkeypatch):
    """The guard against the exit-code trap: this must come out empty, and with
    `git diff --quiet` behind it, it did not."""
    path = repo(tmp_path)
    (path / "app/src/main/Model.kt").write_text("fun predict() = 99\n",
                                                encoding="utf-8")
    subprocess.run(["git", "commit", "-aqm", "new model"], cwd=path, check=True,
                   capture_output=True)
    monkeypatch.setattr(R, "ROOT", path)
    assert R.provenance()["release"] == ""


def test_a_version_with_no_tag_at_all_is_not_a_release(tmp_path, monkeypatch):
    path = repo(tmp_path)
    (path / "app/build.gradle.kts").write_text(
        'versionCode = 7\nversionName = "0.3.0"\n', encoding="utf-8")
    subprocess.run(["git", "commit", "-aqm", "bump"], cwd=path, check=True,
                   capture_output=True)
    monkeypatch.setattr(R, "ROOT", path)
    prov = R.provenance()
    assert prov["version"] == "0.3.0" and prov["release"] == "" and prov["tag"] == ""


# --- journeys with a change --------------------------------------------------
#
# This section starts later than every other one on the page, because the far
# end of a change only began being polled once the second tier existed. The
# failure to guard against is therefore not a wrong number but a section that
# appears anyway, full of zeros, and reads as a result.


def journey(*, crps=1.0, db=2.0, truth=2.0, q10=-1.0, q90=5.0,
            lead_minutes=30.0, candidates=4, miss_p=0.05, num="1",
            beyond_list=False):
    return {
        "eva": "8000001", "cat": "RE", "num": num, "dest": "Musterstadt",
        "tau": 0, "lead": lead_minutes,
        "planned": 8 * 60 + 5, "planned_dep": 8 * 60,
        "read_at": wall_to_epoch(8 * 60) - lead_minutes * 60,
        "candidates": candidates, "miss_p": miss_p,
        "beyond_list": beyond_list,
        "db": db, "truth": truth, "crps": crps,
        "cdf_at": 0.5, "cdf_below": 0.4, "q10": q10, "q50": 1.0, "q90": q90,
        "source": "EMPIRICAL", "runs": 9,
    }


def test_the_journey_section_is_absent_until_there_is_something_in_it(tmp_path):
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn),
             journeys=R.journeys_table([], []))
    assert "Journeys with a change" not in out.read_text(encoding="utf-8")


def test_the_intro_does_not_deny_a_section_the_page_contains(tmp_path):
    """The opening said a two-leg journey is *not* scored here, which was true
    until it was — and then stood two screens above the table that scores it."""
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    trips = [journey(num=str(i)) for i in range(4)]
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn, trips, trips),
             journeys=R.journeys_table(trips, trips))
    page = out.read_text(encoding="utf-8")
    assert "Journeys with a change" in page
    assert "is <em>not</em> scored here" not in page


def test_the_intro_still_says_so_while_the_section_is_missing(tmp_path):
    """And the other way round: with no journeys the caveat has to stay."""
    out = tmp_path / "report.html"
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    R.render(["2026-08-17"], R.arrivals_table(rows, rows),
             R.connections_table(conn, conn), R.outcome_split(conn, conn),
             {"events": 4, "connections": 4}, out,
             gaps=R.headline(rows, rows, conn, conn),
             journeys=R.journeys_table([], []))
    assert "is <em>not</em> scored here" in out.read_text(encoding="utf-8")


def test_the_headline_gains_no_journey_row_until_there_is_one():
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    got = R.headline(rows, rows, conn, conn, [], [])
    assert not any("Journey" in r["what"] for r in got)


def test_the_headline_carries_both_variants_of_the_journey():
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    trips = [journey(num=str(i)) for i in range(4)]
    got = R.headline(rows, rows, conn, conn, trips, trips)
    journeys = [r["what"] for r in got if "Journey" in r["what"]]
    assert journeys == ["Journey with a change, as shipped",
                        "Journey with a change, history only"]


def test_a_journey_is_scored_in_the_same_units_as_a_direct_one():
    """The whole point of the section: CRPS in minutes against DB's own error,
    so a row here and a row from the arrivals table can be read together."""
    rows = [arrival(num=str(i)) for i in range(4)]
    conn = [connection(num=str(i), caught=i > 0) for i in range(4)]
    trips = [journey(num=str(i), crps=1.0, db=2.0, truth=4.0) for i in range(4)]
    got = R.headline(rows, rows, conn, conn, trips, trips)
    row = next(r for r in got if r["what"] == "Journey with a change, as shipped")
    assert row["unit"] == "CRPS, minutes"
    assert row["gap"] == pytest.approx(1.0 - 2.0)


def test_the_journey_table_is_bucketed_by_lead_time():
    trips = (
        [journey(num="1", lead_minutes=5.0)]
        + [journey(num="2", lead_minutes=200.0)]
    )
    got = R.journeys_table(trips, trips)
    assert [r["bucket"] for r in got] == ["<10m", ">3h"]


def test_the_journey_table_reports_what_the_model_was_given():
    """How many trains it chose between, and how much weight it put on catching
    none of them — without those the CRPS is a number with no context."""
    trips = [journey(candidates=6, miss_p=0.2), journey(num="2", candidates=2,
                                                        miss_p=0.0)]
    got = R.journeys_table(trips, trips)[0]
    assert got["candidates"] == pytest.approx(4.0)
    assert got["miss_p"] == pytest.approx(0.1)


def test_the_table_reports_how_often_truth_walked_past_the_apps_list():
    """The share the whole section turns on: a passenger who missed every train
    the app offered. Scoring only the journeys that fit inside the list would
    drop exactly the answers that were most wrong."""
    trips = [journey(num="1", beyond_list=True), journey(num="2"),
             journey(num="3"), journey(num="4")]
    assert R.journeys_table(trips, trips)[0]["beyond"] == pytest.approx(0.25)


def test_an_empty_journey_table_is_empty_not_a_row_of_zeros():
    assert R.journeys_table([], []) == []


# --- the lead-time tables have to cover every event --------------------------
#
# A train that terminates at the station has an arrival and no departure, and
# the anchor was the departure alone: 15,557 of the first seven days' 121,395
# arrivals silently never reached a lead-time bucket. They are also where DB's
# live report helps most, so the omission understated it in the one table that
# is meant to show where it helps.


def test_a_terminating_train_still_gets_a_lead_time():
    rows = [arrival(num="1"), {**arrival(num="2"), "planned_dep": None}]
    got = R.with_lead(rows)
    assert got["_lead"].null_count() == 0
    assert got["_bucket"].null_count() == 0


def test_every_event_reaches_a_bucket():
    rows = [arrival(num=str(i), lead_minutes=lead)
            for i, lead in enumerate((5.0, 30.0, 200.0))]
    rows += [{**arrival(num="t"), "planned_dep": None}]
    assert sum(r["n"] for r in R.arrivals_table(rows, rows)) == len(rows)


def test_an_event_with_no_time_at_all_stops_the_run():
    """Silently dropping it is what this whole block exists to prevent."""
    rows = [{**arrival(), "planned_dep": None, "planned": None}]
    with pytest.raises(SystemExit, match="nothing to measure a lead from"):
        R.with_lead(rows)


def journey_row(**over):
    """One scored two-leg journey, with the fields the pairing keys on."""
    row = {"day": "2026-08-24", "eva": "8000001", "cat": "RE", "num": "100",
           "planned": 29800000,
           "reference_id": "ref-1", "planned_dep": 29800000, "read_at": 1787000000.0,
           "lead": 30.0, "crps": 1.0, "db": 5.0, "truth": 5.0, "q10": 0.0, "q90": 9.0,
           "candidates": 5.0, "miss_p": 0.0, "beyond_list": False}
    row.update(over)
    return row


def test_journeys_are_paired_before_they_are_compared():
    """The two variants do not answer the same journeys.

    A connecting train with too little shared history can be carried by a live
    report and not without one, so the shipped model answers journeys the
    blinded one declines. Measuring each over its own set puts answers to
    different questions in one row.
    """
    shared = [journey_row(reference_id=f"s{i}", crps=1.0) for i in range(4)]
    live = shared + [journey_row(reference_id="live-only", crps=99.0)]
    blind = shared + [journey_row(reference_id="blind-only", crps=0.01)]
    pl_, pb_, n_live, n_blind = R.paired_journeys(live, blind)
    assert (n_live, n_blind) == (5, 5)
    assert pl_.height == pb_.height == 4
    assert "live-only" not in pl_["reference_id"].to_list()
    assert "blind-only" not in pb_["reference_id"].to_list()


def test_the_unpaired_rows_never_reach_a_score():
    """The regression: an outlier only one variant answered moved its column."""
    shared = [journey_row(reference_id=f"s{i}", crps=1.0, db=5.0, truth=5.0)
              for i in range(20)]
    live = shared + [journey_row(reference_id="only", crps=500.0, db=5.0, truth=5.0)]
    rows = R.journeys_table(live, shared)
    assert rows, "the table should still be produced"
    assert rows[0]["live"] == pytest.approx(1.0), (
        "a journey the blinded variant never answered must not move the "
        "shipped column"
    )
    assert rows[0]["n"] == 20


def test_the_headline_journey_rows_use_the_shared_set():
    shared = [journey_row(reference_id=f"s{i}") for i in range(6)]
    live = shared + [journey_row(reference_id="only", crps=99.0)]
    arrivals = [arrival() for _ in range(3)]
    # One missed, or only_missed() hands brier_gap an empty frame.
    conns = [connection() for _ in range(3)] + [connection(caught=False)]
    gaps = R.headline(arrivals, arrivals, conns, conns,
                      journey_live=live, journey_blind=shared)
    journey_rows = [g for g in gaps if "Journey with a change" in g["what"]]
    assert len(journey_rows) == 2
    assert {g["n"] for g in journey_rows} == {6}, (
        "both journey rows must be counted over the journeys both answered"
    )


def test_arrivals_are_left_alone():
    """Arrivals and connections are already paired; pairing them again would
    silently drop rows if a key were ever missing from those files."""
    src = (ROOT / "tools" / "report.py").read_text(encoding="utf-8")
    body = src[src.index("def arrivals_table"):src.index("def connections_table")]
    assert "paired_journeys" not in body


def test_pairing_needs_a_key_it_can_use():
    """Silently pairing on nothing would return an empty table, not an error."""
    rows = [{"crps": 1.0, "db": 1.0, "truth": 1.0}]
    with pytest.raises(SystemExit, match="cannot be paired"):
        R.paired_journeys(rows, rows)


def test_pairing_returns_the_same_number_of_rows_on_both_sides():
    """A key that is not unique fans the inner join out instead of pairing.

    `reference_id` names the feeder run, and one feeder serves several
    destinations, so without `dest` 2,199 journeys collided and the two paired
    columns came back with different row counts — which is the visible symptom
    of comparing something other than one journey against itself.
    """
    live = [journey_row(reference_id="feeder", dest="A"),
            journey_row(reference_id="feeder", dest="B"),
            journey_row(reference_id="feeder", dest="C")]
    blind = [journey_row(reference_id="feeder", dest="A"),
             journey_row(reference_id="feeder", dest="B")]
    pl_, pb_, _, _ = R.paired_journeys(live, blind)
    assert pl_.height == pb_.height == 2
    assert sorted(pl_["dest"].to_list()) == ["A", "B"]


def test_the_journey_key_distinguishes_the_far_end():
    """Without `dest` the key is not a journey, it is a feeder."""
    assert "dest" in R.JOURNEY_KEY
    assert "reference_id" in R.JOURNEY_KEY
