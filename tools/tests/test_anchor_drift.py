"""Tests for the live-anchor drift monitor.

Three things can make this monitor lie, and each has a test here.

It can compare unlike windows: the collector covers six evening hours and the
historical journals cover whole days, so a reference built without the window
filter would be measuring a different traffic mix than the thing it is compared
against.

It can call noise drift. One train contributes an event at every probed lead
time and those errors are the same train's, so an observation-level bootstrap
would shrink the interval by roughly the square root of the events per trip and
flag every week as a regime change.

And it can miss a real move, which is the failure nobody notices.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import anchor_drift as ad  # noqa: E402
import collect_forecasts as cf  # noqa: E402
import score_events as se  # noqa: E402

DAY = dt.date(2026, 9, 1)


def epoch(hour: int, minute: int = 0) -> float:
    """German wall clock on [DAY] to epoch seconds."""
    return dt.datetime.combine(DAY, dt.time(hour, minute), se.BERLIN).timestamp()


def wall(hour: int, minute: int = 0) -> int:
    """The same instant as IRIS spells it: wall clock stored as if UTC."""
    return cf.iris_time(f"{DAY:%y%m%d}{hour:02d}{minute:02d}")


def journal(tmp_path: Path, records: list[dict]) -> Path:
    fh = cf.Journal(tmp_path / f"forecasts-{DAY}.jsonl")
    for r in records:
        fh.append(r)
    fh.close()
    return tmp_path


def train(trip: str, planned_hour: int, *, forecast: int, final: int,
          first_seen: float, eva: str = "8000001") -> list[dict]:
    """One train: its plan, one forecast, and the value it settles on.

    `forecast` is what DB says from `first_seen` onwards; `final` is what it
    says once the train is long past, which is the truth `Stop.settled` reads.
    """
    planned = wall(planned_hour)
    arrival = epoch(planned_hour)
    return [
        {"t": "plan", "at": first_seen, "eva": eva, "id": trip, "cat": "RE",
         "num": trip, "line": None, "par": planned, "pdp": None, "ppth": []},
        {"t": "obs", "at": first_seen, "eva": eva, "id": trip,
         "ar": planned + forecast, "dp": None, "arc": False, "dpc": False,
         "msg": []},
        {"t": "obs", "at": arrival + 60 * (final + 5), "eva": eva, "id": trip,
         "ar": planned + final, "dp": None, "arc": False, "dpc": False,
         "msg": []},
    ]


def polls(eva: str, start: float, end: float, step: float = 600) -> list[dict]:
    at = start
    out = []
    while at < end:
        out.append({"t": "poll", "at": at, "eva": eva, "ok": True, "stops": 1,
                    "tier": 1, "cohort": 1})
        at += step
    return out


# --- the window filter -------------------------------------------------------

def test_the_window_keeps_only_polls_and_observations_inside_it(tmp_path):
    out = journal(tmp_path,
                  polls("8000001", epoch(6), epoch(23))
                  + train("morning", 8, forecast=0, final=2, first_seen=epoch(7))
                  + train("evening", 18, forecast=0, final=2, first_seen=epoch(16)))
    _, all_day = se.read_day(out, DAY)
    _, evening = se.read_day(out, DAY, within=ad.window_of(DAY))
    assert min(all_day["8000001"]) < epoch(15) <= min(evening["8000001"])
    assert max(evening["8000001"]) < epoch(21)


def test_the_window_keeps_the_plan_of_a_train_first_listed_before_it(tmp_path):
    """A run starting at 15:00 fetches its whole horizon at once.

    So a train listed that morning is in its plan too. Clipping `plan` by the
    window would delete every such stop and leave the reference describing only
    trains that appeared late — the one bias this filter exists to avoid.
    """
    out = journal(tmp_path,
                  polls("8000001", epoch(6), epoch(23))
                  + train("early-plan", 18, forecast=0, final=9,
                          first_seen=epoch(9)))
    stops, _ = se.read_day(out, DAY, within=ad.window_of(DAY))
    assert ("8000001", "early-plan") in stops


def test_a_train_the_window_cuts_off_has_no_settled_truth(tmp_path):
    """Polling stops at 21:00, so a 21:00 arrival never settles.

    `Stop.settled` wants half an hour of quiet after the predicted arrival
    before it will call a number final. Nothing here supplies that, and the
    events the probes did build are dropped rather than scored against DB's
    last guess — which at short lead is the very forecast being scored.
    """
    out = journal(tmp_path,
                  polls("8000001", epoch(15), epoch(21))
                  + train("late", 21, forecast=0, final=3, first_seen=epoch(16)))
    stops, station_polls = se.read_day(tmp_path, DAY, within=ad.window_of(DAY))
    assert se.build_events(stops, station_polls, horizons=ad.PROBES), \
        "the probes should have produced events for this train"
    lead, err, _ = ad.residuals(tmp_path, [DAY])
    assert lead.size == 0
    assert err.size == 0


# --- the residual ------------------------------------------------------------

def test_the_residual_is_the_settled_delay_minus_what_db_said(tmp_path):
    out = journal(tmp_path,
                  polls("8000001", epoch(15), epoch(21))
                  + train("t", 18, forecast=2, final=9, first_seen=epoch(15)))
    lead, err, trips = ad.residuals(tmp_path, [DAY])
    assert err.size > 0
    assert set(err.tolist()) == {7.0}, "settled 9 minus a reported 2"
    assert set(trips.tolist()) == {0}, "one train, one cluster"
    # The window opens at 15:00, so no probe reaches further back than that.
    assert lead.max() <= 190


def test_silence_is_a_forecast_of_on_time_and_the_monitor_skips_it(tmp_path):
    """`fchg` leaving a stop out means the timetable stands, not "no data".

    `build_events` is right to score that as a forecast of zero — but a
    forecast of zero is one the app never anchors on, so `residuals` drops it.
    Both halves are asserted together because the second only makes sense given
    the first.
    """
    planned = wall(18)
    out = journal(tmp_path, polls("8000001", epoch(15), epoch(21)) + [
        {"t": "plan", "at": epoch(16), "eva": "8000001", "id": "quiet",
         "cat": "RE", "num": "1", "line": None, "par": planned, "pdp": None,
         "ppth": []},
        # Silent until an hour before it was due, then six minutes late.
        {"t": "obs", "at": epoch(17), "eva": "8000001", "id": "quiet",
         "ar": planned + 6, "dp": None, "arc": False, "dpc": False, "msg": []},
    ])
    stops, station_polls = se.read_day(out, DAY, within=ad.window_of(DAY))
    events = se.build_events(stops, station_polls, horizons=ad.PROBES)
    early = [e for e in events if e["lead"] > 65]
    assert early and all(e["db"] == 0 and not e["db_explicit"] for e in early)

    lead, err, _ = ad.residuals(tmp_path, [DAY])
    assert set(err[lead < 55].tolist()) == {0.0}, "once reported, DB has it right"
    assert lead.size and lead.max() < 65, "the silent stretch is not a residual"


def test_a_stop_db_never_mentions_at_all_supplies_no_truth(tmp_path):
    """The population is conditioned on DB having spoken by the cutoff.

    `Stop.settled` reads DB's forecast half an hour after the predicted
    arrival, and a stop that never appeared in `fchg` has no forecast to read —
    so it drops out rather than counting as a punctual arrival. That is a real
    selection in every number this monitor and `calibrate_live.py` produce: the
    residuals describe trains DB had an opinion about, which are not all trains.
    """
    planned = wall(18)
    out = journal(tmp_path, polls("8000001", epoch(15), epoch(21)) + [
        {"t": "plan", "at": epoch(16), "eva": "8000001", "id": "silent",
         "cat": "RE", "num": "1", "line": None, "par": planned, "pdp": None,
         "ppth": []},
    ])
    _, err, _ = ad.residuals(tmp_path, [DAY])
    assert err.size == 0


# --- binning and monotonicity ------------------------------------------------

def test_bins_follow_the_edges():
    lead = np.array([0.0, 4.9, 5.0, 119.9, 120.0, 5000.0])
    assert ad.bin_of(lead).tolist() == [0, 0, 1, 8, 9, 9]


def test_isotonic_pools_a_dip_and_leaves_a_rising_curve_alone():
    rising = [1.0, 2.0, 3.0]
    assert ad.isotonic(rising) == rising
    # 4 then 2 is physically impossible; both become their mean.
    assert ad.isotonic([4.0, 2.0, 9.0]) == [3.0, 3.0, 9.0]
    # A gap carries no weight and is left where it is.
    assert ad.isotonic([4.0, None, 2.0]) == [3.0, None, 3.0]


def test_isotonic_repairs_a_violation_it_created_by_merging():
    # Merging 5 and 1 gives 3, which is now below the 4 in front of it; a
    # single forward pass would leave 4, 3, 3 behind.
    assert ad.isotonic([4.0, 5.0, 1.0]) == pytest.approx([10 / 3] * 3)


# --- the bootstrap -----------------------------------------------------------

def test_the_bootstrap_resamples_whole_trips_not_events():
    """Twenty readings of one train are one train's luck, not twenty facts.

    Two trips, each contributing many identical residuals of its own. Resampling
    events would almost always draw a mix and report a wide spread; resampling
    trips draws one trip or the other about half the time, and a single trip has
    no spread at all.
    """
    per_trip = 20
    err = np.array([0.0] * per_trip + [10.0] * per_trip)
    bins = np.zeros(2 * per_trip, np.int64)
    trips = np.array([0] * per_trip + [1] * per_trip, np.int64)
    (lo, hi), = [c for c in ad.bootstrap(err, bins, trips, draws=200) if c]
    assert lo == 0.0, "an all-one-trip draw must be possible, and has no spread"
    assert hi > 0.0


def test_offsets_expands_run_lengths():
    assert ad._offsets(np.array([3, 1, 2])).tolist() == [0, 1, 2, 0, 0, 1]


# --- the verdict -------------------------------------------------------------

def bins(widths, n=10_000, ci=None):
    return {"days": [str(DAY)] * ad.MIN_DAYS, "hours": list(ad.WINDOW_HOURS),
            "events": n * len(widths), "trips": n,
            "bins": [{"label": label, "n": n, "width": w, "isotonic": w,
                      "ci": (ci[i] if ci else
                             None if w is None else [w - 0.1, w + 0.1])}
                     for i, (label, w) in enumerate(zip(ad.EDGE_LABELS, widths))]}


def test_a_big_move_outside_the_interval_is_drift():
    now = bins([4.0] + [None] * 9)
    was = bins([10.0] + [None] * 9)
    assert ad.compare(now, was)[0]["state"] == "drift"


def test_a_small_move_is_not_drift_however_significant():
    """Tens of thousands of events make a half-minute shift detectable.

    It is still half a minute, on a model that answers in whole ones.
    """
    now = bins([10.5] + [None] * 9, ci=[[10.4, 10.6]] + [None] * 9)
    was = bins([10.0] + [None] * 9)
    verdict = ad.compare(now, was)[0]
    assert verdict["state"] == "ok"
    assert "significant but small" in verdict["note"]


def test_a_big_move_inside_the_interval_is_not_drift():
    """A bin that moved a lot but cannot resolve the move is not evidence."""
    now = bins([4.0] + [None] * 9, ci=[[1.0, 12.0]] + [None] * 9)
    was = bins([10.0] + [None] * 9)
    assert ad.compare(now, was)[0]["state"] == "ok"


def test_a_thin_bin_is_reported_as_thin_rather_than_drifted():
    now = bins([4.0] + [None] * 9, n=ad.MIN_EVENTS - 1)
    was = bins([10.0] + [None] * 9)
    assert ad.compare(now, was)[0]["state"] == "thin"


# --- the committed reference -------------------------------------------------

def test_the_committed_reference_is_readable_and_covers_the_same_grid():
    """The file `check` compares against in CI, on the grid it compares on.

    An edge list edited here and not there would silently compare bin `20-30`
    against whatever used to be third in the file.
    """
    reference = json.loads(ad.REFERENCE.read_text())
    assert [b["label"] for b in reference["bins"]] == list(ad.EDGE_LABELS)
    assert reference["hours"] == list(ad.WINDOW_HOURS)
    assert reference["edges"] == list(ad.EDGES[:-1]) + ["inf"]
    # The raw curve may dip — quantiles of a few thousand correlated events do.
    # The isotonic one may not: that is the physical claim the monitor rests on.
    rising = [b["isotonic"] for b in reference["bins"] if b["isotonic"] is not None]
    assert rising == sorted(rising), "the reference curve must rise with lead"


# --- the workflows -----------------------------------------------------------
#
# Both files below encode the same two decisions the code does — which hours are
# collected, and how many days a curve needs. A copy of either that drifted from
# the constant would not fail anything: the collector would run a different
# window from the one the reference describes, and the comparison would report
# that difference as drift in the railway.

WORKFLOWS = TOOLS.parent / ".github/workflows"


def test_the_collector_workflow_reads_the_window_from_the_collector():
    text = (WORKFLOWS / "collect-forecasts.yml").read_text()
    assert "from collect_forecasts import WINDOW_HOURS" in text
    for hour in ad.WINDOW_HOURS:
        assert f'"today {hour}:00"' not in text, \
            "the window is hardcoded in the workflow as well as in the code"


def test_the_collector_starts_early_enough_to_cover_the_window():
    """GitHub starts scheduled jobs late, so the cron has to sit ahead of it.

    Not too far ahead: the job idles until the window opens, on a runner that
    is being paid for in minutes.
    """
    text = (WORKFLOWS / "collect-forecasts.yml").read_text()
    cron = next(line.split('"')[1] for line in text.splitlines()
                if line.strip().startswith("- cron:"))
    minute, hour = (int(f) for f in cron.split()[:2])
    # Summer, when the offset is largest and the cron therefore closest to the
    # window it has to be in front of.
    fires = dt.datetime(2026, 7, 1, hour, minute,
                        tzinfo=dt.timezone.utc).astimezone(se.BERLIN)
    opens = dt.datetime(2026, 7, 1, ad.WINDOW_HOURS[0], tzinfo=se.BERLIN)
    ahead = (opens - fires).total_seconds() / 60
    assert 5 <= ahead <= 30, f"cron fires {ahead:.0f} min before the window opens"


def test_the_drift_workflow_checks_at_least_the_days_a_curve_needs():
    text = (WORKFLOWS / "anchor-drift.yml").read_text()
    back = next(line for line in text.splitlines() if '-13 days' in line)
    assert back, "the default window is set by a `date -d` offset"
    # "-13 days" back from yesterday inclusive is a fortnight.
    assert 13 + 1 >= ad.MIN_DAYS


# --- the population the monitor watches --------------------------------------

def test_only_stops_the_app_would_anchor_on_are_counted(tmp_path):
    """A report under a minute is the timetable restated, not evidence.

    `LiveReport.informative` returns null for it and the app never anchors, so
    including it here would measure the error of a forecast the model does not
    make. It also swamps the sample: on-time stops are the overwhelming
    majority, and their residual barely widens with lead.
    """
    out = journal(tmp_path,
                  polls("8000001", epoch(15), epoch(21))
                  + train("quiet", 18, forecast=0, final=8, first_seen=epoch(15))
                  + train("late", 19, forecast=4, final=9, first_seen=epoch(15)))
    _, err, _ = ad.residuals(tmp_path, [DAY])
    assert err.size > 0
    assert set(err.tolist()) == {5.0}, "only the reported-late train counts"


def test_the_threshold_matches_the_one_the_app_enforces():
    """Drift guard: the app's rule lives in Kotlin, and this must follow it."""
    kotlin = (TOOLS.parent / "app/src/main/java/io/github/derweh/bayesianbahn"
              / "model/LiveReport.kt").read_text()
    line = next(l for l in kotlin.splitlines()
                if "MIN_INFORMATIVE_DELAY_MINUTES" in l and "const" in l)
    assert float(line.split("=")[1].strip()) == ad.MIN_REPORT
