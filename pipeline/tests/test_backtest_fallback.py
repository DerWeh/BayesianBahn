"""Tests for the line-keyed fallback backtest.

Two things can go wrong here and neither is visible in the output. The mirror of
the app's model can drift, so the backtest measures a model nobody ships — this
has happened before, in `backtest.py`, where the long-distance category set lost
`WB` for months while the tables kept looking reasonable. And the closed-form
prior can be scored wrongly, which would move the whole comparison the change is
justified by, in whichever direction the bug happened to point.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest_fallback as bf  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app/src/main/java/io/github/derweh/bayesianbahn"
EMPIRICAL_DELAY = APP / "model/EmpiricalDelay.kt"
DELAY_MODEL = APP / "model/DelayModel.kt"


def kotlin_const(path: Path, name: str) -> float:
    """The value of a `const val NAME = <number>` in a Kotlin file."""
    match = re.search(rf"const val {name} = (-?[\d.]+)", path.read_text(encoding="utf-8"))
    assert match, f"{name} not found in {path.name}"
    return float(match.group(1))


# --- the mirror -------------------------------------------------------------


def test_the_weighting_constants_are_the_app_s() -> None:
    """A backtest weighting runs differently from the app measures nothing."""
    assert bf.HALF_LIFE_DAYS == kotlin_const(EMPIRICAL_DELAY, "RECENCY_HALF_LIFE_DAYS")
    assert bf.SAME_WEEKDAY_BOOST == kotlin_const(EMPIRICAL_DELAY, "SAME_WEEKDAY_BOOST")
    assert bf.TOD_WINDOW_MIN == kotlin_const(EMPIRICAL_DELAY, "TIME_OF_DAY_WINDOW_MIN")
    assert bf.MIN_EFFECTIVE_N == kotlin_const(EMPIRICAL_DELAY, "MIN_EFFECTIVE_N")
    assert bf.LIVE_KERNEL_FLOOR == kotlin_const(EMPIRICAL_DELAY, "LIVE_KERNEL_FLOOR")
    assert bf.LIVE_SHRINKAGE == kotlin_const(DELAY_MODEL, "LIVE_SHRINKAGE")
    assert bf.MIN_LIVE_SCALE == kotlin_const(DELAY_MODEL, "MIN_LIVE_SCALE")


def test_the_priors_are_the_app_s() -> None:
    """The prior *is* the incumbent here, so its four numbers decide the result."""
    source = DELAY_MODEL.read_text(encoding="utf-8")
    found = {}
    for name, mu0, kappa0, alpha0, beta0 in re.findall(
        r"TrainClass\.(\w+) to Prior\(mu0 = ([\d.]+), kappa0 = ([\d.]+), "
        r"alpha0 = ([\d.]+), beta0 = ([\d.]+)\)",
        source,
    ):
        found[name] = tuple(float(v) for v in (mu0, kappa0, alpha0, beta0))
    assert found == {
        "LONG_DISTANCE": bf.PRIORS["long_distance"],
        "REGIONAL": bf.PRIORS["regional"],
        "SBAHN": bf.PRIORS["sbahn"],
        "OTHER": bf.PRIORS["other"],
    }


def test_the_prior_is_the_untrained_one() -> None:
    """`Predictor` builds a fresh `DelayModel` and never feeds it, so the
    posterior predictive is the prior — including its time band, which with no
    observations makes no difference at all. If the app ever starts training it,
    this backtest is scoring a distribution the app no longer uses."""
    source = (APP / "data/Predictor.kt").read_text(encoding="utf-8")
    assert "fallbackModel: DelayModel = DelayModel()" in source
    assert "observe(" not in source


def test_train_classes_match_the_app() -> None:
    for category, expected in [
        ("ICE", "long_distance"), ("WB", "long_distance"), ("S", "sbahn"),
        ("RE", "regional"), ("Bus", "regional"), ("", "other"), ("7", "other"),
    ]:
        assert bf.train_class(category) == expected


# --- scoring the prior ------------------------------------------------------


@pytest.mark.parametrize("train_cls", sorted(bf.PRIORS))
def test_the_prior_quantiles_match_a_student_t(train_cls: str) -> None:
    """Against the closed form, via the app's own df/loc/scale arithmetic."""
    df, _, scale = bf.prior_shape(train_cls)
    scorer = bf.prior_grids()[(train_cls, "blind")]
    # t_6 has its 90th percentile at 1.4398 standardised units.
    assert df == 6.0
    assert scorer.quantile(0.9) == pytest.approx(1.4398 * scale, abs=1e-3)
    assert scorer.quantile(0.5) == pytest.approx(0.0, abs=1e-6)
    assert scorer.quantile(0.1) == pytest.approx(-scorer.quantile(0.9), abs=1e-6)


def test_the_prior_crps_matches_a_simulation() -> None:
    """The grid is quadrature, so it is worth checking against sampling once.

    CRPS = E|X - y| - E|X - X'|/2, which needs nothing but draws.
    """
    df, _, scale = bf.prior_shape("regional")
    rng = np.random.default_rng(20260901)
    n = 2_000_000
    draw = lambda: (rng.standard_normal(n)  # noqa: E731
                    / np.sqrt(rng.chisquare(df, n) / df) * scale)
    x, x_prime = draw(), draw()
    scorer = bf.prior_grids()[("regional", "blind")]
    pair = float(np.abs(x - x_prime).mean())
    for y in (0.0, 2.0, -5.0, 20.0):
        simulated = float(np.abs(x - y).mean()) - 0.5 * pair
        assert scorer.crps(np.array([y]))[0] == pytest.approx(simulated, abs=0.02)


def test_a_live_report_narrows_the_prior_but_never_past_the_floor() -> None:
    """`predictiveFor` re-anchors on the report and shrinks the spread to 40%,
    with a floor — so the S-Bahn's prior, whose scale is smallest, is the one
    the floor binds for."""
    for train_cls in bf.PRIORS:
        _, _, scale = bf.prior_shape(train_cls)
        expected = max(scale * bf.LIVE_SHRINKAGE, bf.MIN_LIVE_SCALE)
        live = bf.prior_grids()[(train_cls, "live")]
        assert live.quantile(0.9) == pytest.approx(1.4398 * expected, abs=1e-3)
    assert bf.prior_shape("sbahn")[2] * bf.LIVE_SHRINKAGE < bf.MIN_LIVE_SCALE


# --- the empirical model ----------------------------------------------------


def history(days, delays, prev=None, weekday=None):
    n = len(days)
    return (
        np.array(days, dtype=np.int32),
        np.asarray(delays, dtype=float),
        np.full(n, np.nan) if prev is None else np.asarray(prev, dtype=float),
        np.zeros(n, dtype=np.int32) if weekday is None else np.asarray(weekday),
    )


def test_recency_halves_a_run_s_weight_every_half_life() -> None:
    _, weights = bf.empirical(history([0, -30], [1.0, 2.0]), 0, 1, None)
    assert weights[1] / weights[0] == pytest.approx(0.5)


def test_a_run_on_the_same_weekday_counts_double() -> None:
    _, weights = bf.empirical(
        history([0, 0], [1.0, 2.0], weekday=[3, 4]), 0, 3, None,
    )
    assert weights[0] / weights[1] == pytest.approx(bf.SAME_WEEKDAY_BOOST)


def test_the_live_model_shifts_each_run_s_own_progression() -> None:
    """The delta model: a run that gained two minutes after its previous stop
    contributes `live + 2`, not its own final delay."""
    n = int(bf.MIN_EFFECTIVE_N)
    support, _ = bf.empirical(
        history([0] * n, [5.0] * n, prev=[3.0] * n), 0, 1, live=10.0,
    )
    assert np.allclose(support, 12.0)


def test_a_thin_live_history_falls_back_to_the_plain_runs() -> None:
    """Fewer runs with a known previous stop than the floor and the delta model
    has nothing to shift; the app returns the unconditioned distribution."""
    support, _ = bf.empirical(
        history([0, 0], [5.0, 6.0], prev=[3.0, 3.0]), 0, 1, live=10.0,
    )
    assert np.allclose(support, [5.0, 6.0])


def test_effective_n_counts_equal_runs_and_discounts_lopsided_ones() -> None:
    assert bf.effective_n(np.ones(10)) == pytest.approx(10.0)
    assert bf.effective_n(np.array([1.0, 1e-9])) == pytest.approx(1.0, abs=1e-6)


# --- the history window -----------------------------------------------------


def test_only_strictly_earlier_runs_are_visible() -> None:
    """A walk-forward that can see the day it is predicting measures nothing."""
    store = bf.History(
        np.array([1, 2, 3], dtype=np.int32), np.array([480, 480, 480], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]), np.full(3, np.nan), np.zeros(3, dtype=np.int32),
    )
    days, delay, *_ = store.before(480, 3, bf.TOD_WINDOW_MIN)
    assert list(days) == [1, 2]
    assert list(delay) == [1.0, 2.0]


def test_the_time_of_day_window_wraps_around_midnight() -> None:
    """23:50 and 00:10 are twenty minutes apart, not twenty-three hours."""
    store = bf.History(
        np.array([1, 1], dtype=np.int32), np.array([1430, 720], dtype=np.int32),
        np.array([1.0, 2.0]), np.full(2, np.nan), np.zeros(2, dtype=np.int32),
    )
    _, delay, *_ = store.before(10, 5, bf.TOD_WINDOW_MIN)
    assert list(delay) == [1.0]


def test_a_line_shard_only_carries_the_days_it_publishes() -> None:
    """The backtest must read the shard the pipeline writes, not the archive.

    Reading a line's whole history here would credit the fallback with runs no
    published shard contains, and the app would then underperform its own
    backtest with nothing to point at.
    """
    store = bf.History(
        np.array([0, 50, 99], dtype=np.int32), np.array([480, 480, 480], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]), np.full(3, np.nan), np.zeros(3, dtype=np.int32),
    )
    _, delay, *_ = store.before(480, 100, bf.TOD_WINDOW_MIN, max_days=45)
    assert list(delay) == [3.0]
    _, all_delay, *_ = store.before(480, 100, bf.TOD_WINDOW_MIN)
    assert list(all_delay) == [1.0, 2.0, 3.0]


def test_the_trim_matches_what_the_pipeline_writes() -> None:
    import build_shards

    assert bf.LINE_DAYS == build_shards.LINE_DAYS


# --- the gate ---------------------------------------------------------------


def test_a_report_of_on_time_is_not_evidence() -> None:
    """The same rule as the app's: three of DB's four shapes mean "on time"."""
    assert bf.gated(None) is None
    assert bf.gated(0.0) is None
    assert bf.gated(-3.0) is None
    assert bf.gated(bf.MIN_INFORMATIVE_DELAY) == bf.MIN_INFORMATIVE_DELAY
    assert bf.gated(12.0) == 12.0


def test_the_empirical_crps_is_exact_for_a_point_mass() -> None:
    """A distribution certain of m scores |m - y|, which is what DB's point
    forecast is scored as everywhere else in this project."""
    x, w = np.array([4.0]), np.array([1.0])
    assert bf.crps_empirical(x, w, 9.0) == pytest.approx(5.0)
    assert bf.crps_empirical(x, w, 4.0) == pytest.approx(0.0)


def test_the_empirical_crps_matches_the_closed_form_for_two_points() -> None:
    """Two equally likely points a apart, observed at one of them:
    CRPS = a/4 by direct integration."""
    x, w = np.array([0.0, 8.0]), np.array([1.0, 1.0])
    assert bf.crps_empirical(x, w, 0.0) == pytest.approx(2.0)
    assert bf.crps_empirical(x, w, 8.0) == pytest.approx(2.0)


def test_a_cluster_interval_is_wider_than_the_events_alone_suggest() -> None:
    """Events of one train are not independent, and an interval that resamples
    events instead of trains would be far too narrow to trust."""
    rng = np.random.default_rng(7)
    # Twenty trains, fifty events each, the whole difference between trains.
    per_train = rng.normal(0.5, 1.0, 20)
    cluster = np.repeat(np.arange(20), 50)
    diff = np.repeat(per_train, 50)
    low, high = bf.cluster_ci(cluster, diff, draws=400)
    assert low < diff.mean() < high
    assert high - low > 0.5


# --- the pooled models ------------------------------------------------------


def test_the_sorted_scorers_agree_with_the_general_ones() -> None:
    """The pooled sweep sorts its support once and reuses it across weightings.
    If that shortcut ever disagreed with the plain formula, every pooled number
    would be wrong in a way no table would show."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        x = rng.normal(3.0, 6.0, 40)
        w = rng.random(40) + 0.01
        y = float(rng.normal(3.0, 6.0))
        order = np.argsort(x, kind="stable")
        assert bf.crps_sorted(x[order], w[order], y) == pytest.approx(
            bf.crps_empirical(x, w, y)
        )
        low, high = bf.weighted_quantiles(x, w, (0.1, 0.9))
        assert bf.covered_sorted(x[order], w[order], y) == (low <= y <= high)


def test_a_boost_multiplies_the_query_s_own_runs() -> None:
    w = np.ones(4)
    own = np.array([True, True, False, False])
    assert list(bf.mixed(w, own, ("boost", 4.0))) == [4.0, 4.0, 1.0, 1.0]


def test_a_share_gives_the_own_runs_that_fraction_however_few_they_are() -> None:
    """The reason both knobs are measured: with two own runs against ninety-
    eight of the line's, a multiplier of four still leaves them 8% of the mass
    while a share of 0.7 means 0.7."""
    w = np.ones(100)
    own = np.zeros(100, dtype=bool)
    own[:2] = True
    mixed = bf.mixed(w, own, ("share", 0.7))
    assert mixed[own].sum() == pytest.approx(0.7)
    assert mixed[~own].sum() == pytest.approx(0.3)
    boosted = bf.mixed(w, own, ("boost", 4.0))
    assert boosted[own].sum() / boosted.sum() == pytest.approx(8 / 106, abs=1e-6)


def test_a_mix_with_nothing_on_one_side_changes_nothing() -> None:
    """A share renormalises two groups; with one of them empty there is nothing
    to renormalise and every run must keep the weight it had."""
    w = np.array([1.0, 2.0, 3.0])
    for own in (np.zeros(3, dtype=bool), np.ones(3, dtype=bool)):
        assert list(bf.mixed(w, own, ("share", 0.7))) == [1.0, 2.0, 3.0]
        assert list(bf.mixed(w, own, ("boost", 4.0))) == [1.0, 2.0, 3.0]
    assert list(bf.mixed(w, None, ("share", 0.7))) == [1.0, 2.0, 3.0]


def test_the_pool_counts_the_query_s_own_runs_once() -> None:
    """A line shard contains the query's own runs, so taking both shards whole
    would enter them twice — and then a "boost" of 1 would not be the line."""
    num = (np.array([1, 2], dtype=np.int32), np.array([5.0, 6.0]),
           np.full(2, np.nan), np.zeros(2, dtype=np.int32), None)
    line = (np.array([1, 1, 2], dtype=np.int32), np.array([5.0, 9.0, 6.0]),
            np.full(3, np.nan), np.zeros(3, dtype=np.int32),
            np.array(["4711", "4712", "4711"]))
    combined, own = bf.pooled_history(num, line, "4711")
    assert list(combined[1]) == [5.0, 6.0, 9.0]
    assert list(own) == [True, True, False]


def test_the_pool_keeps_runs_the_line_window_has_already_cut() -> None:
    """The app holds both shards, and the number's is not trimmed — so a run
    older than the line window is still history the user really has."""
    num = (np.array([1, 90], dtype=np.int32), np.array([5.0, 6.0]),
           np.full(2, np.nan), np.zeros(2, dtype=np.int32), None)
    line = (np.array([90], dtype=np.int32), np.array([6.0]),
            np.full(1, np.nan), np.zeros(1, dtype=np.int32), np.array(["4711"]))
    combined, own = bf.pooled_history(num, line, "4711")
    assert list(combined[0]) == [1, 90]
    assert own.all()


def test_a_line_history_knows_which_runs_belong_to_which_number() -> None:
    store = bf.History(
        np.array([1, 1, 2], dtype=np.int32), np.array([480, 481, 480], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]), np.full(3, np.nan), np.zeros(3, dtype=np.int32),
        run_of=np.array(["4711", "4712", "4711"]),
    )
    *_, run_of = store.before(480, 5, bf.TOD_WINDOW_MIN)
    assert list(run_of) == ["4711", "4712", "4711"]


def test_an_unmixed_pool_is_the_line_itself() -> None:
    """The identity the whole sweep is anchored on: a line shard already holds
    the number's runs, so pooling with no reweighting must reproduce it."""
    days = np.arange(1, 21, dtype=np.int32)
    delays = np.linspace(0.0, 10.0, 20)
    hist = (days, delays, np.full(20, np.nan), np.zeros(20, dtype=np.int32))
    own = np.zeros(20, dtype=bool)
    own[:3] = True
    plain, = bf.pooled_scores(hist, own, 30, 1, None, 4.0, (("flat", None),)).values()
    reference = bf.score_one(hist, 30, 1, None, 4.0)
    assert plain[0] == pytest.approx(reference[0])
    assert plain[1] == pytest.approx(reference[1])


def test_the_shrink_share_follows_how_many_own_runs_there_are() -> None:
    """The one weighting that can be right at both ends: all line when the
    number has nothing, all number once it has plenty."""
    w = np.ones(100)
    for own_runs, expected in ((4, 4 / (4 + 8)), (8, 0.5), (64, 64 / 72)):
        own = np.zeros(100, dtype=bool)
        own[:own_runs] = True
        mixed = bf.mixed(w, own, ("shrink", 8.0))
        assert mixed[own].sum() / mixed.sum() == pytest.approx(expected)


def test_shrinking_uses_the_effective_count_not_the_raw_one() -> None:
    """Fifty runs that are all but one stale are not fifty runs, and the gate
    the app switches on counts them the same way."""
    w = np.concatenate([[1.0], np.full(49, 1e-6), np.ones(50)])
    own = np.zeros(100, dtype=bool)
    own[:50] = True
    mixed = bf.mixed(w, own, ("shrink", 8.0))
    # Effective n of the own side is ~1, so the line keeps almost everything.
    assert mixed[own].sum() / mixed.sum() == pytest.approx(1 / 9, abs=0.02)
