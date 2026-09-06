"""Tests for the live-anchor sensitivity study.

The study's conclusion is a shipping decision, so the parts that could make it
wrong are pinned here: the loss it fits and scores by must actually be CRPS,
the optimiser must find a minimum, and the fit must recover a width it is given
rather than one it likes. Without the last of those, "the quadratic buys
nothing over the linear" could just as well mean the optimiser never found the
quadratic.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import sensitivity_live as sl  # noqa: E402


def test_crps_of_a_point_forecast_is_the_absolute_error():
    """The identity every CRPS implementation must satisfy first."""
    y = np.array([3.0, -2.0, 0.0])
    q = np.repeat(np.array([[1.0, 1.0, 1.0]]).T, len(sl.PS), axis=1)
    assert np.allclose(sl.crps(q, y), [2.0, 3.0, 1.0])


def test_crps_rewards_a_sharper_forecast_that_is_still_right():
    y = np.zeros(1)
    tight = np.quantile(np.random.default_rng(0).normal(0, 1, 20000), sl.PS)[None, :]
    loose = np.quantile(np.random.default_rng(0).normal(0, 4, 20000), sl.PS)[None, :]
    assert sl.crps(tight, y)[0] < sl.crps(loose, y)[0]


def test_every_width_form_is_positive_and_only_one_ignores_the_lead():
    lead = np.array([0.0, 1.0, 30.0, 500.0])
    for name, (n_params, width_of) in sl.FORMS.items():
        w = width_of(np.full(n_params, 0.7), lead)
        assert np.all(w > 0), f"{name} produced a non-positive width"
        varies = float(w.max() - w.min())
        assert (varies == 0.0) == (name == "constant"), name


def test_the_fit_recovers_a_width_it_was_given():
    """Synthetic residuals with a known linear width, and no other structure.

    This is what makes the form comparison in the study readable. If the fit
    could not recover a width that is really there, "the quadratic buys
    nothing" would be a statement about the optimiser, not about the railway.
    """
    rng = np.random.default_rng(0)
    lead = rng.uniform(0, 240, 40_000)
    width = 4.0 + 0.12 * lead                     # p90 - p10 of the residual
    z = rng.normal(0, 1, lead.size)
    # A normal's central 80% is 2.563 standard deviations wide.
    eps = 2.0 + width * z / 2.563
    model = sl.fit(lead, eps, "linear")
    theta = np.asarray(model["theta"])
    got = sl.FORMS["linear"][1](theta[sl.MEAN_PARAMS:], np.array([0.0, 240.0]))
    assert np.allclose(got, [4.0, 4.0 + 0.12 * 240], atol=1.5)


def test_a_richer_form_does_not_score_worse_where_it_is_fitted():
    """The quadratic contains the linear, so in-sample it cannot lose.

    It does lose held out in the real study, which is the finding; this pins
    that the finding is about generalisation and not about a failed fit.
    """
    rng = np.random.default_rng(1)
    lead = rng.uniform(0, 240, 20_000)
    eps = 1.0 + (3.0 + 0.1 * lead) * rng.normal(0, 1, lead.size) / 2.563
    linear = sl.fit(lead, eps, "linear")["crps"]
    quadratic = sl.fit(lead, eps, "quadratic")["crps"]
    assert quadratic <= linear + 0.02


def test_k_scales_the_spread_and_leaves_the_centre_alone():
    middle = len(sl.PS) // 2
    shape = np.arange(len(sl.PS), dtype=float) - middle   # exactly zero at `middle`
    model = {"form": "constant", "theta": [1.0, 0.0, 2.0], "shape": shape.tolist()}
    one = sl.predict(model, np.array([10.0]), k=1.0)[0]
    half = sl.predict(model, np.array([10.0]), k=0.5)[0]
    assert one[middle] == half[middle] == 1.0, "the centre must not move with k"
    assert np.isclose(one.max() - one.min(), 2 * (half.max() - half.min()))


def test_the_asymmetric_shape_leans_the_way_delays_do():
    """A train DB has called late rarely makes the time back, often loses more."""
    z = sl.asymmetric_laplace(0.25, 0.87)
    assert np.all(np.diff(z) > 0), "a quantile function must increase"
    middle = len(sl.PS) // 2
    assert abs(z[middle]) < 0.05
    assert z[-1] > -z[0], "the right tail must be the long one"


def test_measure_reports_the_interval_it_states():
    """Half the outcomes inside a hand-made 80% interval, a quarter above it."""
    q = np.repeat(np.linspace(-10, 10, len(sl.PS))[None, :], 4, axis=0)
    lo, hi = q[0, np.searchsorted(sl.PS, 0.1)], q[0, np.searchsorted(sl.PS, 0.9)]
    truth = np.array([0.0, lo + 0.1, hi + 1.0, hi + 2.0])
    _, coverage, above = sl.measure(q, truth)
    assert coverage == 0.5
    assert above == 0.5


def test_the_fit_and_test_days_do_not_overlap():
    """The whole point of the split is that they are different regimes."""
    assert not set(sl.FIT_DAYS) & set(sl.TEST_DAYS)
    assert max(sl.FIT_DAYS) < min(sl.TEST_DAYS)


def test_the_population_is_the_one_the_app_anchors_on(tmp_path):
    """Shared with the drift monitor, so a change there cannot silently split.

    The study fits the width and the monitor watches it; measured over
    different populations they would be two curves with one name.
    """
    import anchor_drift as ad
    cache = tmp_path / "dense.npz"
    np.savez(cache, day=np.array([1.0, 1.0, 1.0]), lead=np.array([10.0, 20.0, 30.0]),
             db=np.array([0.0, ad.MIN_REPORT, 5.0]), truth=np.array([4.0, 3.0, 9.0]))
    import datetime as dt
    lead, eps = sl.load(cache, [dt.date.fromordinal(1)])
    assert lead.tolist() == [20.0, 30.0]
    assert eps.tolist() == [3.0 - ad.MIN_REPORT, 4.0]


# --- the parameter-free alternative and the change question ------------------

def test_the_tabulated_model_is_the_empirical_quantiles_of_its_own_bin():
    """No form, no fit: each bin answers with the residuals it was given."""
    rng = np.random.default_rng(0)
    lead = np.concatenate([np.full(500, 7.0), np.full(500, 100.0)])
    eps = np.concatenate([rng.normal(1.0, 2.0, 500), rng.normal(9.0, 20.0, 500)])
    q = sl.predict_table(sl.tabulate(lead, eps), np.array([7.0, 100.0]))
    assert np.allclose(q[0], np.quantile(eps[:500], sl.PS))
    assert np.allclose(q[1], np.quantile(eps[500:], sl.PS))


def test_a_bin_too_thin_to_speak_borrows_the_nearest_one_that_can():
    """The failure mode a table has and a fitted curve does not.

    Every deployed tabulation needs this rule, so the comparison has to include
    it rather than quietly scoring a model that answers `None` beyond an hour.
    """
    lead = np.concatenate([np.full(400, 7.0), np.full(5, 100.0)])
    eps = np.concatenate([np.linspace(-3, 3, 400), np.full(5, 40.0)])
    table = sl.tabulate(lead, eps)
    near, far = sl.predict_table(table, np.array([7.0, 100.0]))
    assert np.allclose(near, far), "the thin bin did not borrow"
    assert far.max() < 40.0, "it borrowed, so it cannot report the thin sample"


def test_cdf_at_reads_a_probability_off_the_quantiles():
    q = np.linspace(-10.0, 10.0, len(sl.PS))[None, :]
    got = sl.cdf_at(np.repeat(q, 3, axis=0), np.array([-10.0, 0.0, 10.0]))
    assert got[0] == pytest.approx(sl.PS[0])
    assert got[1] == pytest.approx(0.5, abs=0.02)
    assert got[2] == pytest.approx(sl.PS[-1])


def test_cdf_at_saturates_outside_the_stated_quantiles():
    """A margin beyond every quantile is a certainty, not an extrapolation."""
    q = np.linspace(-5.0, 5.0, len(sl.PS))[None, :]
    assert sl.cdf_at(q, np.array([-99.0]))[0] == 0.0
    assert sl.cdf_at(q, np.array([99.0]))[0] == 1.0


def test_the_three_day_sets_are_disjoint_and_in_order():
    """Fit, then unseen days in the same regime, then unseen days after it.

    The order carries the argument: if the middle column holds and the right
    one does not, the model did not overfit — the world moved.
    """
    assert max(sl.FIT_DAYS) < min(sl.HELD_PRE) < max(sl.HELD_PRE) < min(sl.HELD_POST)
    assert not set(sl.FIT_DAYS) & set(sl.HELD_PRE) & set(sl.HELD_POST)
    assert sl.TEST_DAYS == sl.HELD_PRE + sl.HELD_POST
    assert min(sl.HELD_POST) > dt.date(2026, 8, 31), \
        "the shift is the blockade ending on 2026-08-31"


# --- the change is two trains ------------------------------------------------

def test_a_change_is_the_difference_of_two_report_errors():
    """The identity the whole connection analysis rests on.

    A change works when the feeder's arrival error, minus the connecting
    train's departure error, is inside the margin. Modelling only the first
    term is the shipped bug; modelling neither is what came before it.
    """
    rng = np.random.default_rng(3)
    margin = rng.normal(5, 8, 500)
    a, d = rng.normal(2, 6, 500), rng.normal(1, 4, 500)
    caught = (margin >= a - d)
    # Stated the other way round, as `build_connections` computes it.
    assert np.array_equal(caught, (a - margin <= d))


def test_p_difference_reduces_to_the_plain_cdf_when_the_departure_is_certain():
    """A point mass at zero is exactly what the shipped model assumes."""
    q = np.linspace(-20.0, 20.0, len(sl.PS))[None, :].repeat(4, axis=0)
    zero = np.zeros_like(q)
    margin = np.array([-10.0, 0.0, 5.0, 12.0])
    assert np.allclose(sl.p_difference(q, zero, margin),
                       sl.cdf_at(q, margin), atol=1.0 / len(sl.PS) + 1e-9)


def test_a_late_connecting_train_makes_the_change_easier():
    """It hands back slack, so every probability must rise, never fall."""
    q = np.linspace(-20.0, 20.0, len(sl.PS))[None, :].repeat(3, axis=0)
    margin = np.array([-5.0, 0.0, 5.0])
    certain = sl.p_difference(q, np.zeros_like(q), margin)
    late = sl.p_difference(q, np.full_like(q, 3.0), margin)
    assert np.all(late >= certain)
    assert np.any(late > certain), "a three-minute gift changed nothing"


def test_p_difference_is_chunk_independent():
    """The chunking is a memory device and must not touch the answer."""
    rng = np.random.default_rng(1)
    q = np.sort(rng.normal(0, 5, (37, len(sl.PS))), axis=1)
    d = np.sort(rng.normal(1, 3, (37, len(sl.PS))), axis=1)
    margin = rng.normal(0, 4, 37)
    assert np.allclose(sl.p_difference(q, d, margin, chunk=5),
                       sl.p_difference(q, d, margin, chunk=1000))


def test_the_laplace_fit_carries_a_two_number_shape():
    rng = np.random.default_rng(2)
    lead = rng.uniform(0, 120, 1500)
    eps = rng.laplace(0.0, 1.0 + 0.02 * lead)
    model = sl.laplace_fit(lead, eps)
    shape = np.asarray(model["shape"])
    assert shape.shape == sl.PS.shape
    assert np.all(np.diff(shape) >= -1e-9), "a shape must be non-decreasing"
