"""Tests for the live-anchor sensitivity study.

The study's conclusion is a shipping decision, so the parts that could make it
wrong are pinned here: the loss it fits and scores by must actually be CRPS,
the optimiser must find a minimum, and the fit must recover a width it is given
rather than one it likes. Without the last of those, "the quadratic buys
nothing over the linear" could just as well mean the optimiser never found the
quadratic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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


def test_the_optimiser_finds_a_known_minimum():
    """Rosenbrock: curved valley, minimum at (1, 1), a standard trap."""
    def f(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    best, value = sl.nelder_mead(f, [-1.2, 1.0], [0.5, 0.5], iters=4000)
    assert value < 1e-6
    assert np.allclose(best, [1.0, 1.0], atol=1e-2)


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
