"""Whether the live-anchor fix is worth shipping, and how exactly it must be got right.

`calibrate_live.py` established that the app's use of DB's live report is badly
calibrated: it re-anchors on the reported number and keeps 40% of the spread,
which leaves 40% of arrivals above the model's own 90th percentile. This asks
the two questions that decide whether a fix ships.

**How much does the answer depend on sizing the uncertainty correctly?** If the
gain only appears when the width is exactly right, the fix is a liability: the
width is fitted from a fortnight of one region's traffic and the railway moves.
So the width is multiplied by a factor `k` and the whole comparison re-run.
A model that still wins at k=0.5 and k=2 is a model that survives being wrong.

**Is a simple width enough?** Six candidate forms are fitted to raw, unbinned
residuals by pinball loss — the same loss the models are scored on, since CRPS
is twice its average over quantile levels. Binning first would fit the shape of
the bins.

The population is the one the app anchors on: stops DB called at least
`LiveReport.MIN_INFORMATIVE_DELAY_MINUTES` late, which is 4% of all stops and
behaves nothing like the rest. Truth is the archive, not the collector's own
settled forecast — the archive is weeks late, which rules it out for the drift
monitor and rules it *in* here, where the answer is wanted once.

Two deliberate limits. The candidate here answers from the report alone, with
the train's history discarded, exactly as `calibrate_live.py`'s did: it is a
floor, not the proposal, and a real model combining both sources should beat
it. And the fitted residual is measured at a stop's *arrival*; a connection
needs the departure too.

Usage:
    python tools/sensitivity_live.py dataset            # build the cache
    python tools/sensitivity_live.py forms
    python tools/sensitivity_live.py sweep
    python tools/sensitivity_live.py controls
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))

import anchor_drift as ad  # noqa: E402
import score_events as se  # noqa: E402

CACHE = Path(__file__).parent / ".sensitivity/dense.npz"
FORMS_OUT = Path(__file__).parent / ".sensitivity/forms.json"
REGIMES_OUT = Path(__file__).parent / ".sensitivity/regimes.json"
CHANGES_OUT = Path(__file__).parent / ".sensitivity/changes.json"

# Days with both a collector journal and an extracted archive day. The split is
# by date and not at random: the rail-replacement blockade ended on 2026-08-31,
# so fitting on the first block and scoring on the second asks the model to
# survive a regime change rather than to interpolate inside one.
FIT_DAYS = [dt.date(2026, 8, d) for d in range(17, 26)]
# Held out twice over, because "does it generalise" is two questions with
# different answers. HELD_PRE is unseen days in the same regime: what the fit
# costs by being a fit. HELD_POST is the other side of the blockade's end: what
# it costs by the world changing underneath it. A model can pass the first and
# fail the second, and only the second is the risk worth worrying about.
HELD_PRE = [dt.date(2026, 8, d) for d in (29, 30, 31)]
HELD_POST = [dt.date(2026, 9, d) for d in range(1, 5)]
TEST_DAYS = HELD_PRE + HELD_POST
REGIMES = {"fit (in-sample)": FIT_DAYS,
           "held out, same regime": HELD_PRE,
           "held out, after the shift": HELD_POST}

# Lead times probed, minutes. Dense, because the width is a function of lead
# and the point is to fit the function.
PROBES = tuple(range(0, 361, 5)) + (480, 720, 1440)

# Quantile levels. CRPS is twice the average pinball loss over levels, so this
# grid is both the fitting criterion and the scoring one — a model is never
# tuned on something other than what it is judged by.
PS = np.arange(0.0125, 1.0, 0.025)


def crps(quantiles: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of a predictive given as its quantiles at [PS]."""
    d = y[:, None] - quantiles
    return 2.0 * np.mean(np.where(d >= 0, PS * d, (PS - 1.0) * d), axis=1)


def minimise(f, x0) -> tuple[np.ndarray, float]:
    """Nelder-Mead, from SciPy.

    Derivative-free because the loss goes through `np.quantile` of the
    standardised residual, which is not differentiable in the parameters; six
    parameters at most, so a simplex is the right tool. SciPy rather than
    anything written here: the first version of this was hand-rolled, and it
    only ever contracted outwards -- towards a direction already known to be
    worse -- so it stalled instead of shrinking.
    """
    result = optimize.minimize(f, np.asarray(x0, float), method="Nelder-Mead",
                               options={"maxiter": 4000, "xatol": 1e-6,
                                        "fatol": 1e-8})
    return result.x, float(result.fun)


# Widths as a function of lead. Squaring keeps every one of them positive
# without the optimiser ever meeting a boundary.
FORMS = {
    "constant": (1, lambda t, L: np.full_like(L, t[0] ** 2)),
    "linear": (2, lambda t, L: t[0] ** 2 + t[1] ** 2 * L),
    "sqrt": (2, lambda t, L: t[0] ** 2 + t[1] ** 2 * np.sqrt(L)),
    "quadratic": (3, lambda t, L: t[0] ** 2 + t[1] ** 2 * L + t[2] ** 2 * L * L),
    "power": (2, lambda t, L: t[0] ** 2 * np.power(np.maximum(L, 1e-6), t[1])),
    # Growth that levels off: the delay a train can still gain is bounded by
    # how much journey it has left.
    "saturating": (3, lambda t, L: t[0] ** 2
                   + t[1] ** 2 * L / (1.0 + L / (1.0 + t[2] ** 2))),
}

# The median moves with lead too, and more slowly than the width. One form, not
# six: it is not what the question is about, and letting it vary per candidate
# would let a bad width hide behind a flexible centre.
MEAN_PARAMS = 2


def mean_of(theta: np.ndarray, lead: np.ndarray) -> np.ndarray:
    return theta[0] + theta[1] * np.sqrt(np.maximum(lead, 0.0))


def normalise(shape: np.ndarray) -> np.ndarray:
    """Scale a shape so its central 80% is exactly one unit wide.

    Without this the fit is unidentified: only the product of the width and the
    shape is pinned down, so the optimiser is free to halve one and double the
    other. Everything still scores the same, and every reported width becomes a
    number in no particular units — which is worse than useless when the whole
    study is about how wide the answer should be. With it, `s(L)` *is* the
    distance from the 10th percentile to the 90th, in minutes.
    """
    lo, hi = np.searchsorted(PS, 0.1), np.searchsorted(PS, 0.9)
    span = shape[hi] - shape[lo]
    return shape / span if span > 1e-9 else shape


def fit(lead: np.ndarray, eps: np.ndarray, form: str, rounds: int = 4) -> dict:
    """Location, width, and a shared standardised shape, by pinball loss.

    Alternating rather than joint: with the shape held, the location and width
    parameters are a smooth six-dimensional problem; with those held, the shape
    is just the quantiles of the standardised residual, which needs no
    optimiser at all. Four rounds is well past where either stops moving.
    """
    n_width, width_of = FORMS[form]
    theta = np.concatenate([[0.0, 0.1], np.full(n_width, 1.0)])
    shape = normalise(np.quantile(eps - np.median(eps), PS))

    def refit(z: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, float]:
        def loss(t: np.ndarray) -> float:
            q = (mean_of(t[:MEAN_PARAMS], lead)[:, None]
                 + z[None, :] * width_of(t[MEAN_PARAMS:], lead)[:, None])
            return float(crps(q, eps).mean())
        return minimise(loss, start)

    theta, value = refit(shape, theta)
    for _ in range(rounds):
        centred = eps - mean_of(theta[:MEAN_PARAMS], lead)
        scale = np.maximum(width_of(theta[MEAN_PARAMS:], lead), 1e-6)
        shape = normalise(np.quantile(centred / scale, PS))
        theta, value = refit(shape, theta)
    return {"form": form, "theta": theta.tolist(), "shape": shape.tolist(),
            "crps": value}


def predict(model: dict, lead: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Predictive quantiles of the residual, with the width multiplied by `k`."""
    theta = np.asarray(model["theta"])
    shape = np.asarray(model["shape"])
    _, width_of = FORMS[model["form"]]
    return (mean_of(theta[:MEAN_PARAMS], lead)[:, None]
            + k * shape[None, :] * width_of(theta[MEAN_PARAMS:], lead)[:, None])


def asymmetric_laplace(left: float, right: float) -> np.ndarray:
    """A standardised shape from two numbers: exponential tails, unequal rates.

    Offered as the shippable alternative to forty tabulated points. The
    asymmetry is not decoration — a train DB has already called late rarely
    makes the time back and often loses more, so the right tail is the long one.
    """
    return np.where(PS < 0.5, left * np.log(2 * PS), -right * np.log(2 * (1 - PS)))


# --- data --------------------------------------------------------------------

def build_dataset(out: Path, journals: Path, days: list[dt.date]) -> None:
    """Dense (lead, report, archive delay) rows, cached because it is slow."""
    rows = []
    for day in days:
        stops, polls = se.read_day(journals, day)
        events = se.build_events(stops, polls, horizons=PROBES)
        truth_dir = Path(__file__).parent / f".scored/{day}/truthdir"
        truth = se.load_truth([truth_dir], day, {s.eva for s in stops.values()})
        se.attach_truth(events, truth)
        kept = 0
        for e in events:
            if e["cancelled"] or e.get("archive") is None:
                continue
            rows.append((day.toordinal(), e["lead"], e["db"], e["archive"]))
            kept += 1
        print(f"{day}: {len(events)} events, {kept} joined to the archive",
              file=sys.stderr, flush=True)
    a = np.array(rows, float)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, day=a[:, 0], lead=a[:, 1], db=a[:, 2], truth=a[:, 3])
    print(f"{len(a)} rows -> {out}", file=sys.stderr)


def load(cache: Path, days: list[dt.date]) -> tuple[np.ndarray, np.ndarray]:
    """(lead, residual) for the stops the app would anchor on."""
    if not cache.exists():
        raise SystemExit(f"no cache at {cache}; run `dataset` first")
    d = np.load(cache)
    keep = np.isin(d["day"], [x.toordinal() for x in days]) \
        & (d["db"] >= ad.MIN_REPORT)
    return d["lead"][keep], (d["truth"] - d["db"])[keep]


def scored(days: list[dt.date], blind: bool = False) -> np.ndarray:
    """Held-out arrivals as the app itself answered them.

    Columns: lead, report, truth, the shipped model's CRPS, and its stated 10th
    and 90th percentiles. This is the baseline any candidate has to beat, and it
    comes from running the real Kotlin model, not from a reimplementation.
    """
    rows = []
    for day in days:
        live = {}
        path = Path(__file__).parent / f".scored/{day}/arrivals-live.jsonl"
        for line in path.open():
            r = json.loads(line)
            if r["source"] != "EMPIRICAL_LIVE":
                continue
            live[(r["eva"], r["num"], r["planned"], r["tau"])] = r
        other = {}
        if blind:
            path = Path(__file__).parent / f".scored/{day}/arrivals-blind.jsonl"
            for line in path.open():
                r = json.loads(line)
                other[(r["eva"], r["num"], r["planned"], r["tau"])] = r
        for key, r in live.items():
            b = other.get(key) if blind else r
            if b is None:
                continue
            rows.append((r["lead"], r["db"], r["truth"], r["crps"], r["q10"],
                         r["q90"], b["crps"], b["q10"], b["q90"]))
    return np.array(rows, float)


def changes(days: list[dt.date]) -> np.ndarray:
    """Live-anchored one-change journeys, as the app itself answered them.

    Columns: lead to the feeder's scheduled arrival, DB's reported delay, the
    margin (how much later than DB says the feeder may be and still make the
    change), the shipped model's catch probability, DB's yes-or-no, whether it
    was in fact caught, and the feeder's own realised residual.

    The lead is recomputed rather than taken from the record: `build_connections`
    stores minutes until the feeder *sets off*, which is the decision moment,
    while the residual model is indexed by minutes until it *arrives*, which is
    what the forecast is about.
    """
    rows = []
    for day in days:
        path = Path(__file__).parent / f".scored/{day}/connections-live.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if r["source"] != "EMPIRICAL_LIVE" or r["db"] < ad.MIN_REPORT:
                continue
            lead = (se.wall_to_epoch(r["planned"]) - r["read_at"]) / 60
            if lead < 0:
                continue
            rows.append((lead, r["db"], r["threshold"] - r["db"],
                         r["p_catch"], float(r["db_catch_p"]),
                         float(r["caught"]), r["archive"] - r["db"]))
    return np.array(rows, float) if rows else np.empty((0, 7))


def measure(q: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """CRPS, coverage of the stated 80% interval, and the tail that matters.

    "Above q90" is the failure a passenger feels: the train is later than the
    model said it would be nine times in ten, so the connection it promised is
    gone. A model can have a fair CRPS and still get this badly wrong, which is
    exactly what the shipped one does.
    """
    lo, hi = q[:, np.searchsorted(PS, 0.1)], q[:, np.searchsorted(PS, 0.9)]
    return (float(crps(q, truth).mean()),
            float(((truth >= lo) & (truth <= hi)).mean()),
            float((truth > hi).mean()))


# --- the parameter-free alternative ------------------------------------------

def tabulate(lead: np.ndarray, eps: np.ndarray) -> dict:
    """The residual's own quantiles, per lead bin. No functional form at all.

    This is what `tools/calibrate_live.py` proposed and what the report
    described first: bin by lead, keep the empirical distribution of
    (truth - report) in each bin, and read the answer off it. It has no
    parameters to misfit, which is its whole appeal -- and one table row per
    bin per quantile level to carry, keep fresh, and hope stays true, which is
    the cost. Bins are `anchor_drift.EDGES`, the same ones the drift monitor
    watches, so a change there cannot silently split the two.
    """
    b = ad.bin_of(lead)
    rows = []
    for i in range(len(ad.EDGE_LABELS)):
        v = eps[b == i]
        rows.append(np.quantile(v, PS) if v.size >= 30 else None)
    # A bin too thin to speak for itself borrows the nearest one that can,
    # which is what any deployed version of this would have to do too.
    have = [i for i, r in enumerate(rows) if r is not None]
    if not have:
        raise SystemExit("no lead bin has enough residuals to tabulate")
    filled = [rows[i] if rows[i] is not None
              else rows[min(have, key=lambda j: abs(j - i))]
              for i in range(len(rows))]
    return {"kind": "tabulated", "rows": [r.tolist() for r in filled]}


def predict_table(table: dict, lead: np.ndarray) -> np.ndarray:
    return np.asarray(table["rows"], float)[ad.bin_of(lead)]


def quantiles_of(model: dict, lead: np.ndarray) -> np.ndarray:
    """Predictive residual quantiles from either kind of model."""
    return (predict_table(model, lead) if model.get("kind") == "tabulated"
            else predict(model, lead))


def cdf_at(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P(residual <= x) per row, read off quantiles stated at [PS]."""
    return np.array([np.interp(xi, qi, PS, left=0.0, right=1.0)
                     for qi, xi in zip(q, x)])


# --- the three reports -------------------------------------------------------

def report_forms(cache: Path) -> dict:
    lead, eps = load(cache, FIT_DAYS)
    lead_t, eps_t = load(cache, TEST_DAYS)
    print(f"fitted on {len(eps)} residuals, scored on {len(eps_t)}\n")
    print(f"{'width form':<12}{'par':>4}{'fit':>8}{'held out':>10}"
          f"   width at lead 5 / 30 / 120 / 240")
    out = {}
    for form in FORMS:
        model = fit(lead, eps, form)
        held = float(crps(predict(model, lead_t), eps_t).mean())
        n_width, width_of = FORMS[form]
        at = width_of(np.asarray(model["theta"])[MEAN_PARAMS:],
                      np.array([5.0, 30.0, 120.0, 240.0]))
        out[form] = {**model, "held_out": held}
        print(f"{form:<12}{n_width + MEAN_PARAMS:>4}{model['crps']:>8.3f}"
              f"{held:>10.3f}   " + "  ".join(f"{v:5.1f}" for v in at))
    FORMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    FORMS_OUT.write_text(json.dumps(out, indent=2) + "\n")
    return out


def report_sweep(models: dict) -> None:
    a = scored(TEST_DAYS)
    lead, db, truth, shipped, q10, q90 = a[:, :6].T
    print(f"held out: {len(a)} live-anchored arrivals, "
          f"{len(TEST_DAYS)} days the fit never saw\n")
    print(f"{'model':<38}{'CRPS':>8}{'cov80':>8}{'above q90':>11}")
    print(f"{'DB point forecast':<38}{np.abs(db - truth).mean():>8.2f}"
          f"{'-':>8}{'-':>11}")
    print(f"{'as shipped':<38}{shipped.mean():>8.2f}"
          f"{((truth >= q10) & (truth <= q90)).mean():>8.1%}"
          f"{(truth > q90).mean():>11.1%}")
    for form in FORMS:
        c, cov, over = measure(db[:, None] + predict(models[form], lead), truth)
        print(f"{'report + ' + form:<38}{c:>8.2f}{cov:>8.1%}{over:>11.1%}")

    print(f"\nthe width multiplied by k, linear form")
    print(f"{'k':>6}{'CRPS':>8}{'vs shipped':>12}{'cov80':>8}{'above q90':>11}")
    for k in (0.25, 0.4, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0):
        c, cov, over = measure(db[:, None] + predict(models["linear"], lead, k), truth)
        print(f"{k:>6.2f}{c:>8.2f}{c / shipped.mean() - 1:>11.1%}"
              f"{cov:>8.1%}{over:>11.1%}")

    print(f"\nthe tabulated shape replaced by two numbers")
    print(f"{'shape':<38}{'CRPS':>8}{'cov80':>8}{'above q90':>11}")
    shape = np.asarray(models["linear"]["shape"])
    fitted, _ = minimise(
        lambda b: float(np.mean((asymmetric_laplace(b[0] ** 2, b[1] ** 2) - shape) ** 2)),
        [0.5, 0.8])
    left, right = fitted[0] ** 2, fitted[1] ** 2
    for name, z in (("empirical, %d tabulated points" % len(PS), shape),
                    (f"asymmetric Laplace ({left:.2f}, {right:.2f})",
                     asymmetric_laplace(left, right))):
        model = dict(models["linear"], shape=z.tolist())
        c, cov, over = measure(db[:, None] + predict(model, lead), truth)
        print(f"{name:<38}{c:>8.2f}{cov:>8.1%}{over:>11.1%}")


def report_regimes(cache: Path) -> dict:
    """Every candidate, on the fit days and on both kinds of unseen day.

    The question this answers is not "which scores best" -- that was settled --
    but "which of them stops working when the world changes". A tabulated model
    has a row per bin and can track anything; a two-parameter width cannot bend
    to a regime it never saw. Whether that rigidity costs or saves is the whole
    argument for shipping six numbers instead of a table, and it is decidable.
    """
    lead, eps = load(cache, FIT_DAYS)
    table = tabulate(lead, eps)
    linear = fit(lead, eps, "linear")
    shape = np.asarray(linear["shape"])
    two, _ = minimise(
        lambda b: float(np.mean((asymmetric_laplace(b[0] ** 2, b[1] ** 2) - shape) ** 2)),
        [0.5, 0.8])
    left, right = two[0] ** 2, two[1] ** 2
    laplace = dict(linear, shape=asymmetric_laplace(left, right).tolist())
    theta = np.asarray(linear["theta"])
    candidates = {
        "parameter-free (tabulated per bin)": table,
        "fitted width, tabulated shape": linear,
        "fitted width, Laplace shape (6 numbers)": laplace,
    }

    print(f"fitted on {len(eps)} residuals over {len(FIT_DAYS)} days "
          f"before the shift\n")
    print(f"    s(L) = {theta[2] ** 2:.2f} + {theta[3] ** 2:.3f} * L")
    print(f"    m(L) = {theta[0]:.2f} + {theta[1]:.3f} * sqrt(L)")
    print(f"    shape = asymmetric Laplace({left:.2f}, {right:.2f})")
    print(f"    the table it replaces: {len(ad.EDGE_LABELS)} bins x {len(PS)} "
          f"levels = {len(ad.EDGE_LABELS) * len(PS)} numbers\n")

    out: dict = {"model": {"s0": theta[2] ** 2, "s1": theta[3] ** 2,
                           "m0": theta[0], "m1": theta[1],
                           "left": left, "right": right},
                 "regimes": {}}
    head = f"{'model':<42}" + "".join(f"{name:>26}" for name in REGIMES)
    print(head)
    print(f"{'':<42}" + "".join(f"{'CRPS  cov80  >q90':>26}" for _ in REGIMES))
    rows: dict[str, list] = {}
    for name in ["DB point forecast", "as shipped", *candidates]:
        cells = []
        for days in REGIMES.values():
            a = scored(days)
            lead_r, db, truth, shipped, q10, q90 = a[:, :6].T
            if name == "DB point forecast":
                cell = (float(np.abs(db - truth).mean()), None,
                        float((truth > db).mean()))
            elif name == "as shipped":
                cell = (float(shipped.mean()),
                        float(((truth >= q10) & (truth <= q90)).mean()),
                        float((truth > q90).mean()))
            else:
                q = db[:, None] + quantiles_of(candidates[name], lead_r)
                cell = measure(q, truth)
            cells.append(cell)
        rows[name] = cells
        print(f"{name:<42}" + "".join(
            f"{c[0]:>10.2f}{'-' if c[1] is None else format(c[1], '.0%'):>7}"
            f"{c[2]:>9.1%}" for c in cells))
    for i, key in enumerate(REGIMES):
        out["regimes"][key] = {
            "days": [str(d) for d in REGIMES[key]],
            "n": int(len(scored(REGIMES[key]))),
            "models": {name: {"crps": c[i][0], "cov80": c[i][1],
                              "above_q90": c[i][2]}
                       for name, c in rows.items()}}
    REGIMES_OUT.parent.mkdir(parents=True, exist_ok=True)
    REGIMES_OUT.write_text(json.dumps(out, indent=2) + "\n")
    return {"table": table, "linear": linear, "laplace": laplace, **out}


def report_changes(cache: Path) -> dict:
    """Does any of this fix the answer a passenger actually asks?

    An arrival distribution is a means; "will I make my change" is the end, and
    it is a probability, so it is checkable in a way CRPS is not. DB answers
    yes or no. The shipped model answers with a number that is almost always 0
    or 1 anyway, because a two-minute interval leaves no room for doubt. A
    calibrated anchor should be able to say 70% and be right 70% of the time.
    """
    lead, eps = load(cache, FIT_DAYS)
    table = tabulate(lead, eps)
    linear = fit(lead, eps, "linear")
    shape = np.asarray(linear["shape"])
    two, _ = minimise(
        lambda b: float(np.mean((asymmetric_laplace(b[0] ** 2, b[1] ** 2) - shape) ** 2)),
        [0.5, 0.8])
    laplace = dict(linear, shape=asymmetric_laplace(two[0] ** 2, two[1] ** 2).tolist())

    out: dict = {"regimes": {}}
    for key, days in (("held out, after the shift", HELD_POST),
                      ("held out, same regime", HELD_PRE)):
        a = changes(days)
        if not len(a):
            continue
        lead_c, db, margin, shipped_p, db_p, caught, resid = a.T
        answers = {
            # Nine changes in ten work, so a model that has learnt nothing but
            # that still scores well. Without this line a Brier score of 0.07
            # reads as skill when it is the base rate.
            "always the base rate": np.full(len(a), float(caught.mean())),
            "DB: yes or no": db_p,
            "as shipped": shipped_p,
            "parameter-free (tabulated per bin)":
                cdf_at(quantiles_of(table, lead_c), margin),
            "fitted width, Laplace shape (6 numbers)":
                cdf_at(quantiles_of(laplace, lead_c), margin),
        }
        # The changes where the answer is actually in doubt: DB's own number
        # already puts the feeder within ten minutes of losing it. The rest are
        # caught whatever anyone predicts, and they dominate every average.
        tight = margin <= 10
        out["regimes"][key] = {
            "n": int(len(a)), "base": float(caught.mean()),
            "n_tight": int(tight.sum()), "base_tight": float(caught[tight].mean()),
            "days": [str(d) for d in days], "models": {}}
        print(f"\n{key}: {len(a)} live-anchored changes, "
              f"{caught.mean():.1%} caught; {int(tight.sum())} of them tight "
              f"(within 10 min), {caught[tight].mean():.1%} caught\n")
        print(f"{'answer':<42}{'Brier':>8}{'ECE':>8}{'wrong':>8}"
              f"{'Brier/t':>9}{'wrong/t':>9}  reliability: said -> was")
        for name, p in answers.items():
            edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])
            idx = np.searchsorted(edges[1:-1], p, side="right")
            ece, curve = 0.0, []
            for i in range(len(edges) - 1):
                m = idx == i
                if not m.any():
                    curve.append(None)
                    continue
                ece += m.mean() * abs(p[m].mean() - caught[m].mean())
                curve.append({"n": int(m.sum()), "said": float(p[m].mean()),
                              "was": float(caught[m].mean())})
            stats = {
                "brier": float(np.mean((p - caught) ** 2)),
                "ece": float(ece),
                "wrong": float(np.mean((p >= 0.5) != (caught > 0.5))),
                "brier_tight": float(np.mean((p[tight] - caught[tight]) ** 2)),
                "wrong_tight": float(np.mean(
                    (p[tight] >= 0.5) != (caught[tight] > 0.5))),
                "curve": curve}
            out["regimes"][key]["models"][name] = stats
            shown = "  ".join("  -  " if c is None
                              else f"{c['said']:.2f}->{c['was']:.2f}"
                              for c in curve)
            print(f"{name:<42}{stats['brier']:>8.3f}{stats['ece']:>8.3f}"
                  f"{stats['wrong']:>8.1%}{stats['brier_tight']:>9.3f}"
                  f"{stats['wrong_tight']:>9.1%}  {shown}")
        out["regimes"][key]["why"] = why_changes_miss(
            lead_c, db, margin, resid, caught, answers)
    CHANGES_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_OUT.write_text(json.dumps(out, indent=2) + "\n")
    return out


def why_changes_miss(lead: np.ndarray, db: np.ndarray, margin: np.ndarray,
                     resid: np.ndarray, caught: np.ndarray,
                     answers: dict) -> dict:
    """Two reasons a fixed arrival distribution still misjudges a change.

    Worth separating, because one of them is this model's fault and the other
    is the same fault one train further along.
    """
    # 1. The change does not depend on the feeder alone. `caught` compares the
    #    feeder's arrival with the connecting train's *realised* departure,
    #    while the margin was set from DB's forecast of it. A connecting train
    #    that is itself late hands back slack nobody predicted.
    feeder_only = (resid <= margin).astype(float)
    both_ways = {"rescued": float(np.mean((feeder_only == 0) & (caught == 1))),
                 "lost anyway": float(np.mean((feeder_only == 1) & (caught == 0)))}
    print(f"\n  why: the change is two trains, and only one is modelled")
    print(f"    caught in fact                    {caught.mean():>7.1%}")
    print(f"    caught if the feeder were all     {feeder_only.mean():>7.1%}")
    print(f"    the connecting train rescued      {both_ways['rescued']:>7.1%}")
    print(f"    ... and cost                      {both_ways['lost anyway']:>7.1%}")

    # 2. The residual is indexed by lead alone. Its left tail is not: a train
    #    already reported half an hour late has half an hour it can win back,
    #    and one reported two minutes late has two.
    print(f"\n  why: the left tail scales with the delay, not just the lead")
    print(f"    {'reported delay':>16}{'n':>8}{'said':>8}{'feeder made it':>16}")
    tail = []
    for name, p in answers.items():
        if not name.startswith("fitted"):
            continue
        low = p < 0.2
        for lo, hi in ((1, 6), (6, 12), (12, 25), (25, 1e9)):
            m = low & (db >= lo) & (db < hi)
            if m.sum() < 30:
                continue
            row = {"delay": f"{lo}-{'' if hi > 1e8 else int(hi)}",
                   "n": int(m.sum()), "said": float(p[m].mean()),
                   "was": float(feeder_only[m].mean())}
            tail.append(row)
            print(f"    {row['delay']:>16}{row['n']:>8}{row['said']:>8.2f}"
                  f"{row['was']:>16.2f}")
    return {"caught": float(caught.mean()),
            "feeder_only": float(feeder_only.mean()),
            **both_ways, "pessimistic_bucket": tail,
            "feeder_only_scores": {
                name: {"brier": float(np.mean((p - feeder_only) ** 2))}
                for name, p in answers.items()}}


def report_controls(cache: Path, models: dict) -> None:
    a = scored(TEST_DAYS, blind=True)
    lead, db, truth, shipped, q10, q90, bcrps, bq10, bq90 = a.T
    print(f"is the report worth using at all? the same {len(a)} arrivals\n")
    print(f"{'model':<44}{'CRPS':>8}{'cov80':>8}{'above q90':>11}")
    print(f"{'history only, the report ignored':<44}{bcrps.mean():>8.2f}"
          f"{((truth >= bq10) & (truth <= bq90)).mean():>8.1%}"
          f"{(truth > bq90).mean():>11.1%}")
    print(f"{'as shipped: history re-anchored on report':<44}{shipped.mean():>8.2f}"
          f"{((truth >= q10) & (truth <= q90)).mean():>8.1%}"
          f"{(truth > q90).mean():>11.1%}")
    c, cov, over = measure(db[:, None] + predict(models["linear"], lead), truth)
    print(f"{'report + fitted width, history ignored':<44}{c:>8.2f}"
          f"{cov:>8.1%}{over:>11.1%}")

    print("\nfitted on one side of the blockade's end, scored on the other")
    print(f"{'fitted on':<26}{'scored on the fit days':>24}{'scored on the rest':>21}")
    early = load(cache, FIT_DAYS)
    late = load(cache, TEST_DAYS)
    for name, data in (("08-17..08-25 (blockade)", early),
                       ("08-29..09-01 (its end)", late)):
        model = fit(*data, "linear")
        theta = np.asarray(model["theta"])
        on_early = float(crps(predict(model, early[0]), early[1]).mean())
        on_late = float(crps(predict(model, late[0]), late[1]).mean())
        print(f"{name:<26}{on_early:>24.3f}{on_late:>21.3f}"
              f"   s(L)={theta[MEAN_PARAMS] ** 2:.2f}"
              f"+{theta[MEAN_PARAMS + 1] ** 2:.4f}L")

    print("\ncoverage of the stated 80% interval, by lead")
    print(f"{'lead (min)':<12}{'n':>7}{'shipped':>10}{'constant':>10}{'linear':>10}")
    flat = db[:, None] + predict(models["constant"], lead)
    sloped = db[:, None] + predict(models["linear"], lead)
    i10, i90 = np.searchsorted(PS, 0.1), np.searchsorted(PS, 0.9)
    for lo, hi in zip((0, 15, 30, 60, 120), (15, 30, 60, 120, 10 ** 9)):
        rows = (lead >= lo) & (lead < hi)
        if rows.sum() < 100:
            continue
        def covered(low, high):
            return ((truth[rows] >= low[rows]) & (truth[rows] <= high[rows])).mean()
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"{label:<12}{rows.sum():>7}{covered(q10, q90):>10.1%}"
              f"{covered(flat[:, i10], flat[:, i90]):>10.1%}"
              f"{covered(sloped[:, i10], sloped[:, i90]):>10.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["dataset", "forms", "sweep", "controls",
                                        "regimes", "changes"])
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--journals", type=Path,
                    default=Path(__file__).parent / ".forecasts")
    args = ap.parse_args()

    if args.command == "dataset":
        build_dataset(args.cache, args.journals, FIT_DAYS + TEST_DAYS)
        return 0
    if args.command == "forms":
        report_forms(args.cache)
        return 0
    if args.command == "regimes":
        report_regimes(args.cache)
        return 0
    if args.command == "changes":
        report_changes(args.cache)
        return 0
    models = (json.loads(FORMS_OUT.read_text()) if FORMS_OUT.exists()
              else report_forms(args.cache))
    if args.command == "sweep":
        report_sweep(models)
    else:
        report_controls(args.cache, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
