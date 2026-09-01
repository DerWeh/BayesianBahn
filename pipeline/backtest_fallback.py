"""Does a line-keyed history beat the class-wide prior where the number has none?

The app looks a train up by its own run number. IRIS renumbers runs at every
timetable change, so a train that has been running for years can arrive with no
history at all, and [Predictor] then answers from the class-wide Bayesian prior
— a single Student-t per train class that knows nothing about the station, the
hour or the line. The README has proposed a line-keyed shard as the fallback
for that case since the beginning, and `HistoryRepository.candidateKeys`
already asks for one; `build_shards.py` has never written it.

This decides whether writing it is worth the bytes, and it is a different
question from the one `backtest.py --group-by line` answers. That one asks
whether line-keying is a better *primary* key, measured over connections whose
number has plenty of history, and the answer is no. This one asks what to do on
the events where the number has *nothing*, where the incumbent is not the
number-keyed history but the prior. Those events are excluded from the other
backtest by construction (`--min-connection-runs`), so it has never seen them.

Walk-forward, mirroring the app exactly: recency half-life 30 days, same-weekday
boost 2, a circular 20-minute time-of-day window, and the same effective-sample
floor of 8 that decides when [Predictor] gives up and takes the prior. Every
prediction for an event on day D uses only runs strictly before D.

Two scenarios:

  - "blind"  no live information — planning the day before;
  - "live"   the delay measured at the previous stop is known, gated by the
             app's rule that a report counts only when it reports a delay.

The live scenario is a *control*, not the forward evaluation: the archive
records what trains did, not what DB said they would do, so a measured
previous-stop delay is better information than the number the app receives.
See `backtest.py`'s module docstring for why that distinction is the whole
reason two datasets exist.

The two windows the published figures come from — the second sits across the
December timetable change, where renumbering makes the population that needs the
fallback about twice as large:

    STATIONS=$(grep -ho '^[0-9]*' tools/forecast_stations.csv \
                   tools/forecast_stations_cohort2.csv | grep . | paste -sd,)
    pixi run -e pipeline python pipeline/backtest_fallback.py \
        --data-dir pipeline/data --stations "$STATIONS" --eval-weeks 12
    pixi run -e pipeline python pipeline/backtest_fallback.py \
        --data-dir pipeline/data --stations "$STATIONS" \
        --eval-from 2025-12-15 --eval-to 2026-01-31
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from backtest import LONG_DISTANCE, MIN_INFORMATIVE_DELAY, load_connections
from build_shards import LINE_DAYS

# --------------------------------------------------------------------- the app

# Mirrors EmpiricalDelay's companion object. Guarded by
# pipeline/tests/test_backtest_fallback.py, which reads the Kotlin and compares:
# a backtest that has drifted from the app measures a model nobody ships.
HALF_LIFE_DAYS = 30.0
SAME_WEEKDAY_BOOST = 2.0
TOD_WINDOW_MIN = 20
MIN_EFFECTIVE_N = 8.0
LIVE_KERNEL_FLOOR = 0.15
LIVE_BANDWIDTH = 0.3

# Mirrors DelayModel: the fallback model the app ships is never trained — every
# `Predictor` is built with a fresh `DelayModel()` — so the posterior predictive
# is the prior itself, one fixed Student-t per class, the same in every time
# band. That is exactly why it is a weak forecast and why this file exists.
LIVE_SHRINKAGE = 0.4
MIN_LIVE_SCALE = 1.2
PRIORS = {  # class -> (mu0, kappa0, alpha0, beta0)
    "long_distance": (4.0, 4.0, 3.0, 128.0),
    "regional": (1.5, 4.0, 3.0, 32.0),
    "sbahn": (1.0, 4.0, 3.0, 18.0),
    "other": (2.0, 4.0, 3.0, 50.0),
}
REGIONAL_CATEGORIES = {"RE", "RB", "IRE", "MEX", "TER", "BRB", "AG"}

LN2 = math.log(2.0)


def train_class(category: str) -> str:
    """Mirrors TrainClass.fromCategory — the class picks the prior."""
    upper = category.upper()
    if upper in LONG_DISTANCE:
        return "long_distance"
    if upper == "S":
        return "sbahn"
    if upper in REGIONAL_CATEGORIES:
        return "regional"
    if category.strip() and category[0].isalpha():
        return "regional"
    return "other"


def prior_shape(train_cls: str) -> tuple[float, float, float]:
    """(df, loc, scale) of the untrained posterior predictive."""
    mu0, kappa0, alpha0, beta0 = PRIORS[train_cls]
    scale = math.sqrt(beta0 * (kappa0 + 1.0) / (alpha0 * kappa0))
    return 2.0 * alpha0, mu0, scale


class StudentTScores:
    """CRPS and quantiles of a location-scale Student-t, off a precomputed grid.

    The prior is the same distribution for every event of a class, so its scores
    depend on nothing but the observation — and under a live report it is the
    same distribution again, only shifted to sit on the report. One grid per
    (class, scenario) therefore serves every event, which is what makes scoring
    a closed-form distribution as cheap as scoring an empirical one. Quadrature
    rather than the closed form for the t: this environment has no scipy, and
    a grid needs nothing but exp and log.
    """

    def __init__(self, df: float, scale: float, half_width: float = 2000.0,
                 step: float = 0.05) -> None:
        x = np.arange(-half_width, half_width + step, step)
        pdf = np.exp(-(df + 1.0) / 2.0 * np.log1p((x / scale) ** 2 / df))
        pdf /= np.trapezoid(pdf, x)
        cdf = np.concatenate(([0.0], np.cumsum((pdf[1:] + pdf[:-1]) * 0.5 * step)))
        cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
        # CRPS(y) = int_{-inf}^{y} F^2 + int_{y}^{inf} (1-F)^2, both cumulative,
        # so a scored event is two interpolations rather than an integral.
        f2 = cdf**2
        g2 = (1.0 - cdf) ** 2
        self._x = x
        self._cdf = cdf
        self._left = np.concatenate(([0.0], np.cumsum((f2[1:] + f2[:-1]) * 0.5 * step)))
        right = np.concatenate(([0.0], np.cumsum((g2[1:] + g2[:-1]) * 0.5 * step)))
        self._right = right[-1] - right

    def crps(self, z: np.ndarray) -> np.ndarray:
        """CRPS against observations already shifted to the distribution's centre."""
        return np.interp(z, self._x, self._left) + np.interp(z, self._x, self._right)

    def quantile(self, p: float) -> float:
        return float(np.interp(p, self._cdf, self._x))


def prior_grids() -> dict[tuple[str, str], StudentTScores]:
    """(class, scenario) -> scorer, centred on zero.

    Both grids are centred because the caller has already subtracted whatever
    the prior is sitting on: the class mean when blind, the live report when
    not. The live grid is the same shape with the app's shrunken scale.
    """
    grids = {}
    for train_cls in PRIORS:
        df, _, scale = prior_shape(train_cls)
        grids[(train_cls, "blind")] = StudentTScores(df, scale)
        grids[(train_cls, "live")] = StudentTScores(
            df, max(scale * LIVE_SHRINKAGE, MIN_LIVE_SCALE)
        )
    return grids


# ------------------------------------------------------------------- histories


class History:
    """Runs of one identity at one station, answering "what was known by day D".

    Selecting the time-of-day window is the expensive part and depends only on
    the query's own planned time, of which there are few per identity — a line
    calls at a station on a handful of slots and repeats them for months. So the
    window is computed once per slot and cached, and each event is then a binary
    search for the day cutoff over an already sorted array.
    """

    __slots__ = ("day", "tod", "delay", "prev", "weekday", "_cache")

    def __init__(self, day, tod, delay, prev, weekday) -> None:
        order = np.argsort(day, kind="stable")
        self.day = day[order].astype(np.int32)
        # int32 on purpose: the column arrives as int8 and `24 * 60 - gap`
        # overflows it, the same trap backtest.py fell into.
        self.tod = tod[order].astype(np.int32)
        self.delay = delay[order]
        self.prev = prev[order]
        self.weekday = weekday[order]
        self._cache: dict[tuple[int, int], tuple] = {}

    def before(self, query_tod: int, query_day: int, window: int,
               max_days: int | None = None) -> tuple:
        key = (query_tod, window)
        slot = self._cache.get(key)
        if slot is None:
            gap = np.abs(self.tod - query_tod)
            keep = np.minimum(gap, 24 * 60 - gap) <= window
            slot = (self.day[keep], self.delay[keep], self.prev[keep],
                    self.weekday[keep])
            self._cache[key] = slot
        days, delay, prev, weekday = slot
        cut = int(np.searchsorted(days, query_day, side="left"))
        # A published line shard carries only its most recent days, so a
        # backtest that reads every run of the line measures a shard nobody
        # will ever hold.
        start = (0 if max_days is None
                 else int(np.searchsorted(days, query_day - max_days, side="left")))
        return days[start:cut], delay[start:cut], prev[start:cut], weekday[start:cut]


def effective_n(w: np.ndarray) -> float:
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    return s1 * s1 / s2 if s2 > 0 else 0.0


def empirical(hist: tuple, query_day: int, query_weekday: int,
              live: float | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Mirror of EmpiricalDelay.build: (support, weights), or None when empty.

    The gate on effective sample size is the caller's, as it is [Predictor]'s.
    """
    days, delay, prev, weekday = hist
    if days.size == 0:
        return None
    w = np.exp(-LN2 / HALF_LIFE_DAYS * (query_day - days))
    w = np.where(weekday == query_weekday, w * SAME_WEEKDAY_BOOST, w)
    if live is not None:
        known = ~np.isnan(prev)
        if int(known.sum()) >= MIN_EFFECTIVE_N:
            bandwidth = max(3.0, LIVE_BANDWIDTH * abs(live))
            z = (prev[known] - live) / bandwidth
            kernel = LIVE_KERNEL_FLOOR + np.exp(-0.5 * z * z)
            return live + (delay[known] - prev[known]), w[known] * kernel
    return delay, w


# --------------------------------------------------------------------- scoring


def crps_empirical(x: np.ndarray, w: np.ndarray, y: float) -> float:
    """Exact CRPS of a weighted empirical distribution — same formula as backtest.py."""
    w = w / w.sum()
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    exx = 2.0 * float(np.sum(ws * xs * ((cw - ws) - (1.0 - cw))))
    return float(np.sum(w * np.abs(x - y)) - 0.5 * exx)


def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs) -> list[float]:
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws) / ws.sum()
    return [float(xs[min(int(np.searchsorted(cw, q)), len(xs) - 1)]) for q in qs]


def gated(live: float | None) -> float | None:
    """The app's rule: a report counts only when it reports a delay."""
    if live is None or live < MIN_INFORMATIVE_DELAY:
        return None
    return live


# ------------------------------------------------------------------- the sweep


MODELS = ("number", "line", "thin", "prior", "shipped", "proposed")


class Recorder:
    """Per-event scores, one flat column each.

    Kept per event rather than accumulated because the interesting comparisons
    are paired — the same event scored by the prior and by the line — and
    because the confidence interval resamples whole trains, which needs to know
    which event belonged to which.
    """

    FIELDS = ("cluster", "scenario", "bucket", "num_ok", "line_ok", "y",
              "crps_number", "crps_line", "crps_thin", "prior_z",
              "cov_number", "cov_line", "cov_thin")

    def __init__(self) -> None:
        self.rows: dict[str, list] = {f: [] for f in self.FIELDS}

    def add(self, **values) -> None:
        for field, column in self.rows.items():
            column.append(values[field])

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows)


def score_one(hist, query_day, query_weekday, live, y) -> tuple[float, float, bool]:
    """(CRPS, effective n, covered by the 80% interval) of an empirical history."""
    built = empirical(hist, query_day, query_weekday, live)
    if built is None:
        return math.nan, 0.0, False
    x, w = built
    if w.sum() <= 0:
        return math.nan, 0.0, False
    q10, q90 = weighted_quantiles(x, w, (0.1, 0.9))
    return crps_empirical(x, w, y), effective_n(w), q10 <= y <= q90


def sweep(df: pl.DataFrame, eval_from: int, eval_to: int, line_window: int,
          line_days: int | None) -> pl.DataFrame:
    """Walk forward over every event in the window, scoring each candidate."""
    keyed = df.with_columns(
        day=pl.col("date").cast(pl.Int32),
        weekday=pl.col("date").dt.weekday().cast(pl.Int32),
        line=pl.col("line_number").fill_null(""),
    )

    def histories(ident: str) -> dict[tuple, History]:
        built = {}
        for key, grp in keyed.group_by(["eva", "train_type", ident],
                                       maintain_order=False):
            if ident == "line" and not key[2]:
                continue
            built[key] = History(
                grp["day"].to_numpy(), grp["tod_min"].to_numpy(),
                grp["delay"].to_numpy(), grp["prev"].to_numpy(),
                grp["weekday"].to_numpy(),
            )
        return built

    by_line = histories("line")
    print(f"  {len(by_line)} (station, line) histories", flush=True)

    out = Recorder()
    cluster = 0
    for key, grp in keyed.group_by(["eva", "train_type", "train_number"],
                                   maintain_order=False):
        eva, ttype, _ = key
        number = History(
            grp["day"].to_numpy(), grp["tod_min"].to_numpy(),
            grp["delay"].to_numpy(), grp["prev"].to_numpy(),
            grp["weekday"].to_numpy(),
        )
        bucket = train_class(str(ttype))
        _, loc, _ = prior_shape(bucket)
        cluster += 1

        window = grp.filter(
            (pl.col("day") >= eval_from) & (pl.col("day") <= eval_to)
        )
        if window.height == 0:
            continue
        line_key = (eva, ttype, window["line"][0])
        line = by_line.get(line_key)

        for row in window.iter_rows(named=True):
            qday, qtod, qwd = row["day"], int(row["tod_min"]), row["weekday"]
            y = float(row["delay"])
            num_hist = number.before(qtod, qday, TOD_WINDOW_MIN)
            line_hist = (line.before(qtod, qday, line_window, line_days) if line is not None
                         else (np.empty(0), np.empty(0), np.empty(0), np.empty(0)))
            reported = None if row["prev"] is None else gated(float(row["prev"]))

            for scenario in ("blind", "live"):
                # The live scenario is scored only where a report exists and
                # says something: everywhere else the app's live path and its
                # blind path are the same code, so a second copy of the blind
                # numbers would only dilute the comparison.
                live = None if scenario == "blind" else reported
                if scenario == "live" and live is None:
                    continue
                crps_num, eff_num, cov_num = score_one(num_hist, qday, qwd, live, y)
                crps_line, eff_line, cov_line = score_one(line_hist, qday, qwd, live, y)
                out.add(
                    cluster=cluster, scenario=scenario, bucket=bucket,
                    num_ok=eff_num >= MIN_EFFECTIVE_N,
                    line_ok=eff_line >= MIN_EFFECTIVE_N,
                    y=y,
                    crps_number=crps_num if eff_num >= MIN_EFFECTIVE_N else math.nan,
                    crps_line=crps_line if eff_line >= MIN_EFFECTIVE_N else math.nan,
                    crps_thin=crps_num,
                    prior_z=y - loc if live is None else y - live,
                    cov_number=cov_num, cov_line=cov_line, cov_thin=cov_num,
                )
    return out.frame()


# ------------------------------------------------------------------- reporting


def with_prior(scored: pl.DataFrame) -> pl.DataFrame:
    """Adds the prior's score, and the two cascades the decision is between.

    `shipped` is what the app does today: the number's history where it has
    one, the class prior everywhere else. `proposed` inserts the line-keyed
    history between the two. They are identical on every event the number
    covers, which is why the whole difference lives in the thin population and
    why the end-to-end figure is so much smaller than the one measured there.
    """
    grids = prior_grids()
    frames = []
    for (bucket, scenario), grp in scored.group_by(["bucket", "scenario"]):
        scorer = grids[(bucket, scenario)]
        z = grp["prior_z"].to_numpy()
        low, high = scorer.quantile(0.1), scorer.quantile(0.9)
        frames.append(
            grp.with_columns(
                crps_prior=pl.Series(scorer.crps(z)),
                cov_prior=pl.Series((z >= low) & (z <= high)),
            )
        )
    joined = pl.concat(frames)
    fallback_crps = pl.when(pl.col("line_ok")).then(pl.col("crps_line")).otherwise(
        pl.col("crps_prior")
    )
    fallback_cov = pl.when(pl.col("line_ok")).then(pl.col("cov_line")).otherwise(
        pl.col("cov_prior")
    )
    return joined.with_columns(
        crps_shipped=pl.when(pl.col("num_ok")).then(pl.col("crps_number")).otherwise(
            pl.col("crps_prior")
        ),
        cov_shipped=pl.when(pl.col("num_ok")).then(pl.col("cov_number")).otherwise(
            pl.col("cov_prior")
        ),
        crps_proposed=pl.when(pl.col("num_ok")).then(pl.col("crps_number")).otherwise(
            fallback_crps
        ),
        cov_proposed=pl.when(pl.col("num_ok")).then(pl.col("cov_number")).otherwise(
            fallback_cov
        ),
    )


def cluster_ci(cluster: np.ndarray, diff: np.ndarray, draws: int = 1000,
               seed: int = 20260901) -> tuple[float, float]:
    """95% interval for a mean difference, resampling whole trains.

    Events of one train are not independent — a train that runs late runs late
    all week — so an interval that resamples events would be far too narrow.
    Same choice, and the same reason, as the forward evaluation's intervals.
    """
    if diff.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(cluster, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=diff, minlength=n_clusters)
    counts = np.bincount(inverse, minlength=n_clusters).astype(float)
    means = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        total = counts[pick].sum()
        means[i] = sums[pick].sum() / total if total else math.nan
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(rows: pl.DataFrame, models) -> list[dict]:
    out = []
    for model in models:
        crps, cov = f"crps_{model}", f"cov_{model}"
        kept = rows.filter(pl.col(crps).is_not_nan() & pl.col(crps).is_not_null())
        if kept.height == 0:
            continue
        out.append({
            "model": model,
            "n": kept.height,
            "crps": round(float(kept[crps].mean()), 3),
            "crps_p90": round(float(kept[crps].quantile(0.9)), 3),
            "coverage80": round(float(kept[cov].mean()), 3),
        })
    return out


def compare(rows: pl.DataFrame, left: str, right: str) -> dict:
    """Paired difference `left - right` over the events both models answered."""
    both = rows.filter(
        pl.col(f"crps_{left}").is_not_nan() & pl.col(f"crps_{right}").is_not_nan()
    )
    if both.height == 0:
        return {"n": 0}
    diff = (both[f"crps_{left}"] - both[f"crps_{right}"]).to_numpy()
    low, high = cluster_ci(both["cluster"].to_numpy(), diff)
    return {
        "n": both.height,
        "clusters": int(both["cluster"].n_unique()),
        f"crps_{left}": round(float(both[f"crps_{left}"].mean()), 3),
        f"crps_{right}": round(float(both[f"crps_{right}"].mean()), 3),
        "delta": round(float(diff.mean()), 3),
        "ci95": [round(low, 3), round(high, 3)],
    }


def report(scored: pl.DataFrame, out: Path | None) -> None:
    result: dict[str, object] = {}
    for scenario in ("blind", "live"):
        rows = scored.filter(pl.col("scenario") == scenario)
        if rows.height == 0:
            continue
        thin = rows.filter(~pl.col("num_ok"))
        rich = rows.filter(pl.col("num_ok"))
        share = thin.height / rows.height
        covered = (
            thin.filter(pl.col("line_ok")).height / thin.height if thin.height else 0.0
        )
        print(f"\n=== {scenario} ===")
        print(f"{rows.height} events, {thin.height} ({share:.1%}) with no usable "
              f"history for the train number; a line-keyed history covers "
              f"{covered:.1%} of those")

        blocks = {
            # Where the app falls back to the prior today. The comparison the
            # change is for.
            "number_thin": (thin, ("prior", "line", "thin")),
            # Where the app uses the number's history. Line-keying must not be
            # allowed in here, and this says what it would cost if it were.
            "number_ok": (rich, ("number", "line", "prior")),
            # Both together: what a user would actually see.
            "all": (rows, ("shipped", "proposed")),
        }
        for name, (subset, models) in blocks.items():
            print(f"\n  {name}: {subset.height} events")
            print(f"    {'model':<10} {'n':>7} {'CRPS':>7} {'p90':>7} {'cov80':>6}")
            summary = summarise(subset, models)
            for row in summary:
                print(f"    {row['model']:<10} {row['n']:>7} {row['crps']:>7.3f}"
                      f" {row['crps_p90']:>7.3f} {row['coverage80']:>6.3f}")
            result[f"{scenario}|{name}"] = summary

        for label, (subset, left, right) in {
            "prior_vs_line": (thin, "prior", "line"),
            "prior_vs_thin": (thin, "prior", "thin"),
            "thin_vs_line": (thin, "thin", "line"),
            "shipped_vs_proposed": (rows, "shipped", "proposed"),
            # The reason this is a fallback and not a promotion. Paired, over
            # the events both answered — the unpaired table above cannot say
            # it, because the line answers a different set.
            "number_vs_line": (rich, "number", "line"),
        }.items():
            delta = compare(subset, left, right)
            result[f"{scenario}|{label}"] = delta
            if delta["n"]:
                print(f"    {label}: {delta['delta']:+.3f} min "
                      f"(95% {delta['ci95'][0]:+.3f}..{delta['ci95'][1]:+.3f}) "
                      f"over {delta['n']} events, {delta['clusters']} trains")

        by_class = {}
        for bucket in sorted(thin["bucket"].unique().to_list()):
            block = thin.filter(pl.col("bucket") == bucket)
            by_class[bucket] = {
                "summary": summarise(block, ("prior", "line", "thin")),
                "prior_vs_line": compare(block, "prior", "line"),
            }
        result[f"{scenario}|by_class"] = by_class
        print("\n    thin population by class:")
        for bucket, block in by_class.items():
            delta = block["prior_vs_line"]
            if delta["n"]:
                print(f"      {bucket:<14} n={delta['n']:>6} prior "
                      f"{delta['crps_prior']:.3f} line {delta['crps_line']:.3f} "
                      f"delta {delta['delta']:+.3f} "
                      f"(95% {delta['ci95'][0]:+.3f}..{delta['ci95'][1]:+.3f})")

    if out:
        out.write_text(json.dumps(result, indent=1))
        print(f"\nwrote {out}")


def to_day(iso: str) -> int:
    """ISO date -> the same integer the event frame counts days in."""
    return int(pl.select(pl.lit(iso).str.to_date().cast(pl.Int32)).item())


def to_iso(day: int) -> str:
    return str(pl.select(pl.lit(day, dtype=pl.Int32).cast(pl.Date)).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--stations", required=True)
    ap.add_argument("--eval-weeks", type=int, default=12)
    ap.add_argument("--eval-from", help="ISO date; overrides --eval-weeks")
    ap.add_argument("--eval-to", help="ISO date; defaults to the last day of data")
    ap.add_argument("--line-tod-window", type=int, default=TOD_WINDOW_MIN,
                    help="minutes a line\'s run may sit from the query\'s own slot")
    ap.add_argument("--line-days", type=int, default=LINE_DAYS,
                    help="days of line history a shard carries; 0 for all of it")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    df = load_connections(args.data_dir, args.stations.split(","))
    eval_to = to_day(args.eval_to) if args.eval_to else int(df["date"].cast(pl.Int32).max())
    eval_from = to_day(args.eval_from) if args.eval_from else eval_to - args.eval_weeks * 7

    print(f"{df.height} events, evaluating {to_iso(eval_from)} .. {to_iso(eval_to)}")
    scored = sweep(df, eval_from, eval_to, args.line_tod_window,
                   args.line_days or None)
    print(f"{scored.height} scored rows")
    report(with_prior(scored), args.out)


if __name__ == "__main__":
    main()
