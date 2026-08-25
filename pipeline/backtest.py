"""Walk-forward backtest of arrival-delay distribution models.

Mirrors the app's EmpiricalDelay model in Python with every knob exposed,
predicts each historical event using only strictly earlier runs of the same
connection, and scores the *distributions* with proper scoring rules:

  - CRPS (continuous ranked probability score, minutes),
  - pinball loss at q10/q50/q90,
  - empirical coverage of the nominal 80% interval,
  - MAE of the median.

Three scenarios per model:

  - "blind"  no live information at all — planning the day before,
  - "live"   the delay the train had at its previous stop is known,
  - "gated"  the app's shipped rule: believe that delay only when it is a
             delay, and fall back to the history when it is not.

The distinction between "live" and "gated" is the point of the third scenario.
What the app receives is not a measurement but *DB's forecast for this station*,
and DB states a stop in four shapes of which three mean "on time" — the plan
restated rather than an observation. The archive cannot show that, because it
records what the trains did and not what DB said they would do; only the
collector under `tools/collect_forecasts.py` sees DB's forecasts, which is the
one thing it is for. So the two datasets answer different halves:

  - this backtest, on months of archive, decides how history should be
    weighted and whether a *genuine* live signal is worth conditioning on;
  - the collected forecasts decide whether DB's number is a genuine signal.

Running the gate here against a real previous-stop delay is therefore the
control: it should come out *worse* than "live", because a measured zero is
information and DB's zero is not. If it ever comes out better, the gate is
doing something other than what it was justified by.

Usage:
    pixi run -e pipeline python pipeline/backtest.py \
        --data-dir DATA_DIR --stations 8000013,8000261 \
        [--eval-weeks 8] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import polars as pl

GERMAN_HOLIDAYS = holidays_lib.country_holidays("DE") | holidays_lib.country_holidays(
    "DE", subdiv="BY"
)

# Mirrors TrainClass.LONG_DISTANCE_CATEGORIES in DelayModel.kt. This is the one
# boundary the backtest shares with the app: results are reported per class, so
# a category on the wrong side is compared against the wrong prior and tunes the
# app's parameters against a classification the app does not use. It drifted
# once already — "WB" was added in the app and not here — so a test now pins it.
LONG_DISTANCE = {
    "ICE", "IC", "EC", "ECE", "RJ", "RJX", "NJ", "EN", "FLX", "TGV", "D", "IR", "WB",
}

MIN_DELAY, MAX_DELAY = -30, 360


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Variant:
    """One parameterization of the empirical model."""

    name: str
    half_life_days: float = math.inf  # recency decay
    weekday_boost: float = 1.0  # weight multiplier for same day-of-week
    holiday_as_sunday: bool = False  # fold public holidays into the Sunday class
    # Boost for a run on the same *kind* of day rather than the same named one.
    # The two are not the same bet. Same-weekday says Tuesdays resemble
    # Tuesdays and not Wednesdays; same-daytype says working days resemble each
    # other and differ from Saturdays and Sundays. With a 30-day half-life the
    # first has about four runs to work with and the second about twenty, so if
    # the real structure is working-day-versus-weekend the narrower boost is
    # spending its weight on a distinction that is not there. The forward
    # evaluation makes this worth asking: DB's own mean error is 3.14 minutes
    # Monday to Friday against 1.84 at the weekend.
    daytype_boost: float = 1.0
    # The share of the total weight the same-kind runs are to hold, whatever
    # their number. A multiplicative boost cannot express this: on a weekend
    # query the matching runs are two sevenths of the history and the boost
    # moves the mix a long way, while on a working day they are five sevenths
    # and it barely moves it — so one constant does two different things, and
    # the backtest shows it helping weekends and hurting working days by about
    # the same amount. A share does the same thing to both.
    daytype_share: float | None = None
    # How far from the query's own planned time a run may be and still count as
    # the same connection. The app's window is 20 minutes and it binds for only
    # 8% of events — inside a train number the planned time hardly moves — but
    # for a line-numbered S-Bahn whose runs span the day it decides everything.
    tod_window: int = 20
    window_days: int | None = None  # hard cutoff, Bahn-Vorhersage style
    live_bandwidth: float | None = None  # kernel bandwidth factor; None = ignore live
    # "kernel": reweight runs with similar previous-stop delay (app draft).
    # "delta": predict live + (final - prev) residuals of all runs —
    #          Bahn-Vorhersage's delay_diff idea, nonparametrically.
    live_mode: str = "kernel"
    min_effective_n: float = 8.0
    def day_class(self, d: date) -> int:
        if self.holiday_as_sunday and d in GERMAN_HOLIDAYS:
            return 6
        return d.weekday()

    def daytype_class(self, d: date) -> int:
        """Working day, Saturday or Sunday — the coarser grouping."""
        weekday = self.day_class(d)
        return weekday if weekday >= 5 else 0


def base_weights(
    variant: Variant,
    hist_dates: np.ndarray,  # days-ago (int)
    hist_dayclass: np.ndarray,
    query_dayclass: int,
    hist_daytype: np.ndarray | None = None,
    query_daytype: int | None = None,
) -> np.ndarray:
    w = np.ones(len(hist_dates))
    if variant.window_days is not None:
        w *= hist_dates <= variant.window_days
    if math.isfinite(variant.half_life_days):
        w *= np.exp(-math.log(2.0) / variant.half_life_days * hist_dates)
    if hist_daytype is not None and variant.daytype_share is not None:
        same = hist_daytype == query_daytype
        same_mass, other_mass = w[same].sum(), w[~same].sum()
        # Only when there is something on both sides: with no weekend runs at
        # all, renormalising would put every weekend query on nothing.
        if same_mass > 0 and other_mass > 0:
            share = variant.daytype_share
            w = np.where(same, w * (share / same_mass),
                         w * ((1.0 - share) / other_mass))
    elif hist_daytype is not None and variant.daytype_boost != 1.0:
        w = np.where(hist_daytype == query_daytype, w * variant.daytype_boost, w)
    # Applied after, so a run on the same named day gets both: the coarse boost
    # for being a working day and the fine one for being a Tuesday.
    return np.where(hist_dayclass == query_dayclass, w * variant.weekday_boost, w)


def predictive_points(
    variant: Variant,
    hist_delay: np.ndarray,
    hist_prev: np.ndarray,  # delay at previous stop, NaN if unknown
    w: np.ndarray,
    live_prev: float | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Returns (support, weights, whether the live report was conditioned on)."""
    if live_prev is None or variant.live_bandwidth is None:
        return hist_delay, w, False

    if variant.live_mode == "delta":
        # Shift each run's progression residual onto the live report:
        # candidate final = live + (final_i - prev_i). Uses every run with a
        # known previous-stop delay, optionally kernel-sharpened.
        known = ~np.isnan(hist_prev)
        if known.sum() < variant.min_effective_n:
            return hist_delay, w, False
        x = live_prev + (hist_delay[known] - hist_prev[known])
        wk = w[known]
        if variant.live_bandwidth > 0:
            bw = max(3.0, variant.live_bandwidth * abs(live_prev))
            wk = wk * (
                0.15
                + np.exp(-0.5 * ((hist_prev[known] - live_prev) / bw) ** 2)
            )
        return x, wk, True

    # kernel mode (the app draft)
    bw = max(3.0, variant.live_bandwidth * abs(live_prev))
    kernel = np.where(
        np.isnan(hist_prev),
        0.05,
        np.exp(-0.5 * ((hist_prev - live_prev) / bw) ** 2),
    )
    kernel_mass = float(np.nansum(kernel))
    cond = w * kernel
    # Same guard as the app: only trust conditioning when enough genuinely
    # comparable runs exist.
    sum_w, sum_w2 = cond.sum(), (cond**2).sum()
    eff_n = sum_w * sum_w / sum_w2 if sum_w2 > 0 else 0.0
    if eff_n >= variant.min_effective_n and kernel_mass >= variant.min_effective_n:
        return hist_delay, cond, True
    return hist_delay, w, False


# --------------------------------------------------------------------------- scoring


# Mirrors Predictor.MIN_INFORMATIVE_DELAY_MINUTES in the app. Guarded by
# tools/tests/test_route_bench.py, which reads both and compares them: this file
# has drifted from the app before.
MIN_INFORMATIVE_DELAY = 1.0


def scenarios_for(live_prev: float | None) -> list[tuple[str, float | None]]:
    """The (scenario, live input) pairs to score for one event.

    "live" and "gated" are only defined where a previous-stop delay exists at
    all. Scoring "gated" on the events "live" never saw would compare two
    different event sets: those events carry no live signal to gate, so they
    would enter "gated" as pure blind predictions and drag it towards blind
    while "live" was measured without them.
    """
    if live_prev is None:
        return [("blind", None)]
    return [
        # "blind" stays over every event: that is the honest figure for a model
        # with no live data, and it is what the variant comparison uses.
        ("blind", None),
        # ...but comparing it with the two below would compare event sets, so
        # the same prediction is also scored over just the events that have a
        # live signal. Those three rows are the ones to read against each other.
        ("blind_where_live", None),
        ("live", live_prev),
        ("gated", gated_live(live_prev)),
    ]


def gated_live(live_prev: float | None) -> float | None:
    """The app's rule: a report counts only when it reports a delay.

    Applied here to a *measured* previous-stop delay, which is why this is the
    control rather than the treatment — see the module docstring.
    """
    if live_prev is None or live_prev < MIN_INFORMATIVE_DELAY:
        return None
    return live_prev


def crps_empirical(x: np.ndarray, w: np.ndarray, y: float) -> float:
    """Exact CRPS of a weighted empirical distribution vs observation y."""
    w = w / w.sum()
    term1 = np.sum(w * np.abs(x - y))
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    # E|X-X'| = 2 * sum_i w_i x_i (F_i - w_i/2) - ... use pairwise formula:
    # E|X-X'| = 2 * sum_{i<j} w_i w_j (x_j - x_i) = 2 * sum_j w_j x_j C_{j-1} - 2 * sum_j w_j x_j (1 - C_j) ... simpler:
    prev_cw = cw - ws
    exx = 2.0 * np.sum(ws * xs * (prev_cw - (1.0 - cw)))
    return float(term1 - 0.5 * exx)


def quantiles(x: np.ndarray, w: np.ndarray, qs: list[float]) -> list[float]:
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws) / ws.sum()
    return [float(xs[np.searchsorted(cw, q)]) if q <= cw[-1] else float(xs[-1]) for q in qs]


def pinball(q_pred: float, y: float, q: float) -> float:
    return (1 - q) * (q_pred - y) if y < q_pred else q * (y - q_pred)


@dataclass
class Scores:
    """Per-event scores, kept whole rather than accumulated into means.

    A mean CRPS hides exactly the events that matter: nobody minds a forecast
    that is a minute out, and everybody minds the one that is twenty out and
    unflagged. The summary therefore carries the upper tail as well.
    """

    crps: list[float] = field(default_factory=list)
    pinball10: list[float] = field(default_factory=list)
    pinball50: list[float] = field(default_factory=list)
    pinball90: list[float] = field(default_factory=list)
    covered80: list[bool] = field(default_factory=list)

    def add(self, x: np.ndarray, w: np.ndarray, y: float) -> None:
        q10, q50, q90 = quantiles(x, w, [0.1, 0.5, 0.9])
        self.crps.append(crps_empirical(x, w, y))
        self.pinball10.append(pinball(q10, y, 0.1))
        self.pinball50.append(pinball(q50, y, 0.5))
        self.pinball90.append(pinball(q90, y, 0.9))
        self.covered80.append(q10 <= y <= q90)

    def summary(self) -> dict:
        crps = np.asarray(self.crps)
        return {
            "n": len(self.crps),
            "crps": round(float(np.mean(crps)), 3),
            "crps_p50": round(float(np.quantile(crps, 0.50)), 3),
            "crps_p90": round(float(np.quantile(crps, 0.90)), 3),
            "crps_p99": round(float(np.quantile(crps, 0.99)), 3),
            "crps_max": round(float(crps.max()), 3),
            "pinball10": round(float(np.mean(self.pinball10)), 3),
            "pinball50_mae": round(2 * float(np.mean(self.pinball50)), 3),
            "pinball90": round(float(np.mean(self.pinball90)), 3),
            "coverage80": round(float(np.mean(self.covered80)), 3),
        }


# --------------------------------------------------------------------------- data


def load_month(file: Path, station_evas: list[str]) -> pl.DataFrame:
    """One monthly file → evaluation rows at the requested stations.

    Months are processed independently to bound memory (ride ids never span
    months), and the data is semi-joined down to trains calling at the
    stations *before* the sort that the prev-stop feature needs.
    """
    minutes = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_minutes()  # noqa: E731

    lf = pl.scan_parquet(file).select(
        "eva",
        "train_type",
        "train_number",
        "train_line_ride_id",
        "train_line_station_num",
        "arrival_planned_time",
        "arrival_change_time",
        "departure_planned_time",
        "departure_change_time",
        "is_canceled",
    ).with_columns(
        eva=pl.col("eva").str.strip_chars_start("0"),
        identity=pl.col("train_type") + " " + pl.col("train_number"),
    )
    wanted = (
        lf.filter(pl.col("eva").is_in(station_evas)).select("identity").unique()
    )
    lf = lf.join(wanted, on="identity", how="semi")

    df = (
        lf.with_columns(
            arr_delay=minutes("arrival_change_time", "arrival_planned_time"),
            dep_delay=minutes("departure_change_time", "departure_planned_time"),
            planned=pl.coalesce("arrival_planned_time", "departure_planned_time"),
        )
        .filter(pl.col("planned").is_not_null())
        # ride_id is the route pattern shared by all days of a month, so
        # partition by service day too (4h shift keeps post-midnight stops
        # with their run) — otherwise prev_delay crosses days.
        .with_columns(
            service_day=(pl.col("planned") - pl.duration(hours=4)).dt.date()
        )
        .sort("train_line_ride_id", "service_day", "train_line_station_num")
        .with_columns(
            prev_delay=pl.coalesce("arr_delay", "dep_delay")
            .shift(1)
            .over("train_line_ride_id", "service_day")
        )
        .filter(pl.col("eva").is_in(station_evas))
        .with_columns(
            date=pl.col("planned").dt.date(),
            tod_min=pl.col("planned").dt.hour() * 60 + pl.col("planned").dt.minute(),
            delay=pl.coalesce("arr_delay", "dep_delay")
            .clip(MIN_DELAY, MAX_DELAY)
            .cast(pl.Float64),
            prev=pl.col("prev_delay").clip(MIN_DELAY, MAX_DELAY).cast(pl.Float64),
        )
        .filter(~pl.col("is_canceled") & pl.col("delay").is_not_null())
        .select(
            "eva", "train_type", "train_number", "date", "tod_min", "delay", "prev"
        )
        .collect(engine="streaming")
    )
    return df


def load_connections(data_dir: Path, station_evas: list[str]) -> pl.DataFrame:
    files = sorted(data_dir.glob("data-*.parquet"))
    if not files:
        raise SystemExit(f"no data-*.parquet files in {data_dir}")
    frames = []
    for f in files:
        frames.append(load_month(f, station_evas))
        print(f"  {f.name}: {frames[-1].height} rows", flush=True)
    return pl.concat(frames)


# --------------------------------------------------------------------------- backtest


def bucket(train_type: str) -> str:
    if train_type in LONG_DISTANCE:
        return "long_distance"
    return "sbahn" if train_type == "S" else "regional"


def run(data_dir: Path, station_evas: list[str], eval_weeks: int,
        out: Path | None, only: list[str] | None = None,
        min_history: int = 10, min_connection_runs: int = 15) -> None:
    df = load_connections(data_dir, station_evas)
    max_date = df["date"].max()
    eval_start = max_date - timedelta(weeks=eval_weeks)
    print(f"{df.height} events, evaluating {eval_start} .. {max_date}")

    variants = [
        Variant("uniform"),
        Variant("hl60", half_life_days=60),
        Variant("hl30", half_life_days=30),
        Variant("hl14", half_life_days=14),
        Variant("hl7", half_life_days=7),
        Variant("window35", window_days=35),
        Variant("hl14_wd2", half_life_days=14, weekday_boost=2),
        Variant("hl14_wd2_hol", half_life_days=14, weekday_boost=2, holiday_as_sunday=True),
        Variant("hl60_wd2", half_life_days=60, weekday_boost=2),
        Variant("hl30_wd2", half_life_days=30, weekday_boost=2),
        Variant("hl30_wd4", half_life_days=30, weekday_boost=4),
        Variant("hl30_wd2_hol", half_life_days=30, weekday_boost=2, holiday_as_sunday=True),
        # live variants (identical blind behaviour to their base):
        Variant("hl30_wd2_kernel", half_life_days=30, weekday_boost=2, live_bandwidth=0.3),
        Variant(
            "hl30_wd2_delta",
            half_life_days=30,
            weekday_boost=2,
            live_bandwidth=0.3,
            live_mode="delta",
        ),
        Variant(
            "hl30_wd2_delta_nok",
            half_life_days=30,
            weekday_boost=2,
            live_bandwidth=0.0,
            live_mode="delta",
        ),
        Variant(
            "hl14_delta",
            half_life_days=14,
            live_bandwidth=0.3,
            live_mode="delta",
        ),
        # What "a comparable run" should mean on the day axis. The shipped
        # model boosts the same named weekday; these ask whether the coarser
        # working-day/Saturday/Sunday grouping is the better bet, and whether
        # the two are worth having together.
        *[
            Variant(
                f"day_wd{wd:g}_dt{dt_:g}" + ("_hol" if hol else ""),
                half_life_days=30, weekday_boost=wd, daytype_boost=dt_,
                holiday_as_sunday=hol,
                live_bandwidth=0.3, live_mode="delta",
            )
            for wd, dt_, hol in (
                (1.0, 1.0, False),   # neither: recency alone
                (2.0, 1.0, False),   # the shipped model
                (4.0, 1.0, False),
                (1.0, 2.0, False),   # day type alone
                (1.0, 4.0, False),
                (1.0, 8.0, False),
                (2.0, 2.0, False),   # both
                (1.0, 4.0, True),    # day type, holidays folded into Sunday
                (2.0, 2.0, True),
            )
        ],
        *[
            Variant(
                f"share{sh:g}",
                half_life_days=30, weekday_boost=1.0, daytype_share=sh,
                live_bandwidth=0.3, live_mode="delta",
            )
            for sh in (0.7, 0.8, 0.9, 0.95, 1.0)
        ],
        *[
            Variant(
                f"tod{win}",
                half_life_days=30, weekday_boost=2.0, tod_window=win,
                live_bandwidth=0.3, live_mode="delta",
            )
            for win in (5, 10, 20, 40, 90, 180)
        ],
    ]

    if only:
        wanted = set(only)
        missing = wanted - {v.name for v in variants}
        if missing:
            raise SystemExit(f"no such variant: {', '.join(sorted(missing))}")
        variants = [v for v in variants if v.name in wanted]

    results: dict[tuple[str, str, str], Scores] = {}

    # A pooled per-class prior and a median-sharpening knob lived here.
    # Both were measured on this data and rejected; the commit that
    # removed them carries the numbers.
    widest_window = max(v.tod_window for v in variants)
    n_conn = 0
    for (eva, ttype, tnum), grp in df.group_by(
        ["eva", "train_type", "train_number"], maintain_order=False
    ):
        grp = grp.sort("date")
        dates = grp["date"].to_list()  # python datetime.date, for day_class
        tod = grp["tod_min"].to_numpy()
        delays = grp["delay"].to_numpy()
        prevs = grp["prev"].to_numpy()
        n = len(dates)
        # Connections with less history than this are skipped outright, which
        # is why lowering --min-history alone barely changes the event count:
        # the thin-history regime the app's prior fallback exists for is
        # excluded here, upstream of it. Answering whether that fallback beats
        # shrinking a thin history needs this floor lowered too, and then the
        # walk-forward has very little to walk.
        if n < min_connection_runs:
            continue
        n_conn += 1
        dayclass_cache: dict[Variant, np.ndarray] = {}
        daytype_cache: dict[Variant, np.ndarray] = {}
        days = np.array([d.toordinal() for d in dates])

        eval_mask = days >= eval_start.toordinal()
        for i in np.nonzero(eval_mask)[0]:
            # Circular, as EmpiricalDelay.timeOfDayDistance is: 23:50 and
            # 00:10 are twenty minutes apart, not twenty-three hours. The
            # backtest compared them the flat way, which is a divergence from
            # the app it is meant to mirror.
            # int32 on purpose: the column arrives narrow enough that
            # `24 * 60 - gap` overflows it.
            gap = np.abs(tod.astype(np.int32) - int(tod[i]))
            dtod = np.minimum(gap, 24 * 60 - gap)
            hist = (days < days[i]) & (dtod <= widest_window)
            earlier = dtod[days < days[i]]
            spread = ("spans_the_day" if earlier.size and earlier.max() > 20
                      else "one_slot")
            # The floor matters more than it looks: with it at ten, every
            # experiment about thin histories is run on events that do not have
            # one. The app falls back to a class-wide prior below eight
            # effective runs, and whether that beats shrinking the thin history
            # towards the same population is only answerable down here.
            if hist.sum() < min_history:
                continue
            hx = delays[hist]
            hprev = prevs[hist]
            hage = days[i] - days[hist]
            hdtod = dtod[hist]
            y = float(delays[i])
            live = None if np.isnan(prevs[i]) else float(prevs[i])
            b = bucket(str(ttype))
            # Also split by the kind of day being predicted. A boost towards
            # same-kind runs changes the weighting far more for a weekend query
            # — where the matching runs are the minority — than for a working
            # day, and an average over both would hide it.
            query_daytype = "on_a_weekend" if dates[i].weekday() >= 5 else "on_a_workday"

            for variant in variants:
                if variant not in dayclass_cache:
                    dayclass_cache[variant] = np.array(
                        [variant.day_class(d) for d in dates]
                    )
                hdc = dayclass_cache[variant][hist]
                qdc = dayclass_cache[variant][i]
                if variant not in daytype_cache:
                    daytype_cache[variant] = np.array(
                        [variant.daytype_class(d) for d in dates]
                    )
                hdt = daytype_cache[variant][hist]
                w0 = base_weights(variant, hage, hdc, qdc, hdt,
                                  daytype_cache[variant][i])
                if variant.tod_window < widest_window:
                    w0 = w0 * (hdtod <= variant.tod_window)
                    if w0.sum() <= 0:
                        continue
                for scenario, lp in scenarios_for(live):
                    x, w, _conditioned = predictive_points(variant, hx, hprev, w0, lp)
                    if w.sum() <= 0:
                        continue
                    for label in (b, "all", query_daytype, spread):
                        results.setdefault(
                            (variant.name, scenario, label), Scores()
                        ).add(x, w, y)

    print(f"{n_conn} connections evaluated")
    table = {
        f"{name}|{scenario}|{b}": scores.summary()
        for (name, scenario, b), scores in sorted(results.items())
    }
    print(f"{'variant':<22} {'scen':<6} {'bucket':<14} {'n':>7} {'CRPS':>7} {'MAE50':>7} {'cov80':>6}")
    for key, s in table.items():
        name, scenario, b = key.split("|")
        print(
            f"{name:<22} {scenario:<6} {b:<14} {s['n']:>7} {s['crps']:>7.3f}"
            f" {s['pinball50_mae']:>7.3f} {s['coverage80']:>6.3f}"
        )
    if out:
        out.write_text(json.dumps(table, indent=1))
        print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--stations", required=True)
    ap.add_argument("--eval-weeks", type=int, default=8)
    ap.add_argument("--min-history", type=int, default=10,
                    help="same-time-of-day runs required before an event is scored")
    ap.add_argument("--min-connection-runs", type=int, default=15,
                    help="runs a connection needs before any of it is scored")
    ap.add_argument("--only", nargs="*",
                    help="run only these variant names, for a focused experiment")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run(args.data_dir, args.stations.split(","), args.eval_weeks, args.out,
        args.only, args.min_history, args.min_connection_runs)


if __name__ == "__main__":
    main()
