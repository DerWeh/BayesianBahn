"""Walk-forward backtest of arrival-delay distribution models.

Mirrors the app's EmpiricalDelay model in Python with every knob exposed,
predicts each historical event using only strictly earlier runs of the same
connection, and scores the *distributions* with proper scoring rules:

  - CRPS (continuous ranked probability score, minutes),
  - pinball loss at q10/q50/q90,
  - empirical coverage of the nominal 80% interval,
  - MAE of the median.

Two scenarios per model: "blind" (no live information, e.g. planning the day
before) and "live" (the delay the train had at its previous stop is known —
the situation shortly before arrival).

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

LONG_DISTANCE = {"ICE", "IC", "EC", "ECE", "RJ", "RJX", "NJ", "EN", "FLX", "TGV", "D", "IR"}

MIN_DELAY, MAX_DELAY = -30, 360


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class Variant:
    """One parameterization of the empirical model."""

    name: str
    half_life_days: float = math.inf  # recency decay
    weekday_boost: float = 1.0  # weight multiplier for same day-of-week
    holiday_as_sunday: bool = False  # fold public holidays into the Sunday class
    window_days: int | None = None  # hard cutoff, Bahn-Vorhersage style
    live_bandwidth: float | None = None  # kernel bandwidth factor; None = ignore live
    min_effective_n: float = 8.0

    def day_class(self, d: date) -> int:
        if self.holiday_as_sunday and d in GERMAN_HOLIDAYS:
            return 6
        return d.weekday()


def weights(
    variant: Variant,
    hist_dates: np.ndarray,  # days-ago (int)
    hist_dayclass: np.ndarray,
    hist_prev: np.ndarray,  # delay at previous stop, NaN if unknown
    query_dayclass: int,
    live_prev: float | None,
) -> tuple[np.ndarray, float]:
    """Returns (weights, kernel_mass). kernel_mass counts comparable runs."""
    w = np.ones(len(hist_dates))
    if variant.window_days is not None:
        w *= hist_dates <= variant.window_days
    if math.isfinite(variant.half_life_days):
        w *= np.exp(-math.log(2.0) / variant.half_life_days * hist_dates)
    w = np.where(hist_dayclass == query_dayclass, w * variant.weekday_boost, w)

    kernel_mass = math.inf
    if live_prev is not None and variant.live_bandwidth is not None:
        bw = max(3.0, variant.live_bandwidth * abs(live_prev))
        kernel = np.where(
            np.isnan(hist_prev),
            0.05,
            np.exp(-0.5 * ((hist_prev - live_prev) / bw) ** 2),
        )
        kernel_mass = float(np.nansum(kernel))
        cond = w * kernel
        # Same guard as the app: only trust conditioning when enough
        # genuinely comparable runs exist.
        sum_w, sum_w2 = cond.sum(), (cond**2).sum()
        eff_n = sum_w * sum_w / sum_w2 if sum_w2 > 0 else 0.0
        if eff_n >= variant.min_effective_n and kernel_mass >= variant.min_effective_n:
            return cond, kernel_mass
    return w, kernel_mass


# --------------------------------------------------------------------------- scoring


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
        return {
            "n": len(self.crps),
            "crps": round(float(np.mean(self.crps)), 3),
            "pinball10": round(float(np.mean(self.pinball10)), 3),
            "pinball50_mae": round(2 * float(np.mean(self.pinball50)), 3),
            "pinball90": round(float(np.mean(self.pinball90)), 3),
            "coverage80": round(float(np.mean(self.covered80)), 3),
        }


# --------------------------------------------------------------------------- data


def load_connections(data_dir: Path, station_evas: list[str]) -> pl.DataFrame:
    files = sorted(data_dir.glob("data-*.parquet"))
    if not files:
        raise SystemExit(f"no data-*.parquet files in {data_dir}")
    print(f"scanning {len(files)} monthly files")
    minutes = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_minutes()  # noqa: E731

    lf = pl.concat(
        [
            pl.scan_parquet(f).select(
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
            )
            for f in files
        ]
    ).with_columns(eva=pl.col("eva").str.strip_chars_start("0"))

    df = (
        lf.with_columns(
            arr_delay=minutes("arrival_change_time", "arrival_planned_time"),
            dep_delay=minutes("departure_change_time", "departure_planned_time"),
            planned=pl.coalesce("arrival_planned_time", "departure_planned_time"),
        )
        .filter(pl.col("planned").is_not_null())
        .sort("train_line_ride_id", "train_line_station_num")
        .with_columns(
            prev_delay=pl.coalesce("arr_delay", "dep_delay")
            .shift(1)
            .over("train_line_ride_id")
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


# --------------------------------------------------------------------------- backtest


def run(data_dir: Path, station_evas: list[str], eval_weeks: int, out: Path | None) -> None:
    df = load_connections(data_dir, station_evas)
    max_date = df["date"].max()
    eval_start = max_date - timedelta(weeks=eval_weeks)
    print(f"{df.height} events, evaluating {eval_start} .. {max_date}")

    variants = [
        Variant("uniform"),
        Variant("hl60", half_life_days=60),
        Variant("hl30", half_life_days=30),
        Variant("hl14", half_life_days=14),
        Variant("window35", window_days=35),
        Variant("hl60_wd2", half_life_days=60, weekday_boost=2),
        Variant("hl30_wd2", half_life_days=30, weekday_boost=2),
        Variant("hl30_wd4", half_life_days=30, weekday_boost=4),
        Variant("hl30_wd2_hol", half_life_days=30, weekday_boost=2, holiday_as_sunday=True),
        # live variants (identical blind behaviour to their base):
        Variant("hl30_wd2_live", half_life_days=30, weekday_boost=2, live_bandwidth=0.3),
        Variant(
            "hl30_wd2_hol_live",
            half_life_days=30,
            weekday_boost=2,
            holiday_as_sunday=True,
            live_bandwidth=0.3,
        ),
        Variant("hl30_wd2_live_bw15", half_life_days=30, weekday_boost=2, live_bandwidth=0.15),
    ]

    results: dict[tuple[str, str, str], Scores] = {}

    def bucket(train_type: str) -> str:
        if train_type in LONG_DISTANCE:
            return "long_distance"
        return "sbahn" if train_type == "S" else "regional"

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
        if n < 15:
            continue
        n_conn += 1
        dayclass_cache: dict[Variant, np.ndarray] = {}
        days = np.array([d.toordinal() for d in dates])

        eval_mask = days >= eval_start.toordinal()
        for i in np.nonzero(eval_mask)[0]:
            hist = (days < days[i]) & (np.abs(tod - tod[i]) <= 20)
            if hist.sum() < 10:
                continue
            hx = delays[hist]
            hprev = prevs[hist]
            hage = days[i] - days[hist]
            y = float(delays[i])
            live = None if np.isnan(prevs[i]) else float(prevs[i])
            b = bucket(str(ttype))

            for variant in variants:
                if variant not in dayclass_cache:
                    dayclass_cache[variant] = np.array(
                        [variant.day_class(d) for d in dates]
                    )
                hdc = dayclass_cache[variant][hist]
                qdc = dayclass_cache[variant][i]
                for scenario, lp in (("blind", None), ("live", live)):
                    if scenario == "live" and lp is None:
                        continue
                    w, _ = weights(variant, hage, hdc, hprev, qdc, lp)
                    if w.sum() <= 0:
                        continue
                    key = (variant.name, scenario, b)
                    results.setdefault(key, Scores()).add(hx, w, y)
                    results.setdefault((variant.name, scenario, "all"), Scores()).add(
                        hx, w, y
                    )

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
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run(args.data_dir, args.stations.split(","), args.eval_weeks, args.out)


if __name__ == "__main__":
    main()
