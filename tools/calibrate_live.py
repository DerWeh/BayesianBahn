"""What DB's report is worth once its own error is admitted.

When DB reports a delay the app shifts each historical run's last-hop
progression onto that number and treats the number as exact. It is not. The
model issues a median 80% interval two minutes wide for a quantity whose error
spans eleven to twelve, and 46.9% of those arrivals land above their own 90th
percentile — the worst calibration anywhere in the model, on the 8% of
predictions made closest to departure, when a passenger is deciding whether to
run.

This sizes the fix before anyone builds it. `pipeline/backtest.py` cannot: the
archive records what trains did, not what DB said they would do, so the
report-to-final residual exists only in the collector's own journal. The days
are split five to fit and the rest to hold out, and the held-out set spans the
end of the rail-replacement blockade, so the two windows differ in composition
and not only in date.

The comparison model is deliberately crude — DB's report plus the empirical
residual for its lead-time bin, with the train's own history discarded. It is
not a proposal. It is a floor: whatever a real model does with both sources
should beat this, and this already beats what ships.

Usage:
    pixi run -e pipeline python tools/calibrate_live.py
"""
import json, glob, collections, math, statistics
import numpy as np

FIT  = ["2026-08-18","2026-08-19","2026-08-20","2026-08-21","2026-08-22"]
TEST = ["2026-08-23","2026-08-24","2026-08-25","2026-08-29","2026-08-30",
        "2026-08-31","2026-09-01"]

def load(days):
    out = []
    for d in days:
        for l in open(f"tools/.scored/{d}/arrivals-live.jsonl"):
            r = json.loads(l)
            if r["source"] != "EMPIRICAL_LIVE": continue
            out.append((r["lead"], r["db"], r["truth"], r["crps"],
                        r["q10"], r["q90"]))
    return out

def crps_sample(sample, y):
    """CRPS of an equally weighted empirical sample (sorted)."""
    n = len(sample)
    term1 = np.abs(sample - y).mean()
    cw = (np.arange(1, n + 1)) / n
    exx = 2.0 * np.sum(sample * (cw - 1.0 / n - (1.0 - cw))) / n
    return term1 - 0.5 * exx

# Lead-time bins, minutes. Finer than the four used for reporting, because the
# whole point is that the width is a function of lead and we want to see the
# function rather than four steps of it.
EDGES = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 10**9]
def binof(lead):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= lead < EDGES[i + 1]: return i
    return len(EDGES) - 2

fit, test = load(FIT), load(TEST)
res = collections.defaultdict(list)
for lead, db, truth, *_ in fit:
    res[binof(lead)].append(truth - db)

print(f"fit {len(fit)} events, held out {len(test)}")
print("\nspread of (truth - report) by lead, and after enforcing monotonicity")
print(f"{'lead bin':<12}{'n':>7}{'p90-p10':>9}{'isotonic':>10}")
raw = []
for i in range(len(EDGES) - 1):
    v = res.get(i) or []
    if len(v) < 30:
        raw.append(None); continue
    q = np.quantile(v, [0.1, 0.9])
    raw.append(q[1] - q[0])

# Pool-adjacent-violators: the spread cannot fall as the horizon grows.
def isotonic(y, w):
    y = list(y); w = list(w); i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1]:
            tot = w[i] + w[i + 1]
            y[i] = y[i + 1] = (y[i] * w[i] + y[i + 1] * w[i + 1]) / tot
            w[i] = w[i + 1] = tot / 2
            i = max(i - 1, 0)
        else:
            i += 1
    return y

idx = [i for i in range(len(EDGES) - 1) if raw[i] is not None]
iso = isotonic([raw[i] for i in idx], [len(res[i]) for i in idx])
isomap = dict(zip(idx, iso))
def label(i):
    hi = "inf" if EDGES[i + 1] >= 10**9 else str(EDGES[i + 1])
    return f"{EDGES[i]}-{hi}"

for i in idx:
    print(f"{label(i):<12}{len(res[i]):>7}{raw[i]:>9.0f}{isomap[i]:>10.1f}")

# --- does calibrating help anything besides calibration? --------------------

# Residual sample per bin, from the fit window only; nearest populated bin when
# a bin is empty (the collector polls at fixed taus, so leads cluster).
pool = {i: np.sort(np.array(res[i], dtype=float)) for i in idx}
def residuals(i):
    if i in pool: return pool[i]
    return pool[min(idx, key=lambda j: abs(j - i))]

def evaluate(rows, label):
    shipped, db_only, calibrated = [], [], []
    cov_ship, cov_cal, pit_cal = [], [], []
    for lead, db, truth, crps, q10, q90 in rows:
        shipped.append(crps)
        db_only.append(abs(db - truth))
        s = np.sort(db + residuals(binof(lead)))
        calibrated.append(crps_sample(s, truth))
        cov_ship.append(bool(q10 <= truth <= q90))
        lo, hi = np.quantile(s, [0.1, 0.9])
        cov_cal.append(bool(lo <= truth <= hi))
        pit_cal.append(float(np.searchsorted(s, truth) / len(s)))
    n = len(rows)
    print(f"\n{label}  (n={n})")
    print(f"   {'model':<26}{'CRPS':>8}{'cov80':>8}")
    print(f"   {'DB point forecast':<26}{statistics.mean(db_only):>8.2f}{'-':>8}")
    print(f"   {'as shipped':<26}{statistics.mean(shipped):>8.2f}"
          f"{statistics.mean(cov_ship):>8.1%}")
    print(f"   {'report + residual':<26}{statistics.mean(calibrated):>8.2f}"
          f"{statistics.mean(cov_cal):>8.1%}")
    hist = [0] * 10
    for u in pit_cal: hist[min(int(u * 10), 9)] += 1
    print("   calibrated PIT deciles:  " + " ".join(f"{h/n*100:.1f}" for h in hist))

evaluate(fit, "fit window (in-sample)")
evaluate(test, "held out — includes the end of the bus blockade")

# --- where does the report stop being worth more than the history? ----------

blind = {}
for d in FIT + TEST:
    for l in open(f"tools/.scored/{d}/arrivals-blind.jsonl"):
        r = json.loads(l)
        blind[(d, r["eva"], r["num"], r["planned"], r["tau"])] = r["crps"]

rows = collections.defaultdict(lambda: collections.defaultdict(list))
for d in TEST:
    for l in open(f"tools/.scored/{d}/arrivals-live.jsonl"):
        r = json.loads(l)
        if r["source"] != "EMPIRICAL_LIVE": continue
        key = (d, r["eva"], r["num"], r["planned"], r["tau"])
        if key not in blind: continue
        i = binof(r["lead"])
        s = np.sort(r["db"] + residuals(i))
        cell = rows[i]
        cell["shipped"].append(r["crps"])
        cell["history"].append(blind[key])
        cell["calibrated"].append(crps_sample(s, r["truth"]))
        cell["db"].append(abs(r["db"] - r["truth"]))

print("\nheld out, by lead: CRPS of each answer (lower is better)")
print(f"{'lead':<12}{'n':>7}{'DB':>8}{'shipped':>9}{'calibrated':>12}{'history only':>14}")
for i in sorted(rows):
    c = rows[i]
    if len(c["shipped"]) < 100: continue
    print(f"{label(i):<12}{len(c['shipped']):>7}{statistics.mean(c['db']):>8.2f}"
          f"{statistics.mean(c['shipped']):>9.2f}{statistics.mean(c['calibrated']):>12.2f}"
          f"{statistics.mean(c['history']):>14.2f}")
