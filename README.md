# BayesianBahn

Empirical arrival-time predictions for Deutsche Bahn trains — as a
**distribution**, not a point estimate.

Enter from, to and a departure time (future trips included): the app finds
direct trains and one-transfer connections and predicts when you will
*actually* arrive — median arrival, an 80% credible interval, the full
distribution, and per-train catch probabilities — all computed from real
historical runs, not from DB's own forecast. A Deutschland-Ticket filter
restricts the search to regional trains; live station boards and per-train
forecasts remain available behind the list icon.

## How predictions work

- **History**: the [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data)
  archive (CC BY 4.0) provides months of real per-stop delay data collected
  from DB's IRIS API. `pipeline/build_shards.py` condenses it into one small
  shard per train: every station it calls at, with date, planned time,
  arrival/departure delay, delay at the previous stop, and cancellations.
- **Live state**: the app fetches the keyless IRIS timetable API
  (`iris.noncd.db.de`) for the live board — the same source DB's station
  displays use.
- **Prediction**: a weighted empirical distribution over the train's past runs
  at your station. Without live data, past final delays are weighted by
  recency (30-day half-life — short on purpose: construction sites and
  timetable changes make old runs stale) and a same-weekday boost. When a
  live delay is reported, the *delta* model shifts each run's observed
  last-hop progression (final − previous stop) onto the live report,
  sharpened towards runs that were similarly late. Trains without history
  fall back to a Bayesian Normal-inverse-gamma prior per train class and
  time of day (closed-form Student-t predictive).
- **Connections**: for a journey with a transfer, the app propagates the
  feeder's arrival distribution through the transfer with the law of total
  probability: the passenger boards the first connecting train (in planned
  order) that has not yet left, so the final arrival is a mixture
  `Σ_k P(board k) · P(arrival | board k)` over all candidate trains towards
  the destination — including the case where a *delayed* earlier train is
  still catchable. Departure and arrival delays of a candidate come from the
  same historical run, preserving their correlation; feeder and candidates
  are assumed independent (documented in-app). A **Deutschland-Ticket**
  switch (on by default when the feeder is regional) restricts candidates
  to covered trains (RE, RB, IRE, S-Bahn, private regional operators — no
  ICE/IC/EC, night trains or FlixTrain).
- **Data updates**: predictions stay fresh to within ~a day, with minimal
  downloads. The daily `update-data` workflow maintains three assets on the
  `data` release: `meta.json` (tiny descriptor), `history.zip` (monthly
  base, ~16 MB, rebuilt when the archive publishes a new month) and
  `recent.zip` (~1–3 MB, rebuilt daily by `pipeline/build_recent.py` from
  the archive's *raw* IRIS logs, covering the days newer than the newest
  monthly file). The in-app update checks `meta.json` first and fetches
  only the tier that changed; the app overlays recent runs onto the base.
- **Backtesting**: `pipeline/backtest.py` walk-forward evaluates model
  variants on months of archive data with proper scoring rules (CRPS,
  pinball loss, interval coverage). On a 12-week eval (Easter–June 2026,
  91k predictions) the delta model cut live-scenario CRPS 3.2× versus
  ignoring live data (1.53 vs 4.83); a 30-day half-life beat 7/14/60 days;
  explicit holiday handling showed no benefit even across the April–June
  holidays. Parameters above are the backtest winners.

## Roadmap

- Widen the data beyond the draft subset (trains calling at Augsburg Hbf /
  München Hbf) — the pipeline and the update mechanism already handle any
  station list.
- Condition on the *true* previous-stop live delay instead of the current
  station's report.
- On-device [TabICL v2](https://github.com/soda-inria/tabicl) (BSD-3) via ONNX
  Runtime as the conditional model: context = this connection's historical
  runs, query = today's features, output = full predictive distribution.

## Building

Toolchain is pinned with [pixi](https://pixi.sh); the Android SDK must be
available via `local.properties` or `ANDROID_HOME`.

```sh
pixi run ./gradlew assembleDebug      # build the APK
pixi run ./gradlew testDebugUnitTest  # run unit tests
```

### Backtesting

```sh
pixi run -e pipeline python pipeline/backtest.py \
    --data-dir pipeline/data --stations 8000013,8000261 --eval-weeks 12
```

### Regenerating data

```sh
# download monthly parquet files into pipeline/data/, then:
pixi run -e pipeline python pipeline/build_shards.py \
    --data-dir pipeline/data --out-dir pipeline/output --stations 8000013,8000261
cp pipeline/output/shards/*.jgz pipeline/output/index.json app/src/main/assets/history/
```

The bundled station list (`app/src/main/assets/stations.csv`) derives from
[db-stations](https://github.com/derhuerst/db-stations) (DB open data, CC BY 4.0).

## License

MIT — see [LICENSE](LICENSE). Bundled data: CC BY 4.0 by Deutsche Bahn
(station list and delay history).
