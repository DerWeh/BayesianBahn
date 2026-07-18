# BayesianBahn

Empirical arrival-time predictions for Deutsche Bahn trains — as a
**distribution**, not a point estimate.

Pick a station, pick a train, and see when you will *actually* arrive: the
median predicted arrival time, an 80% credible interval, the full delay
distribution, and the probability of staying within 5 minutes — all computed
from that train's real historical runs, not from DB's own forecast.

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
  at your station. Weights combine recency (60-day half-life), a same-weekday
  boost, and — when the train is already reported late — a kernel that
  reweights history towards runs that were similarly late one stop earlier.
  Trains without history fall back to a Bayesian Normal-inverse-gamma prior
  per train class and time of day (closed-form Student-t predictive).

## Roadmap

- Replace bundled shards with a remote shard host updated monthly from the
  archive (draft bundles trains calling at Augsburg Hbf / München Hbf).
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

### Regenerating data

```sh
# download monthly parquet files into pipeline/data/, then:
pixi run -e pipeline python pipeline/build_shards.py \
    --data-dir pipeline/data --out-dir pipeline/output --stations 8000013,8000261
cp pipeline/output/shards/*.json.gz pipeline/output/index.json app/src/main/assets/history/
```

The bundled station list (`app/src/main/assets/stations.csv`) derives from
[db-stations](https://github.com/derhuerst/db-stations) (DB open data, CC BY 4.0).

## License

MIT — see [LICENSE](LICENSE). Bundled data: CC BY 4.0 by Deutsche Bahn
(station list and delay history).
