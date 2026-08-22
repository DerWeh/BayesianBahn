# BayesianBahn

Empirical arrival-time predictions for Deutsche Bahn trains — as a
**distribution**, not a point estimate.

[![Get it on F-Droid](https://fdroid.gitlab.io/artwork/badge/get-it-on.png)](https://f-droid.org/packages/io.github.derweh.bayesianbahn/)

> **Early release.** Predictions are experimental — always cross-check
> times and connections with DB's official apps before relying on them.

Enter from, to and a departure time (future trips included): the app finds
direct trains and one-transfer connections and predicts when you will
*actually* arrive — median arrival, an 80% credible interval, the full
distribution, and per-train catch probabilities — all computed from real
historical runs, not from DB's own forecast. A Deutschland-Ticket filter
restricts the search to regional trains; live station boards and per-train
forecasts remain available behind the list icon.

## Install

From [F-Droid](https://f-droid.org/packages/io.github.derweh.bayesianbahn/) —
the recommended route, because updates arrive automatically and because F-Droid
rebuilds the app from this repository and checks that the result matches the
APK published here, byte for byte, before it distributes anything.

The [GitHub releases](https://github.com/DerWeh/BayesianBahn/releases) carry the
same APKs for anyone who prefers to sideload. They are signed with the key
F-Droid pins for this package, so a sideloaded build and an F-Droid build
upgrade over each other without an uninstall.

Android 8.0 (API 26) or newer.

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
  sharpened towards runs that were similarly late. **A report only counts
  when it reports a delay**: DB states a stop in four shapes and three of
  them mean "on time", which is the plan restated rather than an
  observation — see `LiveReport` for what
  believing it cost. Trains without history
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
  base, rebuilt when the archive publishes a new month) and `recent.zip`
  (rebuilt daily by `pipeline/build_recent.py` from the archive's *raw*
  IRIS logs, covering the days newer than the newest monthly file). The
  in-app update checks `meta.json` first and fetches only the tier that
  changed; the app overlays recent runs onto the base.
- **Future trips**: IRIS publishes live plans only ~a day ahead. Beyond
  that the planner reconstructs station boards from the historical
  timetable (`pipeline/build_boards.py`: per station, every train that
  recently called there with typical times, weekday pattern and
  last-seen date) and predicts blind — clearly labelled in the UI, since
  timetable changes and construction can shift planned times.
- **On-demand shards**: trains outside the local data are fetched
  individually (a few KB each) from the repo's `shards` branch, where the
  workflow publishes the merged base+recent set daily. Fetched shards are
  cached on disk with an 18-hour refresh, so a commuter's usual
  connections cost one download and then work offline.
- **Shard format**: a columnar layout (deduplicated planned times,
  delta-coded dates, departure stored only when it differs from arrival)
  cut shard size ~63% versus naive per-run JSON rows; gzip keeps decoding
  dependency-free (brotli would save a further ~15% at the cost of a
  decoder dependency).
- **Backtesting**: `pipeline/backtest.py` walk-forward evaluates model
  variants on months of archive data with proper scoring rules (CRPS,
  pinball loss, interval coverage), reported with their upper tail rather
  than as means alone. The two evaluations answer different halves and
  neither replaces the other: the archive says how history should be
  weighted and whether a *genuine* live signal is worth conditioning on,
  and only the collected forecasts can say whether DB's number is a genuine
  signal — the archive records what the trains did, not what DB said they
  would do. On a 12-week eval (Easter–June 2026,
  91k predictions) the delta model cut live-scenario CRPS 3.2× versus
  ignoring live data (1.53 vs 4.83); a 30-day half-life beat 7/14/60 days;
  explicit holiday handling showed no benefit even across the April–June
  holidays. Parameters above are the backtest winners. That 3.2× is measured
  where the live signal is a *measured* previous-stop delay, which is not the
  same as where DB has said something. The `gated` scenario applies the
  shipped rule to that measured signal and is deliberately a control: over
  160,299 events it scores 0.580 CRPS against 0.462 for believing the signal
  and 1.906 for ignoring it entirely. Gating a real measurement should cost
  something, and it does — which is the evidence that the rule compensates
  for how DB reports rather than for a property of delays.

## What the journey search does not do

Current restrictions, all of them temporary and all of them things that make
the app report *fewer* connections than DB does — never wrong ones:

- **At most one change.** Journeys needing two or more changes are not
  searched at all. "No connection found" therefore means "none with at most
  one change", not "none exists".
- **A three-hour window** from the requested departure at the origin, and four
  hours at the transfer station.
- **The transfer search is a heuristic**, not an exhaustive one: stations on a
  train's route are ranked by distance to the destination and at most eight are
  evaluated, because each one costs a live board request. Measured with
  `tools/route_bench.py` against exhaustive ground truth — thousands of journeys
  over five days of the archive, each with a one-change connection that provably
  exists inside the windows the app itself searches — recall depends strongly on
  how busy the origin is:

  | origin | typical departures in the 3h window | found at the shipped budget |
  | --- | --- | --- |
  | village halt (weight < 40) | 5 | 89% |
  | small station (40–100) | 9 | 81% |
  | town (100–250) | 16 | 79% |
  | hub (250+) | 45 | **55%** |

  The pattern is the budget, not the ranking: eight transfer boards go a long
  way among five departures and nowhere among forty-five. Quoting one aggregate
  number hides this, and since most journeys start at the busy end, any average
  over stations flatters the cases people actually use. Of the misses, three
  quarters are transfer stations the ranking never reaches; the rest are excluded
  by the detour or weight filters.

  An earlier live measurement over 44 journeys reported 98%. That number was
  too kind: its ground truth was built by walking the same station boards the
  search walks, so it could only ever pose journeys the mechanism already sees.
- **Predictions need history.** A connection can be found and still not be
  shown when too few historical runs exist for the trains involved.

Cross-check with DB's own apps before relying on any of it.

## Roadmap

- Journeys with more than one change. The current board-scanning heuristic
  cannot be extended to two changes without the number of live board requests
  exploding; doing this properly means an exact timetable router (RAPTOR or the
  Connection Scan Algorithm) over a real timetable feed such as GTFS.
- Widen the data beyond the draft subset (trains calling at Augsburg Hbf /
  München Hbf) — the pipeline and the update mechanism already handle any
  station list.
- Condition on the *true* previous-stop live delay instead of the current
  station's report.
- **Score complete journeys, not only their parts.** The evaluation scores an
  arrival time for a journey without a change and a catch probability for one
  with a change, and cannot compare the two: it never follows a passenger
  through a transfer to their destination. Answering "is the app better where
  you have to change?" needs the connecting train's *destination* arrival in
  the collected forecasts, which `collect_forecasts.py` does not record today.
  With it, a one-change itinerary could be scored end to end — predicted final
  arrival against the arrival that happened — in the same units as a direct
  journey. Tagging arrivals by whether a connection leaves from them is not a
  substitute: 96% of them qualify, and the remainder is last trains and
  terminus arrivals rather than direct journeys.
- On-device [TabICL v2](https://github.com/soda-inria/tabicl) (BSD-3) via ONNX
  Runtime as the conditional model: context = this connection's historical
  runs, query = today's features, output = full predictive distribution.

## Building

Toolchain is pinned with [pixi](https://pixi.sh); the Android SDK must be
available via `local.properties` or `ANDROID_HOME`.

```sh
pixi run ./gradlew assembleDebug      # build the APK
pixi run ./gradlew testDebugUnitTest  # run unit tests
pixi run -e pipeline test-pipeline    # data-pipeline tests
```

The pipeline tests matter more than their size suggests: the nightly job's
monthly path only executes when a new archive file appears, so without them a
break in it stays invisible for weeks.

### Running it on an emulator

The unit tests cover the logic; the emulator is for what they cannot check —
that a search actually returns something, and how long it takes.

One-time setup. `emulator` and a system image are separate SDK packages and are
not pulled in by a Gradle build:

```sh
# local.properties points Gradle at the SDK, but not the shell.
export ANDROID_HOME=$HOME/android-sdk        # or whatever sdk.dir says there
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"

sdkmanager platform-tools emulator "platforms;android-35" \
           "system-images;android-35;default;x86_64"
avdmanager create avd -n bb -k "system-images;android-35;default;x86_64" -d pixel_6
```

Start it, and wait until Android is actually up — `adb devices` reports the
device long before the system has booted:

```sh
$ANDROID_HOME/emulator/emulator -avd bb -no-boot-anim &
adb wait-for-device
until [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = 1 ]; do sleep 2; done
```

Install and launch:

```sh
pixi run ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell monkey -p io.github.derweh.bayesianbahn -c android.intent.category.LAUNCHER 1
```

To test what F-Droid will publish, install the release APK instead
(`assembleRelease`, needs `keystore.properties`). Debug and release builds are
signed with different keys, so switching between them needs
`adb uninstall io.github.derweh.bayesianbahn` first.

While testing:

```sh
adb logcat --pid=$(adb shell pidof io.github.derweh.bayesianbahn)  # app log only
adb exec-out screencap -p > screen.png                            # screenshot
adb shell dumpsys package io.github.derweh.bayesianbahn | grep version
adb emu kill                                                      # stop it
```

Some notes that cost time to rediscover:

- **Hardware acceleration.** Without `/dev/kvm` the emulator falls back to
  software and a cold boot takes many minutes. Check with `ls -l /dev/kvm`; the
  user must be in the `kvm` group (`sudo usermod -aG kvm $USER`, then log in
  again).
- **Under WSL2** this works when WSLg provides the display (`echo $DISPLAY`
  should print something) and nested virtualisation is enabled on the Windows
  host. If the window stays black, `-gpu swiftshader_indirect` renders on the
  CPU. Alternatively run the emulator on Windows and reach it from WSL with
  `adb connect`.
- **The app needs the network** — every search hits DB's IRIS API — so an
  emulator without working DNS shows only "could not reach DB's live
  timetable".

### Backtesting

```sh
pixi run -e pipeline python pipeline/backtest.py \
    --data-dir pipeline/data --stations 8000013,8000261 --eval-weeks 12
```

### Benchmarking the journey search

`tools/route_bench.py` measures the routing heuristic offline. Each archived
stop carries a ride id and a stop number, so one day of the archive rebuilds
into the whole national timetable — ~42k train runs over ~5200 stations, which
covers 97% of the transfer-eligible stations in the app's station list. Ground
truth is then exhaustive rather than sampled, and costs no API requests:

```sh
pixi run -e pipeline python tools/route_bench.py snapshot --day 2026-06-10
pixi run -e pipeline python tools/route_bench.py bench --day 2026-06-10 --queries 1200
pixi run -e pipeline python tools/route_bench.py sweep --day 2026-06-10 --queries 1200
```

### Comparing the predictions against DB's own

Does the app actually beat the number already on the platform display? That
needs DB's forecast *as it changed over time*, which exists nowhere
retrospectively — the archive records what a train did, never what was predicted
beforehand. So it is collected live:

```sh
# runs continuously; append-only journal, resumes after a restart or power cut
pixi run -e evaluate collect
pixi run -e evaluate collect-status   # rounds, missed slots, stations answering
```

Twenty stations (`tools/forecast_stations.csv`, stratified and fixed before any
data existed) are polled every ten minutes with jitter. The archive publishes
ground truth the next morning, after which one command scores the day end to end
and rebuilds the report:

```sh
pixi run -e evaluate evaluate 2026-08-17 [2026-08-18 ...]
```

The `evaluate` environment is the only one carrying both toolchains — polars to
build the events, the JDK to run the model over them. Stages skip work already
done, so adding a day costs only that day.

That runs the shipping Kotlin model — not the Python mirror in `backtest.py` —
over the recorded forecasts via the opt-in `ForecastHarness` unit test, with the
history trimmed to runs strictly before the day being predicted. `tools/report.py`
renders the result as a self-contained HTML page carrying its own definitions,
caveats and reproduction commands — including the commit it was generated from
and whether that commit is a released version, since the page is rendered from a
working tree and the model it scores is not always one that has shipped.

To publish it:

```sh
pixi run -e evaluate publish-evaluation 2026-08-17 2026-08-18   # -> docs/evaluation/
git add docs/evaluation && git commit && git push
```

GitHub Pages serves `/docs` from the default branch. This is deliberately not a
CI job: the collector runs on a laptop and the archive is fetched locally, so no
runner has the data to rebuild the page.

Every comparison is reported with a 95% interval from a bootstrap over *trains*,
not over predictions — one late train produces a dozen correlated predictions, and
treating them as independent manufactures confidence that is not there. Each day
is also shown unpooled, because the first pass at this read a single evening's 189
missed connections as a result that the next day did not reproduce.

As of three days (2026-08-17/18/19, 45,408 arrival predictions, 67,796
connections) the app's point forecast beats DB's by **0.65 min of CRPS
[0.60, 0.69]** and is markedly better on the connections that actually failed
(Brier −0.207 [−0.183, −0.232]). It wins in every lead-time bucket and on every
collected day.

Both figures come from taking DB's live number seriously only when it reports a
delay. Before that change the app beat DB by 0.24 min and its stated 80% range
held 53–61% of arrivals; the range now holds **78–86% across every lead bucket**
(77.7 / 84.1 / 80.5 per day) against a nominal 80%. The change is a single
condition in `Predictor`, and its effect was predicted to three decimal places
by scoring the rule against already-collected forecasts before it was written.

What remains open is the other half: where DB *does* report a delay it
understates it, saying +7.2 min for trains that arrive +10.6 on average. Beyond
90 minutes out, believing such a report is still marginally worse than ignoring
it (2.70–2.74 against 2.69–2.71 for history alone), which is the residue of the
previous-stop approximation documented in `Predictor`.

`tools/journey_bench.py` does the same thing against live IRIS. It is limited to
a few dozen journeys by politeness to a keyless public API, so it is no longer
what the constants are tuned on — but it is the only one of the two that
exercises name resolution, since offline every station is an EVA number and
identity is exact. Use it to confirm that a change measured offline survives
contact with the real API, on a handful of journeys.

### Regenerating data

```sh
# download monthly parquet files into pipeline/data/, then:
pixi run -e pipeline python pipeline/build_shards.py \
    --data-dir pipeline/data --out-dir pipeline/output --stations 8000013,8000261
cp pipeline/output/shards/*.jgz pipeline/output/index.json app/src/main/assets/history/
```

The bundled station list (`app/src/main/assets/stations.csv`) derives from
[db-stations](https://github.com/derhuerst/db-stations) (DB open data, CC BY 4.0):

```sh
pixi run -e pipeline python pipeline/build_stations.py
```

Its `eva;name;weight;lat;lon` rows carry coordinates because the journey search
ranks possible transfer stations by how far they are from the destination —
ranked by station size instead, a search from Ulm to Türkheim spent its first
attempts on Stuttgart Hbf, 130 km the wrong way.

## License

MIT — see [LICENSE](LICENSE). Bundled data: CC BY 4.0 by Deutsche Bahn
(station list and delay history).
