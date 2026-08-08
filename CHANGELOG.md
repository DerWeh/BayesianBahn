# Changelog

All notable changes to BayesianBahn are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (0.x: minor = features, patch =
fixes; expect breaking changes between minors until 1.0).

## [Unreleased]

### Changed
- The journey search picks transfer stations by how close they are to the
  destination instead of by how big they are. It used to spend its limited
  search budget on the largest station on a train's route — for Ulm →
  Türkheim (Bay) that meant Stuttgart Hbf, 130 km the wrong way, and the
  change at Memmingen that works was reached only on the fourth of six
  attempts. The bundled station list now carries coordinates for this.
- The journey search tries up to 8 transfer stations instead of 6. Measured
  over 44 journeys that provably have a one-transfer connection, 6 attempts
  find 93% of them and 8 find 98%, with no further gain up to 12.

### Fixed
- Journeys to stations whose name differs between data sources ("Türkheim (Bay)
  Bahnhof" in the station list, "Türkheim(Bay)Bf" in DB's route data) reported
  "No plannable trains found" even for routes served every hour. Stations are
  now identified by their EVA number, asking DB which name it uses, instead of
  by comparing spellings. On a sample of 100 stations, about one in five is
  spelled too differently for any string comparison to match.
- Losing the network no longer ends a search with a raw socket error. The
  downloaded history is a timetable in its own right, so the search falls back
  to it and says that the times carry no live delays.

## [0.1.1] - 2026-07-19

### Added
- Histogram x-axis shows clock times (ticks on round minutes) instead of
  bare +X delay offsets.
- Itinerary cards show the expected travel time alongside the arrival.
- Connections departing shortly before the feeder's arrival are listed
  with their (near-zero) catch probability instead of being hidden — a
  delayed one is occasionally exactly the train that works.
- Opt-in cross-check test against DB's official journey planner
  (`NavigatorCompareE2E`, via the transport.rest proxy).

### Fixed
- The headline "P(first connection)" refers to the first plannable
  connection, not a normally-missed earlier train.

## [0.1.0] - 2026-07-19

First draft release. Predictions are experimental — always cross-check
times and connections with DB's official apps.

### Added
- From/to journey search with direct trains and one-transfer connections;
  arrival predictions as full distributions (median, 80% credible interval,
  histogram) computed from each train's real delay history, not DB's
  forecast.
- Bayesian transfer propagation: the passenger boards the first connecting
  train that has not left yet — per-train catch probabilities, honest
  bimodal distributions when a connection is at risk, and the probability
  of missing all listed connections.
- Live conditioning: when IRIS reports a delay, the delta model anchors the
  prediction on it (backtested: 3.2× lower CRPS than ignoring live data).
- Deutschland-Ticket filter (regional trains only), on by default when the
  feeder is regional.
- Departure time and date pickers; dates beyond DB's ~1-day plan horizon
  are planned from the historical timetable (weekday-aware, public
  holidays run the Sunday timetable) with a clear warning, including when
  the date crosses the biannual timetable change.
- Live station boards and per-train predictions (cancellation rates
  included) behind the top-bar icon.
- Country-wide coverage via on-demand per-train data (a few KB per train,
  cached ~a day, works offline once fetched); bundled starter data for the
  Augsburg/München region; optional bulk download for offline use,
  refreshed daily to within ~a day of reality.

[0.1.1]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.1.1
[0.1.0]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.1.0
