# Changelog

All notable changes to BayesianBahn are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (0.x: minor = features, patch =
fixes; expect breaking changes between minors until 1.0).

## [Unreleased]

### Added
- **A forecast for trains whose run number is too new to have one.** IRIS gives
  every run its own number and renumbers at each timetable change, so a train
  that has run for years can arrive with almost nothing behind it: over eleven
  collected days a quarter of arrivals fell through to a class-wide prior that
  knows neither the station nor the hour. Those trains still have a line, and
  the line has run all along. `pipeline/build_shards.py` now publishes a second
  set of shards keyed by line and station, and the app reads one when — and only
  when — the train's own history comes up short. Walk-forward over 781,000
  archive events at 62 stations, on exactly the events the prior answers today,
  the line scores 0.43 min of CRPS better (95% 0.40..0.46, resampling whole
  trains) with a shard available for 88% of them; across the December 2025
  timetable change, where the population doubles to 11% of events, the same
  0.41. Regional and S-Bahn both gain; long distance almost never carries a
  line number, so there is nothing to say about it either way. The screens say
  which line the numbers came from rather than claiming they are this train's.

### Changed
- **The line shard is a separate lookup, not a second candidate key.**
  `HistoryRepository.load` had always asked for a line-keyed shard when the
  number's key missed. Now that those shards exist, that would have handed a
  line's pooled runs to every caller — including the two-leg model, which pairs
  a candidate's departure and arrival by date and would have joined one train's
  departure to another train's arrival. Line-keying also loses to number-keying
  wherever the number has a history (0.13 min of CRPS, 95% 0.12..0.13), so it
  is now reached only through the fallback that measured better.

## [0.3.0] - 2026-08-29

### Fixed
- **Picking a date without touching the time searched from 06:00.** The
  departure button said "now", and on today it meant it — but choosing another
  day quietly replaced it with six in the morning, so a search for Saturday made
  at lunchtime answered with the first train of the Saturday morning. Nothing in
  the app showed that time or offered a way to see it. "Now" is now the current
  time of day on whichever date is picked, which is what the button has been
  saying all along.
- **A change at a busy station offered six trains that had already left.** For
  a journey with one change the app lists up to six connecting trains, and it
  deliberately includes ones leaving shortly *before* the feeder is due —
  usually missed, but a delayed one is occasionally exactly the connection that
  works. It took the first six by departure time, so where a station has a
  service every few minutes all six were in the past before the feeder even
  arrived: six impossible trains and no possible one, each honestly reported as
  unreachable. Nothing inside the app could notice, because every number it
  showed was right. At most two of the six may now be trains already gone and
  the rest is filled forward, with the reverse allowance late in the day when
  there is only one train left ahead. Found by the two-leg evaluation, where it
  accounted for 31% of the journeys with a change on a single collected day.
- **A predicted arrival could be a day late.** Where a journey ends, the IRIS
  board gives a departure and nothing beyond it, so the arrival was recovered
  by taking the destination's most recent *time of day* from history and
  hanging it on today's date — rolling the date forward whenever that landed
  before the departure. A timetable that had shifted by half an hour was enough
  to trigger it: a 28-minute leg was published as 24 hours and 28 minutes.
  Departure and arrival are now read from the same historical run and the leg
  between them applied to today's departure, so a schedule that shifts moves
  both; the median run sets it, and a leg longer than 14 hours is declined
  rather than published. This affected journeys with a change and direct
  journeys alike — both recovered the arrival the same way. Found by the
  two-leg evaluation on its first end-to-end run, where four candidates in six
  landed a day out on one collected day.

### Changed
- **The search for a change no longer spends its budget on the first trains to
  leave.** A journey with one change costs a live board request per change
  evaluated, and eight are affordable. Those eight used to be handed out feeder
  by feeder in departure order, two stations each, so at a station like Ulm Hbf
  the first four departures used the whole budget while forty other trains were
  never looked at — and when a train leaves says nothing about whether changing
  off it works. The origin's board is a single fetch and every train on it
  arrives with its own route, so all the candidate changes are known before any
  attempt is spent; they are now ranked together, nearest the destination first,
  and the budget goes to the best of them wherever they sit in the board. No
  extra requests: measured over 4,000 archived journeys the mean spend is
  unchanged at 5.8 attempts. Journeys found rise from 82% to 86% overall and
  from 57% to 73% at the biggest origins, and the itinerary found arrives 5
  minutes earlier on average. A station is opened once, and a train that has
  already yielded an itinerary is not changed off a second time — that is the
  same departure by another route, not a second option.
- **Reading a train's delay history is about ten times faster, and drawing its
  histogram far more.** A history shard was decoded by walking a JSON tree and
  asking each field for its number, which is a *string* parse per field per
  run — a median shard holds nine hundred runs across ten stations, so opening
  one train cost several thousand of them. Decoding straight into typed fields
  cuts parsing 600 shards from 1.4 seconds to 0.14; building the tree had never
  been the expensive part, at 88 milliseconds of that. Separately, the
  distribution behind the histogram and the 80% interval answered every
  question by scanning its whole list of points — one per past run per
  connecting train, so hundreds to thousands — where it now binary-searches a
  running total it builds once. Both run on the phone: the first on every shard
  the app reads, the second on every histogram it draws. The evaluation, which
  asks for a cumulative probability 661 times per scored journey, went from an
  hour to six minutes for nine days.

- **The evaluation waits for the next train that really runs.** A journey with
  a change was scored only if the passenger boarded one of the six trains the
  app had offered; if they missed all six and took the seventh, the journey was
  dropped. That excused the answers the app got most wrong — 30% of the two-leg
  journeys on one collected day — so the walk to the train actually boarded now
  runs to the end of the day, while both forecasters still answer over the
  app's own six. The published margin over DB on these journeys falls from
  3.001 to 2.216 minutes of CRPS as a result, and the page states outright how
  often the passenger boarded past the list.

### Added
- **An itinerary now says when DB reports a disruption on it.** DB states
  trouble in more than one place and the app read only one of them. A blocked
  section is not a cancellation: the trains keep their times, the cancellation
  flag stays unset, and the journey is impossible anyway — DB reports it in a
  notice element that both the app and the collector parsed straight past. One
  Memmingen document carries 180 of them, 21 marked as a disruption. So an
  evening where no passenger could travel was recorded as an evening where
  every train ran to time, and the app would have shown a confident prediction
  for a trip DB was publishing as impossible. An itinerary whose feeder or
  connecting train carries one now says so above the times rather than below
  them, because it is the reason not to trust them; the prediction is still
  shown, as DB's own apps show the times too, but no longer silently. Only
  notices categorised as a disruption count — roadworks are attached to half
  the stops in Germany and say nothing about today, so warning on those would
  warn on everything. The validity window and timestamps are not stored: a
  construction notice valid for three months would otherwise ride along on
  every poll of every stop it touches. The evaluation records the flag per
  event without acting on it yet — it is a different failure from a
  cancellation and deserves its own count first.
- **The evaluation scores a complete two-leg journey**, not only its parts: the
  predicted arrival at the far end of a change against the arrival that
  happened, in the same units as a direct journey, so for the first time the
  two kinds of journey can be compared with each other rather than each only
  with DB. Both forecasters answer over the same candidate trains and from the
  same moment, and neither is shown a delay the other was denied: DB's answer
  is the arrival of whichever train its own forecasts say the passenger
  catches. The harness drives the shipping code — `CandidateBuilder`, which the
  app itself uses, and `ConnectionModel.propagate` — rather than a description
  of it.

## [0.2.0] - 2026-08-24

### Added
- A German interface, used on a phone set to German; English stays the default
  and the fallback. Every text the app shows is now a translatable resource,
  including the failures raised while planning a journey, which used to be
  written out in English deep in the planner. Android 13 and newer can set the
  language for this app alone.
- The forecast collector polls a second tier of stations: where the trains
  leaving the twenty pre-registered ones end up. A journey with a change ends
  at the far end of the second leg, IRIS serves forecasts one station at a
  time, and without a reading there DB's answer for that arrival cannot be had
  at all — which is why the evaluation could score the parts of a two-leg
  journey but never the journey. The tier is derived from the timetable rather
  than chosen (`tools/build_destinations.py`), it is recorded per poll so a
  later edit to the station files cannot re-label data already collected, and
  it never originates a scored arrival or connection: the pre-registered set is
  unchanged. Each departure's whole onward path is now kept as well; only its
  next stop was, which is what made the far end unrecoverable after the fact.

### Changed
- **DB's live number is only used when it reports a delay.** DB states a stop
  in four shapes and three of them mean "on time"; only one is an observation.
  Over seven collected days, DB called a train on time for 69% of stops ten
  minutes before departure and 99% of stops three hours out — and 26% of that
  last group arrived more than two minutes late. Reports of "early" are ignored
  on the same grounds: those trains averaged 1.0 minute late. Measured over the
  first three days, anchoring the forecast on that number cost 0.53 minutes of
  CRPS on trains that have history and left the stated 80% range covering 55%
  of arrivals; with the rule in place that range covers 78-89% of arrivals,
  day by day. The app will now disagree with the platform display for most
  trains, and the prediction screen says why.
- The connection model applies the same rule to a live *departure* report. It
  had been treating one as fact — reported later than the passenger can arrive
  meant missed, otherwise caught, with no distribution in between — so a train
  with a history of leaving late became a certainty on the strength of a
  restated timetable.
- The evaluation report shows the distribution of the errors, not only their
  averages: a box plot per lead-time bucket and the tail as figures. The
  medians of the two forecasts are close and the difference between them is in
  the large errors, which an average cannot show. With a weekend in the data it
  also splits working days from the weekend, which had been a caveat it could
  not check: DB's own mean error is 3.14 minutes from Monday to Friday against
  1.84 at the weekend, so the two are different problems.
- Over the seven days published in
  [the evaluation](https://derweh.github.io/BayesianBahn/evaluation/), this
  release scores 0.523 minutes of CRPS below DB's own forecast (95% interval
  0.489 to 0.556) across 121,395 predictions, and 0.211 Brier below it on the
  7,246 connections that were actually missed. Without the rule above the same
  model is 0.310 below DB on arrivals and cannot be separated from it on missed
  connections at all. Two caveats travel with those numbers: seven days is a
  small sample for something that clusters by line and by incident, and every
  feasible change counts as a connection, including ones nobody would make.

### Fixed
- The English interface says "Platform" where it said "Gl." — the German
  abbreviation for *Gleis*. It had been hardcoded in two screens since long
  before there was a German translation for it to belong to, and the
  extraction into resources carried it across faithfully.
- A forecast drawn from a single past run no longer reads "1 past runs". Both
  sentences that count runs are plurals now, in both languages, and so is the
  effective-run count one of them ends on — a plural chooses on one number,
  and either of those two can be 1. The English wording carried this from the
  start; translating it is what made it visible.

## [0.1.4] - 2026-08-22

### Fixed
- The F-Droid listing shows the app icon. The app ships an adaptive icon and
  nothing else, which is everything a device running Android 8 or newer needs,
  but F-Droid builds its listing icon from a plain raster inside the APK and
  there was none — so the app was listed with no icon at all. A PNG rendered
  from the same launcher vector now ships alongside the store description. No
  code changed; this release exists to get that metadata rebuilt.

## [0.1.3] - 2026-08-20

### Fixed
- Opening a station board and going back no longer empties the From and To
  fields. The results stayed on screen while the fields behind them were
  cleared and the search button was disabled, so the search could not be rerun
  without retyping both stations. The chosen date and time, the
  Deutschland-Ticket setting and scroll positions were being lost the same way.
- A train running early is labelled `-1` instead of `+-1`. Delays under a
  minute now read as on time rather than `+0`, and a train 1.6 minutes late
  reads as `+2` rather than `+1`.

### Changed
- The evaluation report shows the delay curve over the day and checks the
  model's time bands against it, and states where the live-number crossover
  falls. Three collected days confirm that delay accumulates from morning to
  mid-afternoon; the shipped time bands were cut from the commuter timetable
  and do not line up with it.
- Searching a date beyond the live timetable is much faster. The historical
  timetable needs one delay-history file per departure, and those were fetched
  one after another for every station the search looked at; they now load
  together and are reused across the search instead of being fetched again for
  each place they appear. Trains the search cannot use — long-distance ones in
  a Deutschland-Ticket search — are no longer fetched at all.

## [0.1.2] - 2026-08-10

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

- "No plannable trains found" now says what it means: only direct journeys and
  journeys with one change are searched, so DB's apps may still find a route
  with more changes. The restriction is also stated in the results list, the
  README and the store description.

### Fixed
- Journeys to stations whose name differs between data sources ("Türkheim (Bay)
  Bahnhof" in the station list, "Türkheim(Bay)Bf" in DB's route data) reported
  "No plannable trains found" even for routes served every hour. Stations are
  now identified by their EVA number, asking DB which name it uses, instead of
  by comparing spellings. On a sample of 100 stations, about one in five is
  spelled too differently for any string comparison to match.
- "Plan a connection from this train" no longer dead-ends with `Transfer
  station "..." not found.` The transfer stations it offers are named by DB's
  route data ("Frankfurt(M) Flughafen Regionalbf") while the station list
  spells them out ("Frankfurt (Main) Flughafen Regionalbahnhof"); they are now
  matched by station number.
- Refreshing the delay history retries a reset connection instead of showing an
  error. A 15 MB download hit by an HTTP/2 stream reset used to fail the tap,
  and worked when tapped again.
- Searching a date more than a day ahead says what it is doing and that it can
  take a few minutes, instead of showing a bare spinner that looks like a hang.
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

- "No plannable trains found" now says what it means: only direct journeys and
  journeys with one change are searched, so DB's apps may still find a route
  with more changes. The restriction is also stated in the results list, the
  README and the store description.

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

[0.3.0]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.3.0
[0.2.0]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.2.0
[0.1.4]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.1.4
[0.1.3]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.1.3
[0.1.2]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.1.2
[0.1.1]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.1.1
[0.1.0]: https://github.com/DerWeh/BayesianBahn/releases/tag/v0.1.0
