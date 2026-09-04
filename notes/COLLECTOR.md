# The forecast collector, and the drift check it feeds

**Read this if the "Live-anchor spread has drifted" issue just opened, or if
the collector has been quiet and you want to know whether that matters.**

Written to be read cold, months later, by someone (or some model) with no
memory of building it.

## Why any of this exists

The app takes DB's live delay report and re-anchors its prediction on that
number, treating it as exact. It is not. On the stops it actually anchors on —
those DB called at least a minute late, 4% of all stops — the central 80% of
(final delay minus what DB was saying) runs from 4 minutes at a five-minute
lead to 45 at two hours. The model states an interval about two minutes wide,
and **40% of arrivals land above its own 90th percentile**. That is its
worst-calibrated part, and it is worst exactly when a passenger is on a
platform deciding whether to run.

`tools/sensitivity_live.py` measured what fixing it is worth: CRPS 4.08 -> 3.37
and 40.1% -> 6.6% above the 90th percentile, on days the fit never saw.

Fixing it means baking a *spread-versus-lead* curve into the model. A baked-in
curve is a claim about the world, and claims about the world go stale:
timetable recasts, new interlockings, a different disruption regime. So the
curve needs a watchdog.

**Nothing in the published archive can be that watchdog.** The archive
(piebro/deutsche-bahn-data, which `pipeline/` builds shards from) records what
trains *did*, not what DB *said they would do*. The residual this is all about
— final delay minus DB's earlier forecast — exists nowhere except in a journal
somebody captured live. That is the collector. A day nobody collected is a day
gone for good, which is why it runs as a job and not as something a person
remembers to start.

## The parts

| | |
|---|---|
| `tools/collect_forecasts.py` | polls IRIS every 10 min, writes append-only JSONL. Stdlib only, on purpose. `WINDOW_HOURS` lives here. |
| `.github/workflows/collect-forecasts.yml` | runs it daily for the window; publishes journals as assets on the `forecasts` release |
| `tools/score_events.py` | turns journals into scoreable events (`read_day`, `build_events`) |
| `tools/anchor_drift.py` | builds the spread-versus-lead curve and compares it to the reference |
| `tools/anchor-reference.json` | the frozen curve, regenerated deliberately |
| `.github/workflows/anchor-drift.yml` | Mondays: rebuild the curve on the last fortnight, open an issue if it moved |
| `tools/calibrate_live.py` | the one-off that sized the problem in the first place |
| `tools/sensitivity_live.py` | fits the width against archive truth, and shows how wrong it may be and still pay |

## The two decisions worth knowing before you touch anything

**The window is 15:00–21:00 Berlin, not the whole day.** Peak-hour residuals
are not all-day residuals, so anything compared against the reference must be
measured over the same hours. `anchor_drift.residuals` clips whole-day journals
with `score_events.read_day(..., within=)` for exactly this reason. The hours
are defined once, in `collect_forecasts.WINDOW_HOURS`; both the workflow and
the analysis read them from there, and `tools/tests/test_anchor_drift.py` fails
if either grows its own copy.

**Only stops DB called at least a minute late are counted.** Below that the app
does not anchor at all (`LiveReport.MIN_INFORMATIVE_DELAY_MINUTES`), so the
residual of a forecast it never makes is not evidence about it — and the
unconditional residual is a different animal: dominated by the 96% of stops
called on time, it saturates around 9 minutes an hour out, while the one the
model depends on keeps widening past 50. A monitor watching the first would sit
still through a regime change in the second.

**Truth is DB's *settled* forecast, not the train's real arrival.** The archive
would be better and lands weeks too late to be a watchdog. Two consequences,
both real and neither a bug to fix in a hurry:

* The widths here are narrower than the archive-truth ones
  `tools/sensitivity_live.py` fits, because DB's last word sits closer to DB's
  earlier word than the train's real arrival does. Constants to ship are fitted
  there, against the archive; this only watches for movement. **This monitor watches the curve move; it
  does not restate the calibration.** If you ship calibration constants, size
  them with `calibrate_live.py`, not with `anchor-reference.json`.
* A stop DB never mentioned before it settles has no truth to read, so it drops
  out. The residuals describe trains DB had an opinion about, which are not all
  trains.

## When the drift check fires

The check calls drift only when a bin is **both** outside its bootstrap
interval **and** moved by ≥30%. That threshold was sized against measured
noise, not chosen: splitting the 2026-08 blockade fortnight in half moved the
long-lead widths from 9.0 to 11.0 minutes — a 22% swing between two adjacent
weeks of the *same* regime. The end of the blockade moved them 35%. So an
issue means something bigger than a bad week.

Work through it in this order.

1. **Is it the collector or the railway?** Open the run and read the Health
   step for the fortnight. Missed slots, failed polls, or a station count below
   281 change the *composition* of the sample, and a composition change looks
   exactly like a regime change. Rerun the check on days where collection was
   clean before believing anything.

2. **Which way did it move, and where?** A uniform shift across every lead bin
   is a regime change — a blockade starting or ending, a timetable recast. A
   move in the short bins only, with the long ones steady, is more likely DB
   changing how it publishes near-term forecasts.

3. **Did something end?** The big one on record: a rail-replacement blockade in
   the collection region ran until 2026-08-31, and its end narrowed every bin
   by ~30% overnight. Check
   [bauinfos.deutschebahn.com](https://bauinfos.deutschebahn.com/) and the
   annual timetable change (second Sunday in December) before looking for a
   subtler cause.

4. **Re-freeze, or act?** If the new regime is the world now, re-freeze:

   ```
   pixi run -e pipeline python tools/anchor_drift.py reference \
     --days 2026-09-01..2026-09-14 --note "why you re-froze it"
   ```

   Commit the JSON with the reason in the message. **If calibration constants
   have shipped in the app by then, re-freezing alone is not enough** — the
   model is still answering with the old spread. Re-run `calibrate_live.py`
   against archive truth and open a release issue.

## Restarting the whole thing from cold

The collector holds no state between runs beyond the journal itself: the
schedule is a pure function of the wall clock, and a restart replays the day's
journal to rebuild what it had already reported. So:

* **Run it by hand:** Actions → *Collect DB forecasts* → Run workflow. It waits
  for the window, so a dispatch at 09:00 idles until 15:00; pass `minutes` to
  override and collect right now instead.
* **Locally:** `python tools/collect_forecasts.py run --minutes 360`, then
  `status` for a health summary. No pixi environment needed.
* **Get the journals:** they are assets on the `forecasts` release,
  `forecasts-YYYY-MM-DD.jsonl.gz`, pruned after 90 days.
  `gh release download forecasts -p 'forecasts-2026-09-*.jsonl.gz' -D tools/.forecasts`
  then `gunzip`.
* **Rebuild the reference from local journals:** the `reference` command above.

## Things that were tried, or ruled out, so you do not redo them

* **Running it round the clock.** Four times the runner minutes for an answer
  to a question nobody asks at 04:00. The window is the decision.
* **Keeping journals in git.** ~11 MB gzipped a day is a gigabyte a year of
  history that can never be pruned. Release assets can.
* **Archive truth in the weekly check.** Correct, and weeks late — useless as a
  tripwire. It stays the right tool for sizing the calibration itself.
* **A shorter check window.** Below a fortnight the week-to-week swing above
  swamps the signal; `anchor_drift.MIN_DAYS` warns when either side is thinner.

## Is this a permitted use of Actions? Yes

Worth writing down, because it looks like the kind of thing that should not be.
The rule for GitHub-hosted runners prohibits "any other activity unrelated to
the production, testing, deployment, or publication of the software project
associated with the repository", alongside named bans on cryptomining, CDN and
serverless use, reselling Actions, and "any activity that places a burden on
our servers... disproportionate to the benefits provided to users".

These journals are the test data for the model this repository ships, and
nothing else reads them. That is testing the software project associated with
the repository — the permitted case, not an exception to it. Six hours a day of
one standard runner on a public repo, next to a nightly pipeline job that
already runs here, is not a disproportionate burden.

What would change the answer: collecting stations the app's evaluation does not
use, keeping the journals for something other than this repository, or growing
the window without a use for the data. If the window ever has to shrink, shrink
the *window* and not the cadence — a shorter window costs sample size in a way
the code already understands (`MIN_EVENTS`, `MIN_DAYS`), while a longer cadence
changes what a lead time means and invalidates the reference.

Source: [GitHub Terms for Additional Products and
Features](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features),
read 2026-09-04.

## One thing about the CI move that was never tested here

* **IRIS from a GitHub runner.** Every reading so far was taken from a domestic
  German connection. IRIS publishes no rate limit and no terms, but a
  datacentre IP is not a domestic one, and a block would show up as a wall of
  `HTTPError 403` in the Health step — which `error_name` records precisely so
  that this case is distinguishable from an outage. If that happens, the
  fallback is a small always-on machine rather than a runner.

## Known state, 2026-09-04

`tools/anchor-reference.json` is **provisional**: it covers 2026-09-01..09-03,
the only three days after the blockade ended. Three days is below `MIN_DAYS`,
and the check will say so on every run. **Re-freeze it once CI has collected 14
post-blockade days** (around 2026-09-18) — that is the first thing to do here,
and it is a one-line command plus a commit.

The calibration itself has not shipped. `DelayModel.LIVE_SHRINKAGE = 0.4` and
`MIN_LIVE_SCALE = 1.2` are still what the app uses, and they were not fitted to
anything. `tools/sensitivity_live.py` has done the work that decides whether
they should be replaced; what it found, in one line: the width may be wrong by
a factor of two either way and the fix still beats what ships, so the risk of
shipping it is small and the risk of not shipping it is 40% of arrivals landing
above their stated 90th percentile.
