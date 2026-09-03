# The forecast collector, and the drift check it feeds

**Read this if the "Live-anchor spread has drifted" issue just opened, or if
the collector has been quiet and you want to know whether that matters.**

Written to be read cold, months later, by someone (or some model) with no
memory of building it.

## Why any of this exists

The app takes DB's live delay report and re-anchors its prediction on that
number, treating it as exact. It is not. The central 80% of (final delay minus
what DB was saying) spans 4 to 10 minutes over the blockade fortnight
2026-08-18..08-31, and 3 to 7 in the three days after that blockade ended —
widening with lead time in both. The model states an 80% interval about two
minutes wide. That is its worst-calibrated part, and it is worst exactly when a
passenger is on a platform deciding whether to run.

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

## The two decisions worth knowing before you touch anything

**The window is 15:00–21:00 Berlin, not the whole day.** Peak-hour residuals
are not all-day residuals, so anything compared against the reference must be
measured over the same hours. `anchor_drift.residuals` clips whole-day journals
with `score_events.read_day(..., within=)` for exactly this reason. The hours
are defined once, in `collect_forecasts.WINDOW_HOURS`; both the workflow and
the analysis read them from there, and `tools/tests/test_anchor_drift.py` fails
if either grows its own copy.

**Truth is DB's *settled* forecast, not the train's real arrival.** The archive
would be better and lands weeks too late to be a watchdog. Two consequences,
both real and neither a bug to fix in a hurry:

* The widths here (3–10 min) are narrower than `calibrate_live.py`'s
  archive-truth widths (6–45 min) — settled truth is DB's own final number, and
  DB's last word is closer to DB's earlier word than the train's real arrival is. **This monitor watches the curve move; it
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

## Two things about the CI move that were never tested here

Both are unknowable without a push, so they are written down rather than
guessed at.

* **IRIS from a GitHub runner.** Every reading so far was taken from a domestic
  German connection. IRIS publishes no rate limit and no terms, but a
  datacentre IP is not a domestic one, and a block would show up as a wall of
  `HTTPError 403` in the Health step — which `error_name` records precisely so
  that this case is distinguishable from an outage. If that happens, the
  fallback is a small always-on machine rather than a runner.
* **Whether this is what Actions is for.** GitHub asks that Actions serve the
  repository's own software. This is the project's evaluation harness and its
  output feeds the model that ships, so the case is good — but it is six hours
  of runner a day, indefinitely, and that is worth being honest about rather
  than discovering in an email. Reduce the window before reducing the
  cadence: a shorter window costs sample size in a way the code understands
  (`MIN_EVENTS`, `MIN_DAYS`); a longer cadence changes what a "lead time" means
  and invalidates the reference.

## Known state, 2026-09-04

`tools/anchor-reference.json` is **provisional**: it covers 2026-09-01..09-03,
the only three days after the blockade ended. Three days is below `MIN_DAYS`,
and the check will say so on every run. **Re-freeze it once CI has collected 14
post-blockade days** (around 2026-09-18) — that is the first thing to do here,
and it is a one-line command plus a commit.

The calibration itself has not shipped. `DelayModel.LIVE_SHRINKAGE = 0.4` and
`MIN_LIVE_SCALE = 1.2` are still what the app uses, and they were not fitted to
this curve.
