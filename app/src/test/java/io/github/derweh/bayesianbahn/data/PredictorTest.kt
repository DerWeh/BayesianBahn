package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.HistoricalRun
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * Predictor had no unit tests: it was reached only through the offline harness
 * and the end-to-end smoke test, neither of which runs on a normal build. The
 * rule these cover — that DB's live number is evidence only when it reports a
 * delay — was worth 0.53 min of CRPS on trains with history and moved the
 * stated 80% interval from covering 55% of arrivals to covering 82%.
 */
class PredictorTest {

    private val zone = ZoneId.of("Europe/Berlin")
    private val today = LocalDate.of(2026, 8, 20)
    private val plannedMillis =
        ZonedDateTime.of(today, java.time.LocalTime.of(8, 0), zone).toInstant().toEpochMilli()

    /** A station's worth of runs, all of them this late, at the same time of day. */
    private fun history(delayMinutes: Int, runs: Int = 40) = TrainHistory(
        trainName = "RE 1",
        trainType = "RE",
        stations = mapOf(
            "Augsburg Hbf" to StationHistory(
                eva = "8000013",
                runs = (1..runs).map {
                    HistoricalRun(
                        date = today.minusDays(it.toLong()),
                        plannedTimeOfDay = "08:00",
                        arrivalDelay = delayMinutes,
                        departureDelay = delayMinutes,
                        previousStopDelay = delayMinutes,
                        cancelled = false,
                    )
                },
            ),
        ),
    )

    private fun forecast(
        live: Double?,
        history: TrainHistory? = history(6),
        line: TrainHistory? = null,
        lineFetches: IntArray = IntArray(1),
    ) = runBlocking {
            Predictor().forecast(
                history = history,
                stationEva = "8000013",
                stationName = "Augsburg Hbf",
                trainCategory = "RE",
                plannedTimeMillis = plannedMillis,
                liveDelayMinutes = live,
                today = today,
                lineHistory = { lineFetches[0]++; line },
            )
        }

    // --- the gate itself ---------------------------------------------------

    @Test
    fun `a reported delay is used`() {
        val f = forecast(live = 12.0)
        assertEquals(ForecastSource.LIVE_ANCHORED, f.source)
        assertNull("a report we acted on is not an ignored one", f.ignoredLiveDelay)
    }

    @Test
    fun `a report of on time is not evidence`() {
        val f = forecast(live = 0.0)
        assertEquals(ForecastSource.EMPIRICAL, f.source)
        assertEquals(0.0, f.ignoredLiveDelay!!, 1e-9)
    }

    @Test
    fun `a report of running early is not evidence either`() {
        // Trains DB called early averaged 1.4 minutes late over 2026-08-17..19.
        val f = forecast(live = -2.0)
        assertEquals(ForecastSource.EMPIRICAL, f.source)
        assertEquals(-2.0, f.ignoredLiveDelay!!, 1e-9)
    }

    @Test
    fun `a delay below a whole minute is not evidence`() {
        assertEquals(ForecastSource.EMPIRICAL, forecast(live = 0.4).source)
    }

    @Test
    fun `a delay of exactly one minute is evidence`() {
        assertEquals(ForecastSource.LIVE_ANCHORED, forecast(live = 1.0).source)
    }

    @Test
    fun `no report at all is not an ignored report`() {
        val f = forecast(live = null)
        assertEquals(ForecastSource.EMPIRICAL, f.source)
        assertNull(f.ignoredLiveDelay)
    }

    // --- what the gate is for ----------------------------------------------

    @Test
    fun `an ignored on-time report leaves the history's own answer`() {
        val ignored = forecast(live = 0.0).distribution.quantile(0.5)
        val none = forecast(live = null).distribution.quantile(0.5)
        assertEquals(none, ignored, 1e-9)
        assertTrue("a train that is always 6 late should not be predicted on time",
            ignored > 4.0)
    }

    @Test
    fun `anchoring on a zero would have pulled the forecast to on time`() {
        // The behaviour before the gate, reproduced by passing a delay the gate
        // does let through: the prediction follows the live number.
        val anchored = forecast(live = 20.0).distribution.quantile(0.5)
        assertTrue("a live report should still move the forecast", anchored > 10.0)
        assertNotEquals(forecast(live = null).distribution.quantile(0.5), anchored)
    }

    @Test
    fun `the interval stays honest when an on-time report is ignored`() {
        val d = forecast(live = 0.0).distribution
        // Anchored at zero with the shrunk live spread this range collapsed
        // around 0; from history it has to sit around the train's real delay.
        assertTrue(d.quantile(0.9) - d.quantile(0.1) >= 0.0)
        assertTrue(d.quantile(0.9) > 3.0)
    }

    // --- the no-history path -----------------------------------------------

    @Test
    fun `the prior fallback ignores an on-time report too`() {
        val f = forecast(live = 0.0, history = null)
        assertEquals(ForecastSource.PRIOR, f.source)
        assertEquals(0.0, f.ignoredLiveDelay!!, 1e-9)
        assertTrue("the prior for a regional train is not zero",
            f.distribution.quantile(0.5) > 0.0)
    }

    @Test
    fun `a live answer still carries the cancellation rate from history`() {
        // The delay comes from the report; whether the train runs at all does
        // not, and cannot. Losing this put "n/a" on the cancellation tile for
        // every train DB had said something about — which is every train
        // anybody was looking at.
        val f = forecast(live = 9.0, history = history(20))
        assertEquals(ForecastSource.LIVE_ANCHORED, f.source)
        assertNotNull("the screen shows n/a when this is null", f.cancelProbability)
        // Forty runs, none cancelled: zero is an answer, "not available" is not.
        assertEquals(0.0, f.cancelProbability!!, 1e-9)
        assertTrue("the run count should survive too", f.runCount > 0)
        assertEquals(9.0, f.distribution.quantile(0.5), 2.0)
    }

    @Test
    fun `a reported delay answers on its own, with no history at all`() {
        // The prior fallback is not reached: a train DB has said something
        // about is answered from the report and the measured error of such
        // reports, which beats anything a category prior contributes.
        val f = forecast(live = 15.0, history = null)
        assertEquals(ForecastSource.LIVE_ANCHORED, f.source)
        assertNull(f.ignoredLiveDelay)
        assertEquals(15.0, f.distribution.quantile(0.5), 2.0)
        assertEquals(0, f.runCount)
    }

    // --- the line fallback -------------------------------------------------

    @Test
    fun `the line carries the answer when the run number is too new to`() {
        // Three runs is well under the eight pseudo-runs the shrinkage gives
        // the line, so the line holds most of the weight — and this number
        // alone would have gone to the prior, as a quarter of arrivals did.
        val f = forecast(live = null, history = history(3, runs = 3), line = history(9))
        assertEquals(ForecastSource.EMPIRICAL_LINE, f.source)
        assertEquals("RE 1", f.lineName)
        assertEquals("mostly the line's delays, not the three runs of the number",
            9.0, f.distribution.quantile(0.5), 1e-9)
        // ...but the train's own three runs are in there, not discarded.
        assertTrue(f.distribution.cdf(3.0) > 0.0)
    }

    @Test
    fun `a train with plenty of history never pays for the line shard`() {
        // Forty runs is far past the ceiling where pooling stops being worth
        // anything, so the shard is not asked for: roughly seven predictions
        // in eight stop here.
        val fetches = IntArray(1)
        val f = forecast(live = null, line = history(9), lineFetches = fetches)
        assertEquals(ForecastSource.EMPIRICAL, f.source)
        assertEquals(0, fetches[0])
        assertNull(f.lineName)
        assertEquals("and its own answer is untouched", 6.0,
            f.distribution.quantile(0.5), 1e-9)
    }

    @Test
    fun `a train just short of the ceiling does consult its line`() {
        val fetches = IntArray(1)
        val f = forecast(live = null, history = history(6, runs = 9),
            line = history(9), lineFetches = fetches)
        assertEquals(1, fetches[0])
        // Nine of its own runs against eight pseudo-runs: the train's own
        // history holds just over half, so the screens still call it its own —
        // but the line is named, because it is a third of the answer and
        // "past runs of this train" would not be true of all of it.
        assertEquals(ForecastSource.EMPIRICAL, f.source)
        assertEquals("RE 1", f.lineName)
        assertTrue(f.ownShare > 0.5 && f.ownShare < 1.0)
    }

    @Test
    fun `a forecast that never saw a line says so by naming none`() {
        val f = forecast(live = null, history = history(6, runs = 9), line = null)
        assertNull(f.lineName)
        assertEquals(1.0, f.ownShare, 1e-9)
    }

    @Test
    fun `the prior still answers when the two together are too thin`() {
        // Pooling removes the switch, not the floor.
        val f = forecast(live = null, history = null, line = history(9, runs = 2))
        assertEquals(ForecastSource.PRIOR, f.source)
        assertNull(f.lineName)
    }

    @Test
    fun `a reported delay outranks the line's history too`() {
        // This used to be EMPIRICAL_LINE_LIVE, the line's runs shifted onto the
        // report. The report plus its own residual is the better answer, and it
        // does not need the line shard fetched to give it.
        val f = forecast(live = 12.0, history = null, line = history(3))
        assertEquals(ForecastSource.LIVE_ANCHORED, f.source)
        assertEquals(12.0, f.distribution.quantile(0.5), 2.0)
    }

    @Test
    fun `an ignored on-time report is still reported from the line path`() {
        val f = forecast(live = 0.0, history = null, line = history(6))
        assertEquals(ForecastSource.EMPIRICAL_LINE, f.source)
        assertEquals(0.0, f.ignoredLiveDelay!!, 1e-9)
    }
}
