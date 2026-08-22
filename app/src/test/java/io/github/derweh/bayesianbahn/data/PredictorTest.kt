package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.HistoricalRun
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
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

    private fun forecast(live: Double?, history: TrainHistory? = history(6)) =
        Predictor().forecast(
            history = history,
            stationEva = "8000013",
            stationName = "Augsburg Hbf",
            trainCategory = "RE",
            plannedTimeMillis = plannedMillis,
            liveDelayMinutes = live,
            today = today,
        )

    // --- the gate itself ---------------------------------------------------

    @Test
    fun `a reported delay is used`() {
        val f = forecast(live = 12.0)
        assertEquals(ForecastSource.EMPIRICAL_LIVE, f.source)
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
        assertEquals(ForecastSource.EMPIRICAL_LIVE, forecast(live = 1.0).source)
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
    fun `the prior fallback still uses a reported delay`() {
        val f = forecast(live = 15.0, history = null)
        assertEquals(ForecastSource.PRIOR, f.source)
        assertNull(f.ignoredLiveDelay)
        assertEquals(15.0, f.distribution.quantile(0.5), 2.0)
    }
}
