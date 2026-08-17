package io.github.derweh.bayesianbahn

import io.github.derweh.bayesianbahn.model.DelayDistribution
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.Instant
import java.time.ZoneId
import kotlin.math.abs

/**
 * The harness deliberately reimplements nothing of the model — but it does add
 * two pieces of arithmetic of its own, and both are the silent kind. A wrong
 * CRPS would rank model changes wrongly forever, and a wrong clock conversion
 * would shift every prediction by the German UTC offset while still producing
 * plausible-looking numbers.
 */
class ForecastHarnessTest {

    /** All mass at [at]: the degenerate case where CRPS has a closed form. */
    private fun pointMass(at: Double) = object : DelayDistribution {
        override fun cdf(x: Double) = if (x >= at) 1.0 else 0.0
        override fun quantile(p: Double) = at
    }

    /** Uniform on [lo, hi]; CRPS against a point inside it is analytic. */
    private fun uniform(lo: Double, hi: Double) = object : DelayDistribution {
        override fun cdf(x: Double) = ((x - lo) / (hi - lo)).coerceIn(0.0, 1.0)
        override fun quantile(p: Double) = lo + p * (hi - lo)
    }

    @Test
    fun `a point forecast that is right scores zero`() {
        assertEquals(0.0, ForecastHarness.crps(pointMass(5.0), 5.0), 1e-9)
    }

    @Test
    fun `a point forecast scores its absolute error`() {
        // CRPS of a degenerate distribution reduces to |y - m|.
        assertEquals(7.0, ForecastHarness.crps(pointMass(3.0), 10.0), 1e-9)
        assertEquals(7.0, ForecastHarness.crps(pointMass(10.0), 3.0), 1e-9)
    }

    @Test
    fun `a sharper forecast beats a vaguer one when both are centred`() {
        val tight = ForecastHarness.crps(uniform(-2.0, 2.0), 0.0)
        val loose = ForecastHarness.crps(uniform(-20.0, 20.0), 0.0)
        assertTrue("sharper must score lower: $tight vs $loose", tight < loose)
    }

    @Test
    fun `spreading probability beats being confidently wrong`() {
        val hedged = ForecastHarness.crps(uniform(0.0, 20.0), 18.0)
        val confident = ForecastHarness.crps(pointMass(0.0), 18.0)
        assertTrue("$hedged should beat $confident", hedged < confident)
    }

    @Test
    fun `crps is never negative`() {
        for (y in listOf(-30.0, 0.0, 4.0, 120.0)) {
            assertTrue(ForecastHarness.crps(uniform(-5.0, 30.0), y) >= 0.0)
        }
    }

    @Test
    fun `the uniform case matches the closed form`() {
        // For U(lo, hi) with y inside, CRPS integrates to
        // [(y-lo)^3 + (hi-y)^3] / 3(hi-lo)^2.
        //
        // The tolerance is about this toy distribution, not about the harness:
        // a continuous CDF sampled on whole minutes carries rectangle-rule
        // error. Real forecasts are step functions that jump at integer
        // minutes, where the same sum is exact — which is why the grid is one
        // minute wide in the first place.
        val lo = 0.0
        val hi = 10.0
        val y = 4.0
        val span = hi - lo
        val exact = ((y - lo) * (y - lo) * (y - lo) + (hi - y) * (hi - y) * (hi - y)) /
            (3 * span * span)
        assertEquals(exact, ForecastHarness.crps(uniform(lo, hi), y), 0.15)
    }

    @Test
    fun `the grid is exact for a step distribution that jumps on whole minutes`() {
        // The real case: an empirical CDF over integer-minute delays.
        val steps = object : DelayDistribution {
            override fun cdf(x: Double) = when {
                x < 2.0 -> 0.0
                x < 5.0 -> 0.5
                else -> 1.0
            }
            override fun quantile(p: Double) = if (p <= 0.5) 2.0 else 5.0
        }
        // y = 5: (0-0)^2 below 2, (0.5)^2 over [2,5) = 3 * 0.25, 0 above.
        assertEquals(0.75, ForecastHarness.crps(steps, 5.0), 1e-9)
    }

    @Test
    fun `wall clock minutes become the right instant in summer`() {
        // 2026-08-17 18:00 German time is 16:00 UTC.
        val minutes = (Instant.parse("2026-08-17T18:00:00Z").epochSecond / 60).toInt()
        val millis = ForecastHarness.wallMinutesToMillis(minutes)
        val utc = Instant.ofEpochMilli(millis).atZone(ZoneId.of("UTC"))
        assertEquals(16, utc.hour)
    }

    @Test
    fun `wall clock minutes become the right instant in winter`() {
        val minutes = (Instant.parse("2026-01-17T18:00:00Z").epochSecond / 60).toInt()
        val utc = Instant.ofEpochMilli(ForecastHarness.wallMinutesToMillis(minutes))
            .atZone(ZoneId.of("UTC"))
        assertEquals("one hour offset outside DST", 17, utc.hour)
    }

    @Test
    fun `the local time the model sees is the one the timetable says`() {
        val minutes = (Instant.parse("2026-08-17T18:00:00Z").epochSecond / 60).toInt()
        val berlin = Instant.ofEpochMilli(ForecastHarness.wallMinutesToMillis(minutes))
            .atZone(ZoneId.of("Europe/Berlin"))
        assertEquals(18, berlin.hour)
        assertTrue(abs(berlin.minute) == 0)
    }
}
