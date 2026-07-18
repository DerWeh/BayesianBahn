package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate

class EmpiricalDelayTest {

    private val today: LocalDate = LocalDate.of(2026, 7, 17)

    private fun run(
        daysAgo: Long,
        arrDelay: Int?,
        prev: Int? = arrDelay,
        cancelled: Boolean = false,
        timeOfDay: String = "17:59",
    ) = HistoricalRun(
        date = today.minusDays(daysAgo),
        plannedTimeOfDay = timeOfDay,
        arrivalDelay = arrDelay,
        departureDelay = arrDelay,
        previousStopDelay = prev,
        cancelled = cancelled,
    )

    @Test
    fun `quantiles of uniform delays`() {
        val runs = (1L..100L).map { run(it, (it % 10).toInt()) }
        val dist = EmpiricalDelay.build(runs, "17:59", today)!!
        assertTrue(dist.quantile(0.5) in 3.0..6.0)
        assertEquals(100, dist.sampleSize)
        assertTrue(dist.cdf(9.0) >= 0.99)
        assertEquals(0.0, dist.cancelProbability, 1e-9)
    }

    @Test
    fun `runs at other times of day are excluded`() {
        val runs = (1L..20L).map { run(it, 3) } +
            (1L..20L).map { run(it, 60, timeOfDay = "06:00") }
        val dist = EmpiricalDelay.build(runs, "17:59", today)!!
        assertEquals(20, dist.sampleSize)
        assertTrue(dist.quantile(0.9) < 10)
    }

    @Test
    fun `no matching runs returns null`() {
        val runs = (1L..20L).map { run(it, 3, timeOfDay = "06:00") }
        assertNull(EmpiricalDelay.build(runs, "17:59", today))
    }

    @Test
    fun `live delta model anchors the distribution at the live delay`() {
        // 50 on-time runs, 25 runs that were ~20 min late at the previous stop
        // and stayed ~20 late. With a +20 live report the delta model must
        // predict around 20, not around the marginal median of ~1.
        val runs = (1L..50L).map { run(it, 1, prev = 0) } +
            (51L..75L).map { run(it - 50, 20 + (it % 3).toInt(), prev = 20) }
        val unconditioned = EmpiricalDelay.build(runs, "17:59", today)!!
        val conditioned = EmpiricalDelay.build(runs, "17:59", today, liveDelayAtPreviousStop = 20.0)!!
        assertTrue(conditioned.conditionedOnLive)
        assertFalse(unconditioned.conditionedOnLive)
        assertTrue(conditioned.quantile(0.5) >= 19.0)
        assertTrue(unconditioned.quantile(0.5) <= 5.0)
    }

    @Test
    fun `delta model extrapolates residuals to unseen live delays`() {
        // All runs gained +1 min on the last hop; a train reported 90 min
        // late at the previous stop is predicted to stay ~91 min late even
        // though no historical run was ever that late.
        val runs = (1L..40L).map { run(it, 2, prev = 1) }
        val dist = EmpiricalDelay.build(runs, "17:59", today, liveDelayAtPreviousStop = 90.0)!!
        assertTrue(dist.conditionedOnLive)
        assertEquals(91.0, dist.quantile(0.5), 0.5)
    }

    @Test
    fun `falls back to marginal when previous-stop delays are unknown`() {
        val runs = (1L..40L).map { run(it, 2, prev = null) }
        val dist = EmpiricalDelay.build(runs, "17:59", today, liveDelayAtPreviousStop = 10.0)!!
        assertFalse(dist.conditionedOnLive)
        assertTrue(dist.quantile(0.5) <= 3.0)
    }

    @Test
    fun `cancelled runs feed cancel probability but not delays`() {
        val runs = (1L..30L).map { run(it, 4) } + (31L..40L).map { run(it - 30, null, cancelled = true) }
        val dist = EmpiricalDelay.build(runs, "17:59", today)!!
        assertEquals(0.25, dist.cancelProbability, 1e-9)
        assertEquals(30, dist.sampleSize)
    }
}
