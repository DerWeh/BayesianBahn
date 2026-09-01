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

    // --- pooling a train's history with its line's -------------------------

    /**
     * Runs at one slot on one day, so every weight is equal and the effective
     * count is exactly the run count — which is what makes the share below an
     * exact number rather than an approximation.
     */
    private fun flat(count: Int, delay: Int, daysAgo: Long) =
        List(count) {
            HistoricalRun(
                date = today.minusDays(daysAgo),
                plannedTimeOfDay = "17:59",
                arrivalDelay = delay,
                departureDelay = delay,
                previousStopDelay = delay,
                cancelled = false,
            )
        }

    @Test
    fun `the own runs take n over n plus eight of the weight`() {
        // The shrinkage the backtest chose, stated as a number rather than as
        // a switch: eight runs of its own and the train is exactly half its
        // line, two and it is a fifth of it.
        for ((count, expected) in listOf(2 to 0.2, 8 to 0.5, 24 to 0.75)) {
            val dist = EmpiricalDelay.build(
                runs = flat(count, 4, daysAgo = 1),
                queryTimeOfDay = "17:59",
                queryDate = today,
                lineRuns = flat(40, 20, daysAgo = 2),
            )!!
            assertEquals("with $count own runs", expected, dist.ownShare, 1e-6)
            // ...and the weight really lands there: the median sits on the
            // train's own delay only once its own side holds more than half.
            if (count != 8) {
                assertEquals(if (count > 8) 4.0 else 20.0, dist.quantile(0.5), 1e-9)
            }
        }
    }

    @Test
    fun `a line run the train already has is not counted twice`() {
        // A line shard contains every run of the line at the station, this
        // train's included. Counting them on both sides would quietly double
        // the train's own weight and break the share above.
        val own = flat(4, 4, daysAgo = 1)
        val withDuplicates = EmpiricalDelay.build(
            runs = own, queryTimeOfDay = "17:59", queryDate = today,
            lineRuns = own + flat(40, 20, daysAgo = 2),
        )!!
        val without = EmpiricalDelay.build(
            runs = own, queryTimeOfDay = "17:59", queryDate = today,
            lineRuns = flat(40, 20, daysAgo = 2),
        )!!
        assertEquals(without.ownShare, withDuplicates.ownShare, 1e-9)
        assertEquals(without.quantile(0.5), withDuplicates.quantile(0.5), 1e-9)
        assertEquals(without.cdf(4.0), withDuplicates.cdf(4.0), 1e-9)
    }

    @Test
    fun `a train with no runs of its own answers entirely from its line`() {
        val dist = EmpiricalDelay.build(
            runs = emptyList(), queryTimeOfDay = "17:59", queryDate = today,
            lineRuns = flat(40, 20, daysAgo = 1),
        )!!
        assertEquals(0.0, dist.ownShare, 1e-9)
        assertEquals(20.0, dist.quantile(0.5), 1e-9)
    }

    @Test
    fun `without a line shard nothing about the old model changes`() {
        val own = flat(20, 4, daysAgo = 1)
        val alone = EmpiricalDelay.build(own, "17:59", today)!!
        assertEquals(1.0, alone.ownShare, 1e-9)
        assertEquals(20, alone.sampleSize)
        assertEquals(4.0, alone.quantile(0.5), 1e-9)
    }

    @Test
    fun `runs outside the time of day window are dropped on both sides`() {
        val dist = EmpiricalDelay.build(
            runs = flat(4, 4, daysAgo = 1),
            queryTimeOfDay = "17:59",
            queryDate = today,
            lineRuns = flat(40, 20, daysAgo = 2).map {
                it.copy(plannedTimeOfDay = "09:00")
            },
        )!!
        assertEquals("a line whose runs are all at another hour adds nothing",
            1.0, dist.ownShare, 1e-9)
    }

    @Test
    fun `the cancel rate blends the two sides rather than pooling them`() {
        // Five hundred of the line's runs must not decide the cancel
        // probability of a train with four of its own.
        val own = flat(4, 4, daysAgo = 1)
        val line = flat(40, 20, daysAgo = 2).map { it.copy(cancelled = true) }
        val dist = EmpiricalDelay.build(own, "17:59", today, lineRuns = line)!!
        // own rate 0, line rate 1, share 4/12.
        assertEquals(1.0 - 4.0 / 12.0, dist.cancelProbability, 1e-6)
    }

    @Test
    fun `a planned time that is not HH mm is still no match`() {
        assertEquals(-1, EmpiricalDelay.minutesOfDay("?"))
        assertEquals(-1, EmpiricalDelay.minutesOfDay("1759"))
        assertEquals(-1, EmpiricalDelay.minutesOfDay("aa:bb"))
        assertEquals(-1, EmpiricalDelay.minutesOfDay("25:00"))
        assertEquals(17 * 60 + 59, EmpiricalDelay.minutesOfDay("17:59"))
        assertEquals(Int.MAX_VALUE, EmpiricalDelay.timeOfDayDistance("?", "17:59"))
    }
}
