package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.abs
import kotlin.random.Random

/**
 * The prefix-summed lookups against the linear scan they replaced.
 *
 * The scan was the app's histogram and interval, and 88% of the time the
 * evaluation spent scoring a day of two-leg journeys. Replacing it is only
 * worth anything if it answers identically, and "identically" has to include
 * the awkward inputs: repeated arrival times (many runs land on the same
 * minute), zero weights (a candidate the model gives no chance), and the two
 * ends of the probability range.
 */
class PointDistributionTest {

    private fun scanCdf(points: List<Pair<Double, Double>>, x: Double): Double {
        val sorted = points.sortedBy { it.first }
        val total = sorted.sumOf { it.second }
        if (sorted.isEmpty()) return Double.NaN
        var acc = 0.0
        for ((value, weight) in sorted) {
            if (value > x) break
            acc += weight
        }
        return acc / total
    }

    private fun scanQuantile(points: List<Pair<Double, Double>>, p: Double): Double {
        val sorted = points.sortedBy { it.first }
        val total = sorted.sumOf { it.second }
        if (sorted.isEmpty()) return Double.NaN
        val target = p * total
        var acc = 0.0
        for ((value, weight) in sorted) {
            acc += weight
            if (acc >= target) return value
        }
        return sorted.last().first
    }

    @Test
    fun `answers exactly what the linear scan answered`() {
        val rng = Random(20260827)
        var checks = 0
        repeat(3000) {
            val n = 1 + rng.nextInt(60)
            val points = List(n) {
                // Repeats on purpose: real arrival times cluster on the minute.
                rng.nextInt(-20, 60).toDouble() to
                    if (rng.nextInt(10) == 0) 0.0 else rng.nextDouble()
            }
            if (points.sumOf { it.second } == 0.0) return@repeat
            val d = PointDistribution(points)
            for (p in listOf(0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)) {
                assertEquals("quantile($p) of $points", scanQuantile(points, p), d.quantile(p), 0.0)
                checks++
            }
            for (x in -25..65) {
                val expected = scanCdf(points, x.toDouble())
                val actual = d.cdf(x.toDouble())
                assertEquals("cdf($x) of $points", expected, actual, 1e-12)
                checks++
            }
        }
        assert(checks > 100_000) { "only $checks comparisons" }
    }

    @Test
    fun `the cdf reaches exactly one and the top quantile is the last point`() {
        // The invariant the separate weight sum could not hold: with the total
        // taken from a different pass, both could land an ulp short, and a
        // quantile on a flat stretch then chose its point on that last bit.
        val rng = Random(7)
        repeat(500) {
            val points = List(1 + rng.nextInt(80)) {
                rng.nextInt(-30, 90).toDouble() to rng.nextDouble()
            }
            val d = PointDistribution(points)
            val top = points.maxOf { it.first }
            assertEquals(1.0, d.cdf(top), 0.0)
            assertEquals(top, d.quantile(1.0), 0.0)
        }
    }

    @Test
    fun `an empty distribution is not a number`() {
        val d = PointDistribution(emptyList())
        assert(d.cdf(0.0).isNaN())
        assert(d.quantile(0.5).isNaN())
    }

    @Test
    fun `every point below x gives one, every point above gives zero`() {
        val d = PointDistribution(listOf(1.0 to 0.5, 3.0 to 0.5))
        assertEquals(0.0, d.cdf(0.0), 0.0)
        assertEquals(0.5, d.cdf(1.0), 0.0)
        assertEquals(1.0, d.cdf(9.0), 0.0)
    }

    @Test
    fun `a zero-weight point is never the answer to a quantile`() {
        // The model gives a candidate no chance; asking for the median must
        // not land on it just because it sorts first.
        val d = PointDistribution(listOf(-5.0 to 0.0, 2.0 to 1.0))
        assertEquals(2.0, d.quantile(0.5), 0.0)
        assertEquals(scanQuantile(listOf(-5.0 to 0.0, 2.0 to 1.0), 0.5), d.quantile(0.5), 0.0)
    }
}
