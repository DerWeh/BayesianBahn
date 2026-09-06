package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.system.measureNanoTime

/**
 * What the calibrated anchor costs, measured rather than assumed.
 *
 * Two things changed that could have been expensive. The feeder's distribution
 * is now a closed form instead of a walk over historical points, which should
 * be faster. And a connecting train's departure is a distribution instead of a
 * comparison, which means [ConnectionModel.propagate] no longer stops walking
 * candidates the moment one is certain — the reason [ConnectionModel.MASS_FLOOR]
 * exists.
 *
 * The budgets below are deliberately loose. They are regression tripwires for
 * an order of magnitude, not benchmarks: this runs on whatever CI happens to
 * give it, and a tight bound would fail for reasons that have nothing to do
 * with the code. The printed numbers are the useful part.
 */
class AnchoredDelayProfileTest {

    private val t0 = 1_700_000_000_000L
    private val minute = 60_000L

    private fun candidates(n: Int, runsEach: Int, live: Double?) =
        (0 until n).map { k ->
            ConnectionModel.Candidate(
                id = "C$k",
                label = "C$k",
                plannedDepartureMillis = t0 + (10L + k * 12) * minute,
                plannedArrivalMillis = t0 + (40L + k * 12) * minute,
                runs = List(runsEach) {
                    ConnectionModel.JointRun(
                        departureDelay = (it % 7).toDouble(),
                        arrivalDelay = (it % 9).toDouble(),
                        weight = 1.0,
                    )
                },
                liveDepartureDelay = live,
                cancelledLive = false,
                cancelRate = 0.02,
            )
        }

    private inline fun bench(warmup: Int, runs: Int, body: () -> Unit): Double {
        repeat(warmup) { body() }
        val nanos = measureNanoTime { repeat(runs) { body() } }
        return nanos.toDouble() / runs
    }

    @Test
    fun `the anchored distribution is cheap to build and to ask`() {
        var sink = 0.0
        val build = bench(50_000, 200_000) {
            sink += AnchoredDelay(6.0, 45.0, ResidualShape.ARRIVAL).median
        }
        val d = AnchoredDelay(6.0, 45.0, ResidualShape.ARRIVAL)
        val quantile = bench(50_000, 500_000) { sink += d.quantile(0.37) }
        val cdf = bench(50_000, 500_000) { sink += d.cdf(11.0) }
        println(
            "AnchoredDelay: build %.0f ns, quantile %.0f ns, cdf %.0f ns"
                .format(build, quantile, cdf),
        )
        assertTrue("unused: $sink", sink != Double.NaN)
        assertTrue("building should be well under a microsecond: $build ns", build < 2_000)
        assertTrue("quantile should be tens of nanoseconds: $quantile ns", quantile < 1_000)
        assertTrue("cdf should be tens of nanoseconds: $cdf ns", cdf < 1_000)
    }

    /**
     * The anchored feeder against the history-backed one it replaces, at the
     * size the app actually asks for: a screen reads three quantiles and a
     * histogram's worth of cdfs.
     */
    @Test
    fun `asking the anchored feeder beats walking historical points`() {
        val points = PointDistribution((0 until 400).map { (it % 40).toDouble() to 1.0 })
        val anchored = AnchoredDelay(6.0, 45.0, ResidualShape.ARRIVAL)
        var sink = 0.0
        val empirical = bench(20_000, 200_000) { sink += points.quantile(0.9) + points.cdf(5.0) }
        val closed = bench(20_000, 200_000) { sink += anchored.quantile(0.9) + anchored.cdf(5.0) }
        println("one quantile + one cdf: empirical %.0f ns, anchored %.0f ns".format(empirical, closed))
        assertTrue("unused: $sink", sink != Double.NaN)
        assertTrue(
            "the closed form should not be slower: $closed vs $empirical ns",
            closed <= empirical * 1.5,
        )
    }

    /**
     * The whole change screen, at a realistic size: six candidates with thirty
     * runs each, all of them carrying a live departure report — the case where
     * the walk no longer breaks early.
     */
    @Test
    fun `a change with every candidate reported stays affordable`() {
        val feeder = AnchoredDelay(6.0, 45.0, ResidualShape.ARRIVAL)
        val reported = candidates(6, 30, live = 4.0)
        val silent = candidates(6, 30, live = null)
        var sink = 0.0
        val withReports = bench(200, 2_000) {
            sink += ConnectionModel.propagate(feeder, t0, 5, reported, t0)!!.missProbability
        }
        val withHistory = bench(200, 2_000) {
            sink += ConnectionModel.propagate(feeder, t0, 5, silent, t0)!!.missProbability
        }
        println(
            "propagate, 6 candidates x 30 runs: reported %.0f us, history-only %.0f us"
                .format(withReports / 1000, withHistory / 1000),
        )
        assertTrue("unused: $sink", sink != Double.NaN)
        // A screen builds a handful of these. Ten milliseconds each would be
        // visible; anything near a millisecond is not.
        assertTrue("a change should cost well under a millisecond: ${withReports / 1000} us",
            withReports < 1_000_000)
    }

    /**
     * The floor's whole purpose: bound how many points survive when nothing
     * stops the walk early. Without it every later candidate contributes its
     * whole run list at a vanishing weight.
     */
    @Test
    fun `the mass floor keeps the point list from growing with the candidates`() {
        val feeder = AnchoredDelay(2.0, 30.0, ResidualShape.ARRIVAL)
        // All reported comfortably late, so each is near-certain and the ones
        // after it are the tail the floor is meant to cut.
        val many = candidates(8, 40, live = 40.0)
        val result = ConnectionModel.propagate(feeder, t0, 5, many, t0)!!
        val carried = result.candidates.count { it.boardProbability > ConnectionModel.MASS_FLOOR }
        val tail = result.candidates.drop(2).sumOf { it.boardProbability }
        println("8 near-certain candidates: $carried carry any real mass, tail %.2e".format(tail))
        assertTrue("the walk should not credit every candidate: $carried", carried < many.size)
        assertTrue("everything past the second should be noise: $tail", tail < 1e-3)
        assertTrue(
            "board probabilities must still sum to at most one: ${result.candidates.sumOf { it.boardProbability }}",
            result.candidates.sumOf { it.boardProbability } <= 1.0 + 1e-9,
        )
    }

    /**
     * The property that matters more than any single timing: the cost must not
     * run away with the candidate list now that a near-certain train no longer
     * ends the walk. Twice the candidates should cost about twice, not more.
     */
    @Test
    fun `cost grows with the candidate list and no faster`() {
        val feeder = AnchoredDelay(6.0, 45.0, ResidualShape.ARRIVAL)
        val six = candidates(6, 30, live = 4.0)
        val twelve = candidates(12, 30, live = 4.0)
        var sink = 0.0
        val a = bench(200, 1_500) {
            sink += ConnectionModel.propagate(feeder, t0, 5, six, t0)!!.missProbability
        }
        val b = bench(200, 1_500) {
            sink += ConnectionModel.propagate(feeder, t0, 5, twelve, t0)!!.missProbability
        }
        println("propagate: 6 candidates %.0f us, 12 candidates %.0f us (x%.2f)"
            .format(a / 1000, b / 1000, b / a))
        assertTrue("unused: $sink", sink != Double.NaN)
        assertTrue("doubling the candidates more than tripled the cost: x${b / a}", b < a * 3)
    }
}
