package io.github.derweh.bayesianbahn.model

import java.time.LocalDate
import java.time.temporal.ChronoUnit
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/** Common interface of every delay distribution the UI can display. */
interface DelayDistribution {
    /** P(delay <= x minutes). */
    fun cdf(x: Double): Double

    /** Delay in minutes at cumulative probability [p]. */
    fun quantile(p: Double): Double
}

/** [StudentT] already provides cdf/quantile; adapt it to the shared interface. */
class StudentTDelay(private val t: StudentT) : DelayDistribution {
    override fun cdf(x: Double) = t.cdf(x)
    override fun quantile(p: Double) = t.quantile(p)
}

/** One historical run of a train past a station. */
data class HistoricalRun(
    val date: LocalDate,
    /** Planned time of day "HH:mm" — distinguishes runs of line-named trains. */
    val plannedTimeOfDay: String,
    /** Final arrival delay in minutes at this station, null if unknown. */
    val arrivalDelay: Int?,
    val departureDelay: Int?,
    /** Delay the same ride had at the previous stop, for live conditioning. */
    val previousStopDelay: Int?,
    val cancelled: Boolean,
)

/**
 * Weighted empirical distribution of arrival delays built from historical
 * runs of one connection. Weights combine:
 *  - recency (exponential decay, half-life [RECENCY_HALF_LIFE_DAYS] days),
 *  - same-weekday boost (delay patterns are weekday dependent),
 *  - optionally a kernel on the live delay at the previous stop, so the
 *    prediction conditions on the state of the approaching train.
 */
class EmpiricalDelay private constructor(
    /** (delay, weight) pairs sorted by delay. */
    private val points: List<Pair<Double, Double>>,
    val sampleSize: Int,
    val effectiveSampleSize: Double,
    val cancelProbability: Double,
    val conditionedOnLive: Boolean,
) : DelayDistribution {

    private val totalWeight = points.sumOf { it.second }

    override fun cdf(x: Double): Double {
        if (points.isEmpty()) return Double.NaN
        var acc = 0.0
        for ((delay, weight) in points) {
            if (delay > x) break
            acc += weight
        }
        return acc / totalWeight
    }

    override fun quantile(p: Double): Double {
        require(p in 0.0..1.0)
        if (points.isEmpty()) return Double.NaN
        val target = p * totalWeight
        var acc = 0.0
        for ((delay, weight) in points) {
            acc += weight
            if (acc >= target) return delay
        }
        return points.last().first
    }

    companion object {
        const val RECENCY_HALF_LIFE_DAYS = 60.0
        const val SAME_WEEKDAY_BOOST = 2.0

        /** Runs whose planned time of day differs more than this are a different connection. */
        const val TIME_OF_DAY_WINDOW_MIN = 20

        /** Minimum effective sample size for a usable distribution. */
        const val MIN_EFFECTIVE_N = 8.0

        fun build(
            runs: List<HistoricalRun>,
            queryTimeOfDay: String,
            queryDate: LocalDate,
            liveDelayAtPreviousStop: Double? = null,
        ): EmpiricalDelay? {
            val relevant = runs.filter {
                timeOfDayDistance(it.plannedTimeOfDay, queryTimeOfDay) <= TIME_OF_DAY_WINDOW_MIN
            }
            if (relevant.isEmpty()) return null
            val cancelRate = relevant.count { it.cancelled }.toDouble() / relevant.size

            val usable = relevant.filter { !it.cancelled && it.delayOrNull() != null }
            if (usable.isEmpty()) return null

            fun baseWeight(run: HistoricalRun): Double {
                val age = ChronoUnit.DAYS.between(run.date, queryDate).coerceAtLeast(0)
                var w = exp(-ln(2.0) / RECENCY_HALF_LIFE_DAYS * age)
                if (run.date.dayOfWeek == queryDate.dayOfWeek) w *= SAME_WEEKDAY_BOOST
                return w
            }

            fun kernel(run: HistoricalRun): Double {
                val live = liveDelayAtPreviousStop ?: return 1.0
                // Unknown previous-stop delay: keep a sliver of weight.
                val prev = run.previousStopDelay?.toDouble() ?: return 0.05
                // Gaussian kernel on the live delay; wider for larger delays
                // because a 40 vs 45 min report means the same thing.
                val bandwidth = maxOf(3.0, 0.3 * abs(live))
                val z = (prev - live) / bandwidth
                return exp(-0.5 * z * z)
            }

            fun buildPoints(conditioned: Boolean): Pair<List<Pair<Double, Double>>, Double> {
                val weighted = usable.map { run ->
                    val w = baseWeight(run) * (if (conditioned) kernel(run) else 1.0)
                    run.delayOrNull()!!.toDouble() to w
                }.sortedBy { it.first }
                val sumW = weighted.sumOf { it.second }
                val sumW2 = weighted.sumOf { it.second * it.second }
                val effN = if (sumW2 > 0) sumW * sumW / sumW2 else 0.0
                return weighted to effN
            }

            // Prefer the live-conditioned distribution, but only when enough
            // genuinely comparable runs exist: effective sample size alone is
            // scale invariant and would accept 40 equally *dissimilar* runs,
            // so additionally require the raw kernel mass — the equivalent
            // number of fully similar runs — to clear the same bar.
            if (liveDelayAtPreviousStop != null) {
                val kernelMass = usable.sumOf { kernel(it) }
                val (points, effN) = buildPoints(conditioned = true)
                if (effN >= MIN_EFFECTIVE_N && kernelMass >= MIN_EFFECTIVE_N) {
                    return EmpiricalDelay(points, usable.size, effN, cancelRate, true)
                }
            }
            val (points, effN) = buildPoints(conditioned = false)
            return EmpiricalDelay(points, usable.size, effN, cancelRate, false)
        }

        private fun HistoricalRun.delayOrNull(): Int? = arrivalDelay ?: departureDelay

        private fun timeOfDayDistance(a: String, b: String): Int {
            val am = toMinutes(a) ?: return Int.MAX_VALUE
            val bm = toMinutes(b) ?: return Int.MAX_VALUE
            val diff = abs(am - bm)
            return minOf(diff, 24 * 60 - diff)
        }

        private fun toMinutes(hhmm: String): Int? {
            val parts = hhmm.split(':')
            if (parts.size != 2) return null
            val h = parts[0].toIntOrNull() ?: return null
            val m = parts[1].toIntOrNull() ?: return null
            return h * 60 + m
        }
    }
}
