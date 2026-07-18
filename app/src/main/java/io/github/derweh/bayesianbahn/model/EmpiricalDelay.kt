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
 * runs of one connection.
 *
 * Without live information the support is the runs' final delays, weighted by
 * recency (exponential decay, half-life [RECENCY_HALF_LIFE_DAYS] days — kept
 * short because timetable changes and construction sites make old runs stale)
 * and a same-weekday boost.
 *
 * With a live delay report the *delta* model is used: each historical run
 * contributes `live + (finalDelay - previousStopDelay)` — its observed
 * last-hop progression shifted onto the live report — mildly sharpened by a
 * Gaussian kernel towards runs whose previous-stop delay resembled the live
 * one. Backtesting on 8 months of IRIS history (12-week walk-forward eval)
 * showed this cuts CRPS ~3.2x versus ignoring live data, while kernel-only
 * reweighting (the previous approach) recovered barely half the gain.
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
        // Half-life chosen by backtest (12-week eval): 30d beat 14/60d on
        // CRPS, 7d clearly worse — schedules drift (construction sites,
        // timetable changes), so memory stays moderately short.
        const val RECENCY_HALF_LIFE_DAYS = 30.0
        const val SAME_WEEKDAY_BOOST = 2.0

        /** Keeps far-away runs contributing residuals instead of vanishing. */
        const val LIVE_KERNEL_FLOOR = 0.15

        /** Runs whose planned time of day differs more than this are a different connection. */
        const val TIME_OF_DAY_WINDOW_MIN = 20

        /** Minimum effective sample size for a usable distribution. */
        const val MIN_EFFECTIVE_N = 8.0

        /** Recency × same-weekday weight, shared with the connection model. */
        fun baseWeight(runDate: LocalDate, queryDate: LocalDate): Double {
            val age = ChronoUnit.DAYS.between(runDate, queryDate).coerceAtLeast(0)
            var w = exp(-ln(2.0) / RECENCY_HALF_LIFE_DAYS * age)
            if (runDate.dayOfWeek == queryDate.dayOfWeek) w *= SAME_WEEKDAY_BOOST
            return w
        }

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

            fun baseWeight(run: HistoricalRun): Double = baseWeight(run.date, queryDate)

            fun effectiveN(points: List<Pair<Double, Double>>): Double {
                val sumW = points.sumOf { it.second }
                val sumW2 = points.sumOf { it.second * it.second }
                return if (sumW2 > 0) sumW * sumW / sumW2 else 0.0
            }

            // Delta model: shift each run's last-hop progression residual
            // (final - previous stop) onto the live report. Only runs with a
            // known previous-stop delay can contribute.
            if (liveDelayAtPreviousStop != null) {
                val withPrev = usable.filter { it.previousStopDelay != null }
                if (withPrev.size >= MIN_EFFECTIVE_N) {
                    val bandwidth = maxOf(3.0, 0.3 * abs(liveDelayAtPreviousStop))
                    val points = withPrev.map { run ->
                        val prev = run.previousStopDelay!!.toDouble()
                        val z = (prev - liveDelayAtPreviousStop) / bandwidth
                        val kernel = LIVE_KERNEL_FLOOR + exp(-0.5 * z * z)
                        val value = liveDelayAtPreviousStop + (run.delayOrNull()!! - prev)
                        value to baseWeight(run) * kernel
                    }.sortedBy { it.first }
                    return EmpiricalDelay(
                        points, usable.size, effectiveN(points), cancelRate, true,
                    )
                }
            }

            val points = usable.map { it.delayOrNull()!!.toDouble() to baseWeight(it) }
                .sortedBy { it.first }
            return EmpiricalDelay(points, usable.size, effectiveN(points), cancelRate, false)
        }

        private fun HistoricalRun.delayOrNull(): Int? = arrivalDelay ?: departureDelay

        fun timeOfDayDistance(a: String, b: String): Int {
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
