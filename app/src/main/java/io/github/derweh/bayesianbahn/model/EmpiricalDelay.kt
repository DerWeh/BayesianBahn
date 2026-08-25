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
 *
 * That result holds where a live report carries information, which is not the
 * same as where one exists. Forward evaluation against DB (2026-08-17..19, see
 * README) found the opposite sign wherever DB had not actually reported a
 * delay: it calls a train on time for 99% of stops three hours out because it
 * has nothing to say yet, and 31% of those arrive more than two minutes late.
 * Deciding which reports to believe is therefore [Predictor]'s job, not this
 * one's — a report that is not evidence arrives here as null, the same as no
 * report at all.
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
        // Half-life chosen by backtest. Re-measured over eight weeks in August
        // 2026, the curve is one-sided: 7 days costs 0.071 minutes of CRPS
        // against this, 14 days 0.024, while 60 days is 0.001 *better* and no
        // decay at all only 0.008 worse. So the number is not "memory stays
        // short because schedules drift", as this comment used to claim — it
        // is "do not make it short". Anything from a month to forever is the
        // same model; 30 is kept because it is what the published figures were
        // measured on.
        const val RECENCY_HALF_LIFE_DAYS = 30.0

        // Worth 0.001 minutes of CRPS against not having it, on the same
        // eight weeks — which is to say nothing. Kept only because removing it
        // would change every published number for no gain. The weekend effect
        // it is reaching for is real: the same train is 0.78 minutes less
        // delayed at the weekend. But the runs that share a day type are a
        // minority for a weekend query and a majority for a working-day one,
        // so a single multiplier helps one and hurts the other by about the
        // same amount, and giving the group a fixed share of the weight
        // instead helps only where the live report is believed — which is 13%
        // of predictions, too few to pay for the 87% it costs.
        const val SAME_WEEKDAY_BOOST = 2.0

        /** Keeps far-away runs contributing residuals instead of vanishing. */
        const val LIVE_KERNEL_FLOOR = 0.15

        /**
         * Runs whose planned time of day differs more than this are a different
         * connection.
         *
         * It binds for 8% of predictions — inside one train number the planned
         * time hardly moves, so nearly every run is in the window already — and
         * where it does bind, widening it from 5 minutes to 3 hours moves CRPS
         * by less than 0.003 either way. It is a guard against a line-numbered
         * S-Bahn pooling its whole day, not a source of information.
         */
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
