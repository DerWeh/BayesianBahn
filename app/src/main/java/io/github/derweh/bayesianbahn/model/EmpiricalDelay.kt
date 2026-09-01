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
) {
    /**
     * [plannedTimeOfDay] as minutes past midnight, or -1 if it is not "HH:mm".
     *
     * Parsed here, once per run when the shard is read, rather than in the
     * time-of-day filter that every forecast runs over every run. A line's
     * shard at a busy station holds thousands of runs and the filter is the
     * only thing that touches all of them: measured over the 8,505 runs S41
     * has at Berlin Hermannstraße, splitting the strings in the filter instead
     * costs 0.640 ms a forecast against 0.217 — see ForecastLatencyTest.
     */
    @JvmField
    val plannedMinutes: Int = EmpiricalDelay.minutesOfDay(plannedTimeOfDay)
}

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
    /**
     * How much of the weight the train's own runs hold, 0 when the answer is
     * entirely its line's and 1 when the line contributed nothing. The screens
     * use it to decide whose history they are describing.
     */
    val ownShare: Double = 1.0,
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

        /**
         * Pseudo-runs in the shrinkage towards the line: a train's own runs
         * take `n / (n + this)` of the weight, its line's the rest.
         *
         * Chosen by `pipeline/backtest_fallback.py` over 781,000 archive
         * events. A *switch* between the two histories is a step function
         * fitted to a curve: pooling half and half wins by 0.119 min of CRPS
         * where the number has almost nothing and loses 0.032 across the 87% of
         * predictions where it has plenty, and no fixed weighting is right at
         * both ends. This one is — 0.105 better than the line alone on the
         * fallback population (95% 0.094..0.116) and 0.002 better than the
         * number alone everywhere else (95% 0.002..0.003).
         */
        const val LINE_PSEUDO_RUNS = 8.0

        /**
         * Above this many effective runs of its own, a train ignores its line.
         *
         * Not a modelling constant but a bandwidth one: the shrinkage has
         * converged by here — from 32 effective runs up, pooling is worth a
         * thousandth of a minute — so stopping costs 0.002 and saves fetching a
         * line shard for roughly seven predictions in eight.
         */
        const val LINE_CEILING_N = 32.0

        /** Recency × same-weekday weight, shared with the connection model. */
        fun baseWeight(runDate: LocalDate, queryDate: LocalDate): Double {
            val age = ChronoUnit.DAYS.between(runDate, queryDate).coerceAtLeast(0)
            var w = exp(-ln(2.0) / RECENCY_HALF_LIFE_DAYS * age)
            if (runDate.dayOfWeek == queryDate.dayOfWeek) w *= SAME_WEEKDAY_BOOST
            return w
        }

        /**
         * The distribution for one connection, optionally shrunk towards its
         * line.
         *
         * [lineRuns] are the runs of the same line at the same station, from
         * the line-keyed shard. They are *not* an alternative to [runs]: both
         * enter one distribution, with the train's own runs taking
         * `n / (n + LINE_PSEUDO_RUNS)` of the weight, so a run number with
         * nothing behind it answers almost entirely from its line and one with
         * months of history barely notices the line is there. A line shard
         * already contains the train's own runs, so those are dropped from
         * [lineRuns] rather than counted twice.
         */
        fun build(
            runs: List<HistoricalRun>,
            queryTimeOfDay: String,
            queryDate: LocalDate,
            liveDelayAtPreviousStop: Double? = null,
            lineRuns: List<HistoricalRun> = emptyList(),
        ): EmpiricalDelay? {
            val queryMinutes = minutesOfDay(queryTimeOfDay)
            fun inWindow(run: HistoricalRun) =
                timeOfDayDistance(run.plannedMinutes, queryMinutes) <= TIME_OF_DAY_WINDOW_MIN

            val own = runs.filter(::inWindow)
            val fromLine = if (lineRuns.isEmpty()) emptyList() else {
                // (date, planned time) is the same identity mergeHistories uses
                // to overlay the recent tier onto the base.
                val mine = own.mapTo(HashSet()) { it.date to it.plannedMinutes }
                lineRuns.filter { inWindow(it) && (it.date to it.plannedMinutes) !in mine }
            }
            if (own.isEmpty() && fromLine.isEmpty()) return null

            // How much of the weight the train's own runs keep. Their effective
            // count, not their raw one: fifty runs that are all but one stale
            // are not fifty runs, and the shrinkage has to see that.
            val ownEffective = effectiveN(own.map { baseWeight(it.date, queryDate) })
            val ownShare = when {
                fromLine.isEmpty() -> 1.0
                own.isEmpty() -> 0.0
                else -> ownEffective / (ownEffective + LINE_PSEUDO_RUNS)
            }
            val ownScale = groupScale(own, queryDate, ownShare)
            val lineScale = groupScale(fromLine, queryDate, 1.0 - ownShare)
            fun weight(run: HistoricalRun, mine: Boolean): Double =
                baseWeight(run.date, queryDate) * (if (mine) ownScale else lineScale)

            // Each side's own rate, blended by the same share. Not one rate
            // over the pooled runs: that would let five hundred of the line's
            // runs decide the cancel probability of a train with three of its
            // own, which is the mistake the whole weighting exists to avoid.
            val cancelRate = blend(cancelRate(own), cancelRate(fromLine), ownShare)

            val usable = own.filter { !it.cancelled && it.delayOrNull() != null }
            val usableLine = fromLine.filter { !it.cancelled && it.delayOrNull() != null }
            if (usable.isEmpty() && usableLine.isEmpty()) return null
            val sampleSize = if (ownShare >= 0.5) usable.size else usableLine.size

            // Delta model: shift each run's last-hop progression residual
            // (final - previous stop) onto the live report. Only runs with a
            // known previous-stop delay can contribute.
            if (liveDelayAtPreviousStop != null) {
                val withPrev = usable.filter { it.previousStopDelay != null }
                val lineWithPrev = usableLine.filter { it.previousStopDelay != null }
                if (withPrev.size + lineWithPrev.size >= MIN_EFFECTIVE_N) {
                    val bandwidth = maxOf(3.0, 0.3 * abs(liveDelayAtPreviousStop))
                    fun point(run: HistoricalRun, mine: Boolean): Pair<Double, Double> {
                        val prev = run.previousStopDelay!!.toDouble()
                        val z = (prev - liveDelayAtPreviousStop) / bandwidth
                        val kernel = LIVE_KERNEL_FLOOR + exp(-0.5 * z * z)
                        val value = liveDelayAtPreviousStop + (run.delayOrNull()!! - prev)
                        return value to weight(run, mine) * kernel
                    }
                    val points = ArrayList<Pair<Double, Double>>(
                        withPrev.size + lineWithPrev.size,
                    )
                    withPrev.mapTo(points) { point(it, true) }
                    lineWithPrev.mapTo(points) { point(it, false) }
                    points.sortBy { it.first }
                    return EmpiricalDelay(
                        points, sampleSize, effectiveN(points.map { it.second }),
                        cancelRate, true, ownShare,
                    )
                }
            }

            val points = ArrayList<Pair<Double, Double>>(usable.size + usableLine.size)
            usable.mapTo(points) { it.delayOrNull()!!.toDouble() to weight(it, true) }
            usableLine.mapTo(points) { it.delayOrNull()!!.toDouble() to weight(it, false) }
            points.sortBy { it.first }
            return EmpiricalDelay(
                points, sampleSize, effectiveN(points.map { it.second }), cancelRate,
                false, ownShare,
            )
        }

        /**
         * The factor that gives a group [share] of the total weight.
         *
         * A share rather than a multiplier because the two groups are so
         * lopsided: with three of the train's own runs against five hundred of
         * its line's, even a multiplier of sixteen leaves the train's own
         * history a rounding error, while a share of one half means one half.
         */
        private fun groupScale(
            group: List<HistoricalRun>,
            queryDate: LocalDate,
            share: Double,
        ): Double {
            if (group.isEmpty() || share <= 0.0) return 0.0
            val mass = group.sumOf { baseWeight(it.date, queryDate) }
            return if (mass > 0) share / mass else 0.0
        }

        private fun cancelRate(group: List<HistoricalRun>): Double? =
            if (group.isEmpty()) null
            else group.count { it.cancelled }.toDouble() / group.size

        private fun blend(own: Double?, line: Double?, ownShare: Double): Double = when {
            own == null -> line ?: 0.0
            line == null -> own
            else -> ownShare * own + (1.0 - ownShare) * line
        }

        private fun effectiveN(weights: List<Double>): Double {
            var sumW = 0.0
            var sumW2 = 0.0
            for (w in weights) {
                sumW += w
                sumW2 += w * w
            }
            return if (sumW2 > 0) sumW * sumW / sumW2 else 0.0
        }

        private fun HistoricalRun.delayOrNull(): Int? = arrivalDelay ?: departureDelay

        fun timeOfDayDistance(a: String, b: String): Int =
            timeOfDayDistance(minutesOfDay(a), minutesOfDay(b))

        /** Circular distance in minutes; [Int.MAX_VALUE] if either is unknown. */
        fun timeOfDayDistance(a: Int, b: Int): Int {
            if (a < 0 || b < 0) return Int.MAX_VALUE
            val diff = abs(a - b)
            return minOf(diff, 24 * 60 - diff)
        }

        /** "HH:mm" as minutes past midnight, or -1 when it is not that. */
        fun minutesOfDay(hhmm: String): Int {
            if (hhmm.length != 5 || hhmm[2] != ':') return -1
            val h = digits(hhmm, 0)
            val m = digits(hhmm, 3)
            if (h < 0 || m < 0 || h > 23 || m > 59) return -1
            return h * 60 + m
        }

        /** Two ASCII digits at [at], or -1 — no substring, no boxing. */
        private fun digits(text: String, at: Int): Int {
            val tens = text[at] - '0'
            val ones = text[at + 1] - '0'
            if (tens !in 0..9 || ones !in 0..9) return -1
            return tens * 10 + ones
        }
    }
}
