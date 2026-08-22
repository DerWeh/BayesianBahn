package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.DelayDistribution
import io.github.derweh.bayesianbahn.model.DelayModel
import io.github.derweh.bayesianbahn.model.EmpiricalDelay
import io.github.derweh.bayesianbahn.model.StudentTDelay
import io.github.derweh.bayesianbahn.model.TimeBand
import io.github.derweh.bayesianbahn.model.TrainClass
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

enum class ForecastSource {
    /** Empirical distribution conditioned on the train's live delay. */
    EMPIRICAL_LIVE,

    /** Empirical distribution of past runs, no usable live signal. */
    EMPIRICAL,

    /** Prior-based fallback — no history available for this train. */
    PRIOR,
}

data class Forecast(
    val distribution: DelayDistribution,
    val source: ForecastSource,
    val runCount: Int,
    val effectiveRuns: Double,
    /** Fraction of past runs that were cancelled, null when unknown. */
    val cancelProbability: Double?,
    /**
     * A live report that arrived but was not treated as evidence, in minutes.
     * Null when DB reported a delay we did use, or reported nothing at all.
     * The screens use it to say why the forecast disagrees with the board.
     */
    val ignoredLiveDelay: Double? = null,
)

/**
 * Combines a train's historical runs with its live state into an arrival
 * delay forecast; falls back to the Bayesian prior model when no history
 * exists for the train.
 */
class Predictor(private val fallbackModel: DelayModel = DelayModel()) {

    fun forecast(
        history: TrainHistory?,
        stationEva: String,
        stationName: String,
        trainCategory: String,
        plannedTimeMillis: Long,
        liveDelayMinutes: Double?,
        today: LocalDate = LocalDate.now(ZONE),
    ): Forecast {
        val reported = informativeLiveDelay(liveDelayMinutes)
        val ignored = liveDelayMinutes.takeIf { reported == null }

        val stationHistory = history?.stations?.entries?.firstOrNull { (name, sh) ->
            sh.eva == stationEva || StationNames.matches(name, stationName)
        }?.value

        if (stationHistory != null) {
            val timeOfDay = Instant.ofEpochMilli(plannedTimeMillis).atZone(ZONE).format(HHMM)
            // Draft approximation: the live delay reported for this station
            // stands in for the delay at the previous stop that historical
            // runs were annotated with. Replace with the true previous-stop
            // live delay once the board fetches neighbouring stations.
            val empirical = EmpiricalDelay.build(
                runs = stationHistory.runs,
                queryTimeOfDay = timeOfDay,
                queryDate = today,
                liveDelayAtPreviousStop = reported,
            )
            if (empirical != null && empirical.effectiveSampleSize >= EmpiricalDelay.MIN_EFFECTIVE_N) {
                return Forecast(
                    distribution = empirical,
                    source = if (empirical.conditionedOnLive) {
                        ForecastSource.EMPIRICAL_LIVE
                    } else {
                        ForecastSource.EMPIRICAL
                    },
                    runCount = empirical.sampleSize,
                    effectiveRuns = empirical.effectiveSampleSize,
                    cancelProbability = empirical.cancelProbability,
                    ignoredLiveDelay = ignored,
                )
            }
        }

        val trainClass = TrainClass.fromCategory(trainCategory)
        val band = TimeBand.fromEpochMillis(plannedTimeMillis)
        return Forecast(
            distribution = StudentTDelay(
                fallbackModel.predictiveFor(trainClass, band, reported),
            ),
            source = ForecastSource.PRIOR,
            runCount = 0,
            effectiveRuns = 0.0,
            cancelProbability = null,
            ignoredLiveDelay = ignored,
        )
    }

    companion object {
        private val ZONE: ZoneId = ZoneId.of("Europe/Berlin")
        private val HHMM: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm")

        /**
         * Minutes. Below this a live report is not treated as evidence.
         *
         * DB reports a stop in four shapes and three of them mean "on time":
         * the predicted time moved, it was confirmed unchanged, the stop is
         * listed without a time, or it is absent. Only the first is an
         * observation; the rest are the plan, restated. Scored against the
         * archive over 2026-08-17..19, DB called a train on time for 61% of
         * stops ten minutes out and 99% of stops three hours out — and 31% of
         * that last group arrived more than two minutes late. Anchoring the
         * forecast on a number that carries no information cost 0.53 min of
         * CRPS on trains that have history, and left the stated 80% interval
         * covering 55% of arrivals instead of 80%.
         *
         * A report of "early" is no better: those trains averaged 1.4 minutes
         * *late*. So the threshold sits above zero rather than at it.
         */
        const val MIN_INFORMATIVE_DELAY_MINUTES = 1.0

        /**
         * The live report if it is evidence, null if it is the plan restated.
         *
         * Returning null is what makes the rest of the model ignore it: every
         * live path is already written to handle "no live data", because that
         * is the normal case for a journey planned more than a day ahead.
         */
        fun informativeLiveDelay(liveDelayMinutes: Double?): Double? =
            liveDelayMinutes?.takeIf { it >= MIN_INFORMATIVE_DELAY_MINUTES }
    }
}
