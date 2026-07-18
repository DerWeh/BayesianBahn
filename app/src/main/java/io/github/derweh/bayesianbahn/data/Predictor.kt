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
        val stationHistory = history?.stations?.entries?.firstOrNull { (name, sh) ->
            sh.eva == stationEva || name.equals(stationName, ignoreCase = true)
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
                liveDelayAtPreviousStop = liveDelayMinutes,
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
                )
            }
        }

        val trainClass = TrainClass.fromCategory(trainCategory)
        val band = TimeBand.fromEpochMillis(plannedTimeMillis)
        return Forecast(
            distribution = StudentTDelay(
                fallbackModel.predictiveFor(trainClass, band, liveDelayMinutes),
            ),
            source = ForecastSource.PRIOR,
            runCount = 0,
            effectiveRuns = 0.0,
            cancelProbability = null,
        )
    }

    private companion object {
        val ZONE: ZoneId = ZoneId.of("Europe/Berlin")
        val HHMM: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm")
    }
}
