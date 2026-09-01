package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.DelayDistribution
import io.github.derweh.bayesianbahn.model.DelayModel
import io.github.derweh.bayesianbahn.model.EmpiricalDelay
import io.github.derweh.bayesianbahn.model.LiveReport
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

    /** As [EMPIRICAL_LIVE], but from the line's runs — this number has none. */
    EMPIRICAL_LINE_LIVE,

    /** As [EMPIRICAL], but from the line's runs — this number has none. */
    EMPIRICAL_LINE,

    /** Prior-based fallback — no history for this train and none for its line. */
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
    /**
     * The line the runs came from, when they are not this train's own. The
     * screens name it, because "past runs of this train" would be a lie and
     * the difference is one a user can judge.
     */
    val lineName: String? = null,
)

/**
 * Combines a train's historical runs with its live state into an arrival
 * delay forecast.
 *
 * Three sources, tried in that order: this train's own past runs at this
 * station; the runs of its *line* there, when the run number is too new to
 * have a history of its own; and finally the Bayesian prior for its class,
 * which knows nothing about the station or the hour.
 *
 * The middle step is worth having because the last one is weak and common.
 * Over eleven days of collected forecasts a quarter of arrivals fell through
 * to the prior, and `pipeline/backtest_fallback.py`, walking 781,000 archive
 * events across 62 stations, puts the line's answer 0.43 minutes of CRPS ahead
 * of the prior's on exactly that population (95% 0.40..0.46, resampling whole
 * trains), with a shard available for 88% of it. It is a fallback and not a
 * promotion: where the number *does* have a history, its own runs beat its
 * line's by 0.13 minutes, so the line is never consulted there.
 */
class Predictor(private val fallbackModel: DelayModel = DelayModel()) {

    suspend fun forecast(
        history: TrainHistory?,
        stationEva: String,
        stationName: String,
        trainCategory: String,
        plannedTimeMillis: Long,
        liveDelayMinutes: Double?,
        today: LocalDate = LocalDate.now(ZONE),
        lineHistory: suspend () -> TrainHistory? = { null },
    ): Forecast {
        val reported = LiveReport.informative(liveDelayMinutes)
        val ignored = liveDelayMinutes.takeIf { reported == null }
        val timeOfDay = Instant.ofEpochMilli(plannedTimeMillis).atZone(ZONE).format(HHMM)

        fun usable(from: TrainHistory?): EmpiricalDelay? {
            val stationHistory = from?.stations?.entries?.firstOrNull { (name, sh) ->
                sh.eva == stationEva || StationNames.matches(name, stationName)
            }?.value ?: return null
            // Draft approximation: the live delay reported for this station
            // stands in for the delay at the previous stop that historical
            // runs were annotated with. Replace with the true previous-stop
            // live delay once the board fetches neighbouring stations.
            return EmpiricalDelay.build(
                runs = stationHistory.runs,
                queryTimeOfDay = timeOfDay,
                queryDate = today,
                liveDelayAtPreviousStop = reported,
            )?.takeIf { it.effectiveSampleSize >= EmpiricalDelay.MIN_EFFECTIVE_N }
        }

        usable(history)?.let { own ->
            return Forecast(
                distribution = own,
                source = if (own.conditionedOnLive) {
                    ForecastSource.EMPIRICAL_LIVE
                } else {
                    ForecastSource.EMPIRICAL
                },
                runCount = own.sampleSize,
                effectiveRuns = own.effectiveSampleSize,
                cancelProbability = own.cancelProbability,
                ignoredLiveDelay = ignored,
            )
        }

        // Only now is the line shard worth a fetch: it is one file per line
        // rather than per run, so it is larger, and three quarters of
        // predictions never need it.
        val fromLine = lineHistory()
        usable(fromLine)?.let { line ->
            return Forecast(
                distribution = line,
                source = if (line.conditionedOnLive) {
                    ForecastSource.EMPIRICAL_LINE_LIVE
                } else {
                    ForecastSource.EMPIRICAL_LINE
                },
                runCount = line.sampleSize,
                effectiveRuns = line.effectiveSampleSize,
                cancelProbability = line.cancelProbability,
                ignoredLiveDelay = ignored,
                lineName = fromLine?.trainName,
            )
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

    private companion object {
        val ZONE: ZoneId = ZoneId.of("Europe/Berlin")
        val HHMM: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm")
    }
}
