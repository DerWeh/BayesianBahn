package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.DelayDistribution
import io.github.derweh.bayesianbahn.model.DelayModel
import io.github.derweh.bayesianbahn.model.EmpiricalDelay
import io.github.derweh.bayesianbahn.model.HistoricalRun
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
     * The line whose runs helped, null when the forecast is the train's own
     * alone. The screens name it, because "past runs of this train" would
     * otherwise be a lie, and the difference is one a user can judge.
     */
    val lineName: String? = null,
    /**
     * How much of the forecast is the train's own history rather than its
     * line's: 1 when the line was never consulted, 0 when it is the whole
     * answer.
     */
    val ownShare: Double = 1.0,
)

/**
 * Combines a train's historical runs with its live state into an arrival
 * delay forecast.
 *
 * Two sources, not three. This train's own past runs at this station and the
 * runs of its *line* there go into one distribution, the train's own taking
 * `n / (n + 8)` of the weight — so a run number with nothing behind it answers
 * almost entirely from its line, and one with months of history barely notices
 * the line is there. Only when even the two together are too thin does the
 * Bayesian prior for its class answer, which knows neither the station nor the
 * hour.
 *
 * That last case used to be a quarter of arrivals: IRIS renumbers a run at
 * every timetable change, and over eleven days of collected forecasts the
 * trains that fell through to the prior had a median of two runs at the station
 * against 106 for the trains that did not. Pooling instead of switching is what
 * `pipeline/backtest_fallback.py` measured over 781,000 archive events: worth
 * 0.105 min of CRPS against answering from the line alone (95% 0.094..0.116)
 * and 0.002 against answering from the number alone (95% 0.002..0.003), where
 * a *switch* between the two — or any fixed weighting — has to give up one end
 * to have the other. End to end, 0.025 min against the version with no line at
 * all (95% 0.023..0.026), and the share of arrivals left to the prior falls
 * from 4.8% to 0.5%.
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

        fun runsAt(from: TrainHistory?): List<HistoricalRun> =
            from?.stations?.entries?.firstOrNull { (name, sh) ->
                sh.eva == stationEva || StationNames.matches(name, stationName)
            }?.value?.runs.orEmpty()

        // Draft approximation: the live delay reported for this station stands
        // in for the delay at the previous stop that historical runs were
        // annotated with. Replace with the true previous-stop live delay once
        // the board fetches neighbouring stations.
        fun build(own: List<HistoricalRun>, line: List<HistoricalRun>) =
            EmpiricalDelay.build(
                runs = own,
                queryTimeOfDay = timeOfDay,
                queryDate = today,
                liveDelayAtPreviousStop = reported,
                lineRuns = line,
            )

        val ownRuns = runsAt(history)
        val alone = build(ownRuns, emptyList())
        // The line shard is one file per line and station rather than per run
        // number, so it costs a fetch — and above this much history of its own
        // a train has nothing to gain from it. Roughly seven predictions in
        // eight stop here.
        var fromLine: TrainHistory? = null
        var empirical = alone
        if ((alone?.effectiveSampleSize ?: 0.0) < EmpiricalDelay.LINE_CEILING_N) {
            fromLine = lineHistory()
            if (fromLine != null) empirical = build(ownRuns, runsAt(fromLine))
        }

        empirical?.takeIf { it.effectiveSampleSize >= EmpiricalDelay.MIN_EFFECTIVE_N }
            ?.let { model ->
                // Whose history the screens should describe: whichever side
                // holds most of the weight. The own runs cross a half at
                // exactly the effective count that used to switch the model.
                val mostlyOwn = model.ownShare >= 0.5
                return Forecast(
                    distribution = model,
                    source = when {
                        mostlyOwn && model.conditionedOnLive -> ForecastSource.EMPIRICAL_LIVE
                        mostlyOwn -> ForecastSource.EMPIRICAL
                        model.conditionedOnLive -> ForecastSource.EMPIRICAL_LINE_LIVE
                        else -> ForecastSource.EMPIRICAL_LINE
                    },
                    runCount = model.sampleSize,
                    effectiveRuns = model.effectiveSampleSize,
                    cancelProbability = model.cancelProbability,
                    ignoredLiveDelay = ignored,
                    ownShare = model.ownShare,
                    // Named whenever it contributed at all, not only when it
                    // dominates: at eight to thirty runs of its own a train
                    // still takes a fifth to a half of its answer from its
                    // line, and the screens should not call that its own.
                    lineName = fromLine?.trainName?.takeIf { model.ownShare < 1.0 },
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
