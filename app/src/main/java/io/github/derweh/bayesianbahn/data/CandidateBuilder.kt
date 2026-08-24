package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.ConnectionModel
import io.github.derweh.bayesianbahn.model.EmpiricalDelay
import io.github.derweh.bayesianbahn.model.HistoricalRun
import java.time.Instant
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Turns one train's history into a [ConnectionModel.Candidate] for a given
 * transfer and destination.
 *
 * Lifted out of [ConnectionPlanner] so the evaluation harness can build
 * candidates the same way the app does instead of describing them a second
 * time. The Python mirror of the arrival model has already drifted once — its
 * long-distance set is missing `WB` — and a second description of *this* would
 * drift the same way, silently, while still producing plausible numbers.
 *
 * Everything here is a pure function of the history and the stop's own facts,
 * which is why it needed no repository to move.
 */
object CandidateBuilder {

    private val ZONE = ZoneId.of("Europe/Berlin")
    private val HHMM = DateTimeFormatter.ofPattern("HH:mm")

    /** Fewer joint runs than this and only a live report can carry a candidate. */
    const val MIN_JOINT_RUNS = 5

    /** The longest leg a reconstruction may claim: 14 hours, night trains included. */
    const val MAX_LEG_MINUTES = 14 * 60

    fun build(
        history: TrainHistory?,
        id: String,
        label: String,
        transferEva: String,
        transferName: String,
        destinationEva: String?,
        destinationName: String,
        plannedDepartureMillis: Long,
        liveDepartureDelay: Double?,
        cancelledLive: Boolean,
        today: LocalDate,
    ): ConnectionModel.Candidate? {
        val transferHistory = history?.stations?.entries?.firstOrNull { (name, sh) ->
            sh.eva == transferEva || StationNames.matches(name, transferName)
        }?.value
        // Shards carry the eva, so prefer it and fall back to the name only
        // for entries that predate it.
        val destinationHistory = history?.stations?.entries?.firstOrNull { (name, sh) ->
            (destinationEva != null && sh.eva == destinationEva) ||
                StationNames.matches(name, destinationName)
        }?.value

        val depHhmm = Instant.ofEpochMilli(plannedDepartureMillis).atZone(ZONE).format(HHMM)
        val relevant = transferHistory?.runs.orEmpty().filter {
            EmpiricalDelay.timeOfDayDistance(it.plannedTimeOfDay, depHhmm) <=
                EmpiricalDelay.TIME_OF_DAY_WINDOW_MIN
        }
        val cancelRate = if (relevant.isEmpty()) 0.0 else {
            relevant.count { it.cancelled }.toDouble() / relevant.size
        }

        val arrivalByDate = destinationHistory?.runs.orEmpty()
            .filter { !it.cancelled && (it.arrivalDelay ?: it.departureDelay) != null }
            .associateBy { it.date }
        // Departure and arrival come from the same historical run, so their
        // correlation survives into the mixture; pairing them across runs would
        // throw away exactly the dependence a two-leg journey turns on.
        val joint = relevant.mapNotNull { run ->
            val dep = (run.departureDelay ?: run.arrivalDelay) ?: return@mapNotNull null
            if (run.cancelled) return@mapNotNull null
            val arrRun = arrivalByDate[run.date] ?: return@mapNotNull null
            val arr = (arrRun.arrivalDelay ?: arrRun.departureDelay) ?: return@mapNotNull null
            ConnectionModel.JointRun(
                departureDelay = dep.toDouble(),
                arrivalDelay = arr.toDouble(),
                weight = EmpiricalDelay.baseWeight(run.date, today),
            )
        }
        if (joint.size < MIN_JOINT_RUNS && liveDepartureDelay == null && !cancelledLive) {
            return null
        }

        val plannedArr = plannedArrival(
            relevant, destinationHistory?.runs.orEmpty(), plannedDepartureMillis,
        ) ?: return null

        return ConnectionModel.Candidate(
            id = id,
            label = label,
            plannedDepartureMillis = plannedDepartureMillis,
            plannedArrivalMillis = plannedArr,
            runs = joint,
            liveDepartureDelay = liveDepartureDelay,
            cancelledLive = cancelledLive,
            cancelRate = cancelRate,
        )
    }

    /**
     * Planned arrival at the destination, from how long the leg has taken before.
     *
     * The IRIS board gives a departure and nothing beyond it, so the arrival has
     * to be recovered from history. Doing that by taking the destination's most
     * recent *time of day* and hanging it on today's date looks equivalent and
     * is not: the two come from different rides, and when the recovered time of
     * day falls a little before today's departure — a timetable that shifted by
     * half an hour is enough — the date has to roll forward to keep the arrival
     * after the departure, and a 28-minute leg is published as 24 hours and 28
     * minutes. Measured over one collected day, four candidates in six landed a
     * day late that way.
     *
     * The *leg* does not have that failure mode, because departure and arrival
     * are read from the same run: a schedule that shifts moves both. Median over
     * the runs rather than the latest one, so a single odd run cannot set it.
     *
     * Elapsed minutes are added to the departure instant, which on the two
     * nights a year the clocks change puts the wall-clock arrival an hour out.
     * That is the smaller error, and the one that cannot compound.
     */
    fun plannedArrival(
        originRuns: List<HistoricalRun>,
        destinationRuns: List<HistoricalRun>,
        plannedDepartureMillis: Long,
    ): Long? {
        val byDate = destinationRuns.associateBy { it.date }
        val legs = originRuns
            .mapNotNull { run ->
                byDate[run.date]?.let { legMinutes(run.plannedTimeOfDay, it.plannedTimeOfDay) }
            }
            .sorted()
        if (legs.isEmpty()) return null
        val leg = legs[legs.size / 2]
        // Longer than any scheduled leg on this network, night trains included:
        // a reconstruction this far out is a failure, and declining the
        // candidate is better than predicting an arrival tomorrow.
        if (leg > MAX_LEG_MINUTES) return null
        return plannedDepartureMillis + leg * 60_000L
    }

    /** Scheduled minutes from one time of day to the next, wrapping past midnight. */
    fun legMinutes(departure: String, arrival: String): Int? {
        val from = runCatching { LocalTime.parse(departure, HHMM) }.getOrNull() ?: return null
        val to = runCatching { LocalTime.parse(arrival, HHMM) }.getOrNull() ?: return null
        return Math.floorMod(to.toSecondOfDay() / 60 - from.toSecondOfDay() / 60, 24 * 60)
    }
}
