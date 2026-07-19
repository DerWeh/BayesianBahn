package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.model.DelayDistribution
import io.github.derweh.bayesianbahn.model.DeutschlandTicket

/**
 * From/to journey search in the style people know from the DB Navigator:
 * the user names origin, destination and departure time; the planner finds
 * direct trains and one-transfer connections and predicts the *arrival
 * distribution* at the destination for each.
 *
 * Routing is heuristic, not exhaustive: candidate trains come from the
 * origin's IRIS board; for trains not running through the destination, the
 * highest-weight station on their route serves as the transfer, evaluated
 * via [ConnectionPlanner]'s Bayesian propagation.
 */
class JourneyPlanner(
    private val stationRepository: StationRepository,
    private val historyRepository: HistoryRepository,
    private val irisClient: IrisClient,
    private val connectionPlanner: ConnectionPlanner,
    private val predictor: Predictor = Predictor(),
    /** Fallback for departure times beyond IRIS's plan horizon. */
    private val syntheticTimetable: SyntheticTimetable? = null,
) {

    /** One planned journey option with its predicted arrival distribution. */
    data class Itinerary(
        val feeder: TimetableStop,
        val departureMillis: Long,
        /** Null for a direct train. */
        val transferStation: String?,
        /** Arrival delay at the destination, minutes relative to [referenceArrivalMillis]. */
        val distribution: DelayDistribution,
        val referenceArrivalMillis: Long,
        /** Probability of catching the first listed connecting train (transfer only). */
        val catchProbability: Double?,
        /** Probability of missing every listed connecting train (transfer only). */
        val missProbability: Double?,
        /** Full connection outcome for detail display (transfer only). */
        val connection: ConnectionPlanner.Outcome.Success?,
    ) {
        val medianArrivalMillis: Long
            get() = referenceArrivalMillis + (distribution.quantile(0.5) * 60_000).toLong()
    }

    sealed interface Outcome {
        data class Error(val message: String) : Outcome
        data class Success(
            val itineraries: List<Itinerary>,
            val from: Station,
            val to: Station,
            /** True when planned from the historical timetable, not IRIS. */
            val synthetic: Boolean = false,
        ) : Outcome
    }

    suspend fun plan(
        fromQuery: String,
        toQuery: String,
        departMillis: Long,
        deutschlandTicketOnly: Boolean,
        transferMinutes: Int = 5,
    ): Outcome {
        val from = stationRepository.search(fromQuery).firstOrNull()
            ?: return Outcome.Error("Station \"$fromQuery\" not found.")
        val to = stationRepository.search(toQuery).firstOrNull()
            ?: return Outcome.Error("Station \"$toQuery\" not found.")
        if (from.eva == to.eva) return Outcome.Error("Origin and destination are the same.")

        var board = try {
            irisClient.board(from.eva, hours = 3, startMillis = departMillis)
        } catch (e: Exception) {
            return Outcome.Error(e.message ?: "network error")
        }
        var synthetic = false
        if (board.isEmpty() && syntheticTimetable != null) {
            // Beyond IRIS's ~1 day plan horizon: reconstruct the board from
            // the historical timetable (weekday-aware, no live data).
            board = syntheticTimetable.board(from.eva, departMillis, hours = 3)
            synthetic = true
        }

        val departures = board
            .filter { it.departure?.plannedTime != null && !it.departure!!.cancelled }
            .filter { it.departure!!.plannedTime!! >= departMillis }
            .filter { !deutschlandTicketOnly || DeutschlandTicket.covers(it.label.category) }
            .sortedBy { it.departure!!.plannedTime }

        val (direct, others) = departures.partition { stop ->
            stop.departure!!.plannedPath.any { pathMatches(it, to.name) }
        }

        val itineraries = mutableListOf<Itinerary>()
        for (stop in direct.take(MAX_DIRECT)) {
            directItinerary(stop, to)?.let { itineraries += it }
        }
        var transferBudget = MAX_TRANSFER_ATTEMPTS
        var found = 0
        // Many feeders share the same best transfer (e.g. every S-Bahn via
        // the same hub) — evaluate each transfer station only once so the
        // budget reaches genuinely different routes.
        val triedTransfers = mutableSetOf<String>()
        for (stop in others.take(MAX_TRANSFER_SCAN)) {
            if (transferBudget <= 0 || found >= MAX_TRANSFER_RESULTS) break
            val itinerary = transferItinerary(
                stop, to, transferMinutes, deutschlandTicketOnly, triedTransfers,
            ) { transferBudget-- > 0 }
            if (itinerary != null) {
                itineraries += itinerary
                found++
            }
        }
        if (itineraries.isEmpty()) {
            return Outcome.Error(
                if (departures.isEmpty()) {
                    "No timetable data for ${from.name} at that time."
                } else {
                    "No plannable trains from ${from.name} towards ${to.name} " +
                        "found around that time."
                },
            )
        }
        return Outcome.Success(
            itineraries.sortedBy { it.medianArrivalMillis }.take(MAX_RESULTS),
            from,
            to,
            synthetic = synthetic,
        )
    }

    /** A train running through the destination: predict its arrival there. */
    private suspend fun directItinerary(stop: TimetableStop, to: Station): Itinerary? {
        val departure = stop.departure?.plannedTime ?: return null
        val history = historyRepository.load(stop.label.category, stop.label.number, stop.label.line)
        val destHistory = history?.stations?.entries
            ?.firstOrNull { (name, sh) -> sh.eva == to.eva || pathMatches(name, to.name) }
            ?.value ?: return null
        val arrivalTod = destHistory.runs.maxByOrNull { it.date }?.plannedTimeOfDay ?: return null
        val plannedArr = ConnectionPlanner.arrivalMillis(departure, arrivalTod) ?: return null
        val forecast = predictor.forecast(
            history = history,
            stationEva = to.eva,
            stationName = to.name,
            trainCategory = stop.label.category,
            plannedTimeMillis = plannedArr,
            liveDelayMinutes = stop.departure?.liveDelayMinutes,
        )
        return Itinerary(
            feeder = stop,
            departureMillis = departure,
            transferStation = null,
            distribution = forecast.distribution,
            referenceArrivalMillis = plannedArr,
            catchProbability = null,
            missProbability = null,
            connection = null,
        )
    }

    /**
     * A train not reaching the destination: try changing at the biggest
     * stations on its route (largest first) until one works. [tryAttempt]
     * gates each network-heavy evaluation against the shared budget.
     */
    private suspend fun transferItinerary(
        stop: TimetableStop,
        to: Station,
        transferMinutes: Int,
        deutschlandTicketOnly: Boolean,
        triedTransfers: MutableSet<String>,
        tryAttempt: () -> Boolean,
    ): Itinerary? {
        val departure = stop.departure?.plannedTime ?: return null
        val transfers = stop.departure!!.plannedPath
            .mapNotNull { name -> stationRepository.byName(name) }
            .filter { it.eva != to.eva && it.weight >= MIN_TRANSFER_WEIGHT }
            .filter { it.name !in triedTransfers }
            .sortedByDescending { it.weight }
            .take(TRANSFERS_PER_FEEDER)
        for (transfer in transfers) {
            if (!tryAttempt()) return null
            triedTransfers += transfer.name
            val outcome = connectionPlanner.plan(
                feeder = stop,
                transferQuery = transfer.name,
                destinationQuery = to.name,
                transferMinutes = transferMinutes,
                deutschlandTicketOnly = deutschlandTicketOnly,
                boardStartMillis = departure,
            ) as? ConnectionPlanner.Outcome.Success ?: continue
            return Itinerary(
                feeder = stop,
                departureMillis = departure,
                transferStation = outcome.transferStation.name,
                distribution = outcome.result.distribution,
                referenceArrivalMillis = outcome.result.referenceArrivalMillis,
                // The first *plannable* connection (departing after the
                // feeder's planned arrival) — normally-missed earlier trains
                // are listed in the details but don't define the headline.
                catchProbability = outcome.result.candidates
                    .firstOrNull {
                        !it.candidate.cancelledLive &&
                            it.candidate.plannedDepartureMillis >=
                            outcome.feederPlannedArrivalMillis
                    }?.boardProbability
                    ?: outcome.result.candidates
                        .firstOrNull { !it.candidate.cancelledLive }?.boardProbability,
                missProbability = outcome.result.missProbability,
                connection = outcome,
            )
        }
        return null
    }

    companion object {
        const val MAX_DIRECT = 3

        /** Feeders considered; the attempt budget below limits network work. */
        const val MAX_TRANSFER_SCAN = 15
        const val MAX_TRANSFER_RESULTS = 3

        /** Cap on connection evaluations (each fetches a transfer board). */
        const val MAX_TRANSFER_ATTEMPTS = 6
        const val TRANSFERS_PER_FEEDER = 2
        const val MAX_RESULTS = 5

        /** Skip pure village halts; real junctions can be small (Buchloe: 167). */
        const val MIN_TRANSFER_WEIGHT = 40

        /** Route entries sometimes differ in suffixes ("Hbf") — match loosely. */
        fun pathMatches(pathStation: String, destination: String): Boolean =
            pathStation.equals(destination, ignoreCase = true) ||
                pathStation.contains(destination, ignoreCase = true) ||
                destination.contains(pathStation, ignoreCase = true)
    }
}
