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
 * **This searches direct journeys and journeys with one change only.** Two or
 * more changes are not attempted at all, so "nothing found" here never means
 * "no connection exists" — see [UserMessages.ONE_CHANGE_ONLY], which is what
 * the user is told.
 *
 * Routing is heuristic, not exhaustive, even within that one-change scope:
 * candidate trains come from the origin's IRIS board for [ORIGIN_HOURS] hours;
 * for trains not reaching the destination, stations on their route are ranked
 * by distance to the destination (see [transferCandidates]) and at most
 * [MAX_TRANSFER_ATTEMPTS] of them are evaluated via [ConnectionPlanner]'s
 * Bayesian propagation.
 *
 * `tools/journey_bench.py` measures what that costs: over 44 journeys that
 * provably have a one-change connection, 98% were found. The residual misses
 * are transfer stations that never enter the candidate list, which no amount
 * of budget reaches.
 */
class JourneyPlanner(
    private val stationRepository: StationRepository,
    private val historyRepository: HistoryRepository,
    private val irisClient: IrisClient,
    private val connectionPlanner: ConnectionPlanner,
    private val predictor: Predictor = Predictor(),
    /** Fallback for departure times beyond IRIS's plan horizon. */
    private val syntheticTimetable: SyntheticTimetable? = null,
    private val routeStations: RouteStationMatcher = RouteStationMatcher(irisClient),
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
            /** True when IRIS was unreachable, so nothing here reflects live data. */
            val offline: Boolean = false,
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

        // A dead connection costs live delays, not the whole search: the
        // downloaded history is a timetable in its own right, and the same
        // fallback that covers dates beyond IRIS's horizon covers this too.
        var offline = false
        var board = try {
            irisClient.board(from.eva, hours = ORIGIN_HOURS, startMillis = departMillis)
        } catch (e: Exception) {
            offline = true
            emptyList()
        }
        var synthetic = false
        if (board.isEmpty() && syntheticTimetable != null) {
            // Beyond IRIS's ~1 day plan horizon: reconstruct the board from
            // the historical timetable (weekday-aware, no live data).
            board = syntheticTimetable.board(from.eva, departMillis, hours = ORIGIN_HOURS)
            synthetic = true
        }
        if (board.isEmpty() && offline) {
            return Outcome.Error(UserMessages.TIMETABLE_UNREACHABLE_NO_HISTORY)
        }

        val departures = board
            .filter { it.departure?.plannedTime != null && !it.departure!!.cancelled }
            .filter { it.departure!!.plannedTime!! >= departMillis }
            .filter { !deutschlandTicketOnly || DeutschlandTicket.covers(it.label.category) }
            .sortedBy { it.departure!!.plannedTime }

        val isDestination = routeStations.matcherFor(to)
        val (direct, others) = departures.partition { stop ->
            stop.departure!!.plannedPath.any { isDestination(it) }
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
                stop, from, to, transferMinutes, deutschlandTicketOnly, triedTransfers,
            ) { transferBudget-- > 0 }
            if (itinerary != null) {
                itineraries += itinerary
                found++
            }
        }
        if (itineraries.isEmpty()) {
            return Outcome.Error(
                when {
                    // Transfers need the transfer station's board, so offline
                    // only direct trains can be planned. Say that rather than
                    // claiming no train goes there.
                    offline -> UserMessages.TIMETABLE_UNREACHABLE
                    departures.isEmpty() -> "No timetable data for ${from.name} at that time."
                    else -> UserMessages.noConnection(from.name, to.name)
                },
            )
        }
        return Outcome.Success(
            itineraries.sortedBy { it.medianArrivalMillis }.take(MAX_RESULTS),
            from,
            to,
            synthetic = synthetic,
            offline = offline,
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
        from: Station,
        to: Station,
        transferMinutes: Int,
        deutschlandTicketOnly: Boolean,
        triedTransfers: MutableSet<String>,
        tryAttempt: () -> Boolean,
    ): Itinerary? {
        val departure = stop.departure?.plannedTime ?: return null
        val transfers = transferCandidates(
            path = routeStations.stationsOn(stop.departure!!.plannedPath),
            origin = from,
            destination = to,
            exclude = triedTransfers,
        ).take(TRANSFERS_PER_FEEDER)
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

        /** Hours of the origin's board that are searched for departures. */
        const val ORIGIN_HOURS = 3

        /** Feeders considered; the attempt budget below limits network work. */
        const val MAX_TRANSFER_SCAN = 15
        const val MAX_TRANSFER_RESULTS = 3

        /**
         * Cap on connection evaluations (each fetches a transfer board, which
         * is five IRIS requests).
         *
         * Measured with `tools/journey_bench.py` over 44 journeys that provably
         * have a one-transfer connection: 84% of them are found within 4
         * attempts, 93% within 6, 98% within 8 — and nothing more up to 12. The
         * remaining case fails for a different reason (its transfer station
         * never enters the candidate list), so more budget cannot reach it.
         * Eight is where the curve flattens; it costs on average one extra
         * board fetch per search over six.
         */
        const val MAX_TRANSFER_ATTEMPTS = 8

        /**
         * Two per feeder beat one (98% vs 95% found) and three (95%, and slower
         * to the first result): a third candidate is usually a worse station on
         * a route already shown not to work.
         */
        const val TRANSFERS_PER_FEEDER = 2
        const val MAX_RESULTS = 5

        /** Skip pure village halts; real junctions can be small (Buchloe: 167). */
        const val MIN_TRANSFER_WEIGHT = 40

        /**
         * How much farther from the destination than the origin a transfer may
         * lie. Above 1.0 so that changing at a hub slightly "behind" the origin
         * stays possible, which is sometimes the only way out of a small station.
         */
        const val DETOUR_TOLERANCE = 1.25

        /**
         * Transfer stations worth spending an attempt on, best first.
         *
         * Ranking by station size alone sent Ulm → Türkheim (Bay) through
         * Stuttgart Hbf (weight 1009, and 130 km the wrong way) and then
         * Göppingen, exhausting the attempt budget before reaching the change
         * at Memmingen that actually works. Distance to the destination is a
         * far better predictor of a useful change than importance is, so rank
         * by it and drop candidates that clearly lead away.
         *
         * Stations without coordinates keep the old weight ordering and go
         * last, so a gap in the station list can only cost ranking quality.
         */
        fun transferCandidates(
            path: List<Station>,
            origin: Station,
            destination: Station,
            exclude: Set<String> = emptySet(),
        ): List<Station> {
            val usable = path.filter {
                it.eva != destination.eva &&
                    it.weight >= MIN_TRANSFER_WEIGHT &&
                    it.name !in exclude
            }
            val goal = origin.distanceKm(destination)
                ?: return usable.sortedByDescending { it.weight }
            val (located, unlocated) = usable.partition { it.distanceKm(destination) != null }
            return located
                .map { it to it.distanceKm(destination)!! }
                .filter { (_, d) -> d <= goal * DETOUR_TOLERANCE }
                .sortedBy { (_, d) -> d }
                .map { (station, _) -> station } +
                unlocated.sortedByDescending { it.weight }
        }

        /** IRIS route entries spell stations differently — see [StationNames]. */
        fun pathMatches(pathStation: String, destination: String): Boolean =
            StationNames.matches(pathStation, destination)
    }
}
