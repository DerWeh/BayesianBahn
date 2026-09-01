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
 * "no connection exists" — see [UserMessage.NoConnection], which is what
 * the user is told.
 *
 * Routing is heuristic, not exhaustive, even within that one-change scope:
 * candidate trains come from the origin's IRIS board for [ORIGIN_HOURS] hours;
 * for trains not reaching the destination, stations on their route are ranked
 * by distance to the destination (see [transferCandidates]) and at most
 * [MAX_TRANSFER_ATTEMPTS] of the resulting (feeder, transfer) pairs are
 * evaluated via [ConnectionPlanner]'s Bayesian propagation.
 *
 * Which pairs, though, is the whole game. The origin's board is one fetch and
 * every train on it carries its own route, so the set of pairs is already paid
 * for before a single attempt is spent — and the budget goes to the best of
 * them, across all departures at once, rather than to whichever train leaves
 * first. See [transferPairs].
 *
 * `tools/route_bench.py` measures what that costs, against ground truth that is
 * exhaustive within these same windows: of journeys that provably have a
 * one-change connection, 86% are found, spanning 91% from a village halt down
 * to 73% from a big hub. See [MAX_TRANSFER_ATTEMPTS].
 */
class JourneyPlanner(
    private val stationRepository: StationRepository,
    private val historyRepository: HistoryRepository,
    private val irisClient: IrisClient,
    private val connectionPlanner: ConnectionPlanner,
    private val predictor: Predictor = Predictor(),
    /** Fallback for departure times beyond IRIS's plan horizon. */
    private val syntheticTimetable: SyntheticTimetable? = null,
    private val routeStations: RouteStationMatcher =
        RouteStationMatcher(irisClient, stationRepository),
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

        /**
         * Trains on this journey DB is reporting trouble on.
         *
         * Not the same as a delay or a cancellation. When a section is blocked
         * the trains keep their times, so every number here is computed from a
         * timetable that is not going to happen — which is exactly how a
         * journey can be shown as fine while no passenger can travel it.
         */
        val disrupted: List<String>
            get() = (
                (if (feeder.disrupted) listOf("${feeder.label.category} ${feeder.label.number}")
                else emptyList()) + (connection?.disrupted ?: emptyList())
                ).distinct()
    }

    sealed interface Outcome {
        data class Error(val message: UserMessage) : Outcome
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
            ?: return Outcome.Error(UserMessage.StationNotFound(fromQuery))
        val to = stationRepository.search(toQuery).firstOrNull()
            ?: return Outcome.Error(UserMessage.StationNotFound(toQuery))
        if (from.eva == to.eva) return Outcome.Error(UserMessage.SameOriginAndDestination)

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
            // the historical timetable (weekday-aware, no live data). The
            // ticket filter goes in rather than being applied to the result:
            // fetching a shard for a train this search cannot use is the
            // slowest way to discard it.
            board = syntheticTimetable.board(from.eva, departMillis, hours = ORIGIN_HOURS) {
                !deutschlandTicketOnly || DeutschlandTicket.covers(it.category)
            }
            synthetic = true
        }
        if (board.isEmpty() && offline) {
            return Outcome.Error(UserMessage.TimetableUnreachableNoHistory)
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
            directItinerary(stop, from, to)?.let { itineraries += it }
        }
        itineraries += spendTransferBudget(
            transferPairs(others.take(MAX_TRANSFER_SCAN), from, to),
        ) { stop, transfer ->
            transferItinerary(stop, transfer, to, transferMinutes, deutschlandTicketOnly)
        }
        if (itineraries.isEmpty()) {
            return Outcome.Error(
                when {
                    // Transfers need the transfer station's board, so offline
                    // only direct trains can be planned. Say that rather than
                    // claiming no train goes there.
                    offline -> UserMessage.TimetableUnreachable
                    departures.isEmpty() -> UserMessage.NoTimetableData(from.name)
                    else -> UserMessage.NoConnection(from.name, to.name)
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
    private suspend fun directItinerary(
        stop: TimetableStop,
        from: Station,
        to: Station,
    ): Itinerary? {
        val departure = stop.departure?.plannedTime ?: return null
        val history = historyRepository.load(stop.label.category, stop.label.number)
        val destHistory = history?.stations?.entries
            ?.firstOrNull { (name, sh) -> sh.eva == to.eva || pathMatches(name, to.name) }
            ?.value ?: return null
        // From the same runs at both ends, not from the destination's latest
        // time of day: see CandidateBuilder.plannedArrival for what the latter
        // does to a train whose schedule has shifted since.
        val originHistory = history.stations.entries
            .firstOrNull { (name, sh) -> sh.eva == from.eva || pathMatches(name, from.name) }
            ?.value ?: return null
        val plannedArr = CandidateBuilder.plannedArrival(
            originHistory.runs, destHistory.runs, departure,
        ) ?: return null
        val forecast = predictor.forecast(
            history = history,
            stationEva = to.eva,
            stationName = to.name,
            trainCategory = stop.label.category,
            plannedTimeMillis = plannedArr,
            liveDelayMinutes = stop.departure?.liveDelayMinutes,
            lineHistory = {
                historyRepository.loadLine(
                    stop.label.category, stop.label.line, to.eva, history,
                )
            },
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
     * Every (feeder, transfer station) pair worth an attempt, best first.
     *
     * The shipped order used to be feeder by feeder in departure order, up to
     * two stations each, which meant the budget was consumed by the earliest
     * departures whether or not they led anywhere: at a hub the first four
     * trains out spent all eight attempts while forty others were never looked
     * at. Departure time is not evidence about whether a change works.
     *
     * Nothing here costs a request that the board has not already paid for —
     * a train's route comes with it, and [RouteStationMatcher.stationsOn]
     * resolves names against the bundled list before it considers asking IRIS.
     * So the pairs are free and only the *attempts* are dear, which is the
     * argument for choosing them globally.
     *
     * Ordered by distance from the transfer to the destination, exactly as
     * [transferCandidates] ranks within one train, with ties going to the
     * earliest feeder: arriving earlier at the same station can only see more
     * onward trains, never fewer. Pairs are kept rather than reduced to one per
     * station, because the caller retires a feeder once it has produced an
     * itinerary — a station whose earliest feeder has just been retired is
     * still worth reaching on the next train.
     *
     * Measured over five archived days and 4,000 journeys that provably have a
     * one-change connection, against the same budget: 82% found becomes 86%,
     * and at origins the size of a Hbf 57% becomes 73%. The itinerary found is
     * also 5 minutes earlier on average, because a change near the destination
     * is usually a change late in the journey.
     */
    private suspend fun transferPairs(
        feeders: List<TimetableStop>,
        from: Station,
        to: Station,
    ): List<Pair<TimetableStop, Station>> = rankTransferPairs(
        routes = feeders.map { it to routeStations.stationsOn(it.departure!!.plannedPath) },
        origin = from,
        destination = to,
    )

    /** One attempt: this feeder, this change. Costs a transfer board. */
    private suspend fun transferItinerary(
        stop: TimetableStop,
        transfer: Station,
        to: Station,
        transferMinutes: Int,
        deutschlandTicketOnly: Boolean,
    ): Itinerary? {
        val departure = stop.departure?.plannedTime ?: return null
        val outcome = connectionPlanner.plan(
            feeder = stop,
            transferQuery = transfer.name,
            destinationQuery = to.name,
            transferMinutes = transferMinutes,
            deutschlandTicketOnly = deutschlandTicketOnly,
            boardStartMillis = departure,
        ) as? ConnectionPlanner.Outcome.Success ?: return null
        return Itinerary(
            feeder = stop,
            departureMillis = departure,
            transferStation = outcome.transferStation.name,
            distribution = outcome.result.distribution,
            referenceArrivalMillis = outcome.result.referenceArrivalMillis,
            // The first *plannable* connection (departing after the feeder's
            // planned arrival) — normally-missed earlier trains are listed in
            // the details but don't define the headline.
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

    companion object {
        const val MAX_DIRECT = 3

        /** Hours of the origin's board that are searched for departures. */
        const val ORIGIN_HOURS = 3

        /**
         * Departures whose routes are read into [transferPairs].
         *
         * Reading a route is free — it arrives with the board — so this is not
         * a cost limit but a guard on the one thing that is not free:
         * [RouteStationMatcher.stationsOn] falls back to asking IRIS for a
         * route none of whose stations are in the bundled list, and that
         * fallback should not scale with the size of a Hbf's board.
         *
         * Raising it from 15 to 25 is worth 2 points of recall at the biggest
         * origins (71% to 73%); lifting it altogether is worth one more.
         */
        const val MAX_TRANSFER_SCAN = 25
        const val MAX_TRANSFER_RESULTS = 3

        /**
         * Cap on connection evaluations (each fetches a transfer board, which
         * is five IRIS requests).
         *
         * Measured with `tools/route_bench.py` against exhaustive ground truth
         * over five archived days: 77% are found within 4 attempts, 86% within
         * 8, 90% within 12, and 92% is the ceiling at any budget. The curve has
         * no elbow — it just flattens — so this constant trades recall against
         * requests rather than finding a natural stopping point, and eight is a
         * compromise, not an optimum.
         *
         * What changed with [transferPairs] is where those eight go, not how
         * many there are: at the shipped budget recall went from 82% to 86%
         * overall and from 57% to 73% at the biggest origins, at the same mean
         * spend of 5.8 attempts a search. It stays the binding constraint at
         * busy stations, just much less wastefully.
         *
         * An earlier measurement over 44 live journeys reported 98% at eight
         * attempts. Its ground truth came from walking the same boards this
         * search walks, so it could only pose journeys the heuristic already
         * finds; the number was an artefact of the harness.
         */
        const val MAX_TRANSFER_ATTEMPTS = 8

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
         * Merges the per-train rankings into one order across the whole board.
         *
         * Ordered by distance from the transfer to the destination, exactly as
         * [transferCandidates] ranks within one train, with ties going to the
         * earliest feeder: arriving earlier at the same station can only see
         * more onward trains, never fewer. A station with no coordinates keeps
         * its weight ordering and goes last, as it does within a route.
         *
         * Pairs are kept rather than reduced to one per station, because
         * [spendTransferBudget] retires a feeder once it has produced an
         * itinerary — a station whose earliest feeder has just been retired is
         * still worth reaching on the next train.
         */
        fun rankTransferPairs(
            routes: List<Pair<TimetableStop, List<Station>>>,
            origin: Station,
            destination: Station,
        ): List<Pair<TimetableStop, Station>> = routes
            .flatMap { (stop, path) ->
                transferCandidates(path, origin, destination).map { stop to it }
            }
            .sortedWith(
                compareBy(
                    { (_, transfer) -> transfer.distanceKm(destination) ?: Double.MAX_VALUE },
                    { (_, transfer) -> -transfer.weight },
                    { (stop, _) -> stop.departure?.plannedTime ?: Long.MAX_VALUE },
                ),
            )

        /**
         * Spends [MAX_TRANSFER_ATTEMPTS] on [pairs], best first, and stops at
         * [MAX_TRANSFER_RESULTS] itineraries.
         *
         * Two things are never spent twice. A transfer station is opened once,
         * because that is what an attempt buys — a board — and a second look at
         * the same board with a later feeder can only find the same trains from
         * further behind. And a feeder that has already produced an itinerary
         * is retired: a second change off the same train is not a second option
         * to the passenger, it is the same departure reached by another route.
         *
         * [evaluate] is the expensive part, and it is called at most
         * [MAX_TRANSFER_ATTEMPTS] times whatever the size of [pairs].
         */
        suspend fun spendTransferBudget(
            pairs: List<Pair<TimetableStop, Station>>,
            evaluate: suspend (TimetableStop, Station) -> Itinerary?,
        ): List<Itinerary> {
            val found = mutableListOf<Itinerary>()
            val openedBoards = mutableSetOf<String>()
            val servedFeeders = mutableSetOf<String>()
            var budget = MAX_TRANSFER_ATTEMPTS
            for ((stop, transfer) in pairs) {
                if (budget <= 0 || found.size >= MAX_TRANSFER_RESULTS) break
                if (transfer.name in openedBoards || stop.id in servedFeeders) continue
                openedBoards += transfer.name
                budget--
                val itinerary = evaluate(stop, transfer) ?: continue
                found += itinerary
                servedFeeders += stop.id
            }
            return found
        }

        /**
         * Transfer stations on one train's route worth an attempt, best first.
         *
         * [transferPairs] merges these per-train rankings into one order across
         * the whole board; this function decides only what is admissible and
         * how one route's stations compare.
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
        ): List<Station> {
            val usable = path.filter {
                it.eva != destination.eva && it.weight >= MIN_TRANSFER_WEIGHT
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
