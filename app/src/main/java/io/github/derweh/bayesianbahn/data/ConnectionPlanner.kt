package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.model.ConnectionModel
import io.github.derweh.bayesianbahn.model.DeutschlandTicket
import java.time.LocalDate
import java.time.ZoneId

/**
 * Evaluates a connection: feeder train → transfer station → any train towards
 * the destination. Fetches the transfer station's live board once — it yields
 * both the feeder's arrival there and all candidate departures — then builds
 * the Bayesian mixture via [ConnectionModel].
 */
class ConnectionPlanner(
    private val stationRepository: StationRepository,
    private val historyRepository: HistoryRepository,
    private val irisClient: IrisClient,
    private val predictor: Predictor = Predictor(),
    /** Fallback for departure times beyond IRIS's plan horizon. */
    private val syntheticTimetable: SyntheticTimetable? = null,
    private val routeStations: RouteStationMatcher =
        RouteStationMatcher(irisClient, stationRepository),
) {

    sealed interface Outcome {
        data class Error(val message: UserMessage) : Outcome
        data class Success(
            val result: ConnectionModel.Result,
            val transferStation: Station,
            val destinationName: String,
            val feederForecast: Forecast,
            val feederPlannedArrivalMillis: Long,
        ) : Outcome
    }

    suspend fun plan(
        feeder: TimetableStop,
        transferQuery: String,
        destinationQuery: String,
        transferMinutes: Int,
        deutschlandTicketOnly: Boolean = false,
        boardStartMillis: Long? = null,
        today: LocalDate = LocalDate.now(ZONE),
    ): Outcome {
        // The transfer is usually named by IRIS (the chips on the connection
        // screen come straight from a train's route), so resolve it as a route
        // entry first and only then as free text the user typed.
        val transfer = routeStations.station(transferQuery)
            ?: stationRepository.search(transferQuery).firstOrNull()
            ?: return Outcome.Error(UserMessage.TransferStationNotFound(transferQuery))
        val destination = stationRepository.search(destinationQuery).firstOrNull()
        val destinationName = destination?.name ?: destinationQuery.trim()
        // Identity against IRIS's own spelling where the station is known,
        // canonicalised names otherwise.
        val isDestination: (String) -> Boolean = destination
            ?.let { routeStations.matcherFor(it) }
            ?: { entry -> StationNames.matches(entry, destinationName) }

        // As in JourneyPlanner: fall through to the historical timetable rather
        // than failing, so a connection can still be evaluated without a
        // network — blind to live delays, but evaluated.
        var offline = false
        var board = try {
            irisClient.board(transfer.eva, hours = 4, startMillis = boardStartMillis)
        } catch (e: Exception) {
            offline = true
            emptyList()
        }
        if (board.isEmpty() && syntheticTimetable != null && boardStartMillis != null) {
            // The feeder itself has to survive the filter: it is looked up on
            // this board by name below, and a Deutschland-Ticket search still
            // needs to find the train it is already sitting on.
            board = syntheticTimetable.board(transfer.eva, boardStartMillis, hours = 4) {
                !deutschlandTicketOnly || DeutschlandTicket.covers(it.category) ||
                    (it.category == feeder.label.category && it.number == feeder.label.number)
            }
        }
        if (board.isEmpty() && offline) {
            return Outcome.Error(UserMessage.TimetableUnreachable)
        }

        // The feeder's own stop at the transfer station.
        val feederThere = board.firstOrNull {
            it.label.category == feeder.label.category && it.label.number == feeder.label.number
        }
        val feederArrival = feederThere?.arrival
            ?: return Outcome.Error(
                UserMessage.FeederDoesNotReach(feeder.label.display, transfer.name),
            )
        val feederPlanned = feederArrival.plannedTime
            ?: return Outcome.Error(UserMessage.NoPlannedArrival(transfer.name))

        val feederHistory = historyRepository.load(
            feeder.label.category, feeder.label.number, feeder.label.line,
        )
        val feederForecast = predictor.forecast(
            history = feederHistory,
            stationEva = transfer.eva,
            stationName = transfer.name,
            trainCategory = feeder.label.category,
            plannedTimeMillis = feederPlanned,
            liveDelayMinutes = feederArrival.liveDelayMinutes,
            today = today,
        )

        val towardsDestination = board
            .asSequence()
            .filter { it.departure?.plannedTime != null }
            .filter { stop ->
                !(stop.label.category == feeder.label.category &&
                    stop.label.number == feeder.label.number)
            }
            .filter { stop -> stop.departure!!.plannedPath.any(isDestination) }
            .filter { !deutschlandTicketOnly || DeutschlandTicket.covers(it.label.category) }
            .sortedBy { it.departure!!.plannedTime }
            .toList()
        val candidates = pickCandidates(towardsDestination, feederPlanned) {
            it.departure!!.plannedTime!!
        }
        if (candidates.isEmpty()) {
            return Outcome.Error(
                UserMessage.NoTrainsTowards(destinationName, transfer.name, deutschlandTicketOnly),
            )
        }

        val modelCandidates = candidates.mapNotNull { stop ->
            buildCandidate(stop, transfer, destination, destinationName, today)
        }
        val result = ConnectionModel.propagate(
            feederArrival = feederForecast.distribution,
            feederPlannedArrivalMillis = feederPlanned,
            transferMinutes = transferMinutes,
            candidates = modelCandidates,
        ) ?: return Outcome.Error(UserMessage.NotEnoughHistory(destinationName))
        return Outcome.Success(result, transfer, destinationName, feederForecast, feederPlanned)
    }

    /** Joins a candidate's historical delays at the transfer and destination. */
    private suspend fun buildCandidate(
        stop: TimetableStop,
        transfer: Station,
        destination: Station?,
        destinationName: String,
        today: LocalDate,
    ): ConnectionModel.Candidate? {
        val departure = stop.departure ?: return null
        val plannedDep = departure.plannedTime ?: return null
        return CandidateBuilder.build(
            history = historyRepository.load(
                stop.label.category, stop.label.number, stop.label.line,
            ),
            id = stop.id,
            label = "${stop.label.display} → ${stop.destination ?: "?"}",
            transferEva = transfer.eva,
            transferName = transfer.name,
            destinationEva = destination?.eva,
            destinationName = destinationName,
            plannedDepartureMillis = plannedDep,
            liveDepartureDelay = departure.liveDelayMinutes,
            cancelledLive = departure.cancelled,
            today = today,
        )
    }

    companion object {
        private val ZONE = ZoneId.of("Europe/Berlin")
        const val MAX_CANDIDATES = 6

        /** How many already-departed trains may take up room in the list. */
        const val MAX_ALREADY_GONE = 2

        /** How far back a train counts as "just missed" rather than irrelevant. */
        const val LOOK_BACK_MINUTES = 30

        /**
         * The trains to offer for a change, newest missed ones first.
         *
         * A train leaving shortly before the feeder is due is worth showing:
         * it is usually missed, but a delayed one is sometimes exactly the
         * connection that works. Taking the first six by departure time let
         * those crowd out every train the passenger could actually catch —
         * where a station has a service every few minutes, all six were in the
         * past before the feeder even arrived, so the app offered six
         * impossible trains and no possible one. Measured over one collected
         * day, that was 31% of the journeys with a change.
         *
         * At most [MAX_ALREADY_GONE] of the list may be trains already gone,
         * and the rest is filled forward.
         */
        fun <T> pickCandidates(
            sorted: List<T>,
            feederPlannedMillis: Long,
            plannedTime: (T) -> Long,
        ): List<T> {
            val gone = sorted.filter {
                plannedTime(it) < feederPlannedMillis &&
                    plannedTime(it) >= feederPlannedMillis - LOOK_BACK_MINUTES * 60_000
            }
            val ahead = sorted.filter { plannedTime(it) >= feederPlannedMillis }
            // Room is reserved for the trains ahead first; whatever is left
            // over goes to the most recently missed ones. Late in the day there
            // may be only one train ahead, and then showing more of the missed
            // ones is better than showing a shorter list.
            val reserved = minOf(MAX_ALREADY_GONE, gone.size)
            val taken = ahead.take(MAX_CANDIDATES - reserved)
            return gone.takeLast(MAX_CANDIDATES - taken.size) + taken
        }

        // Moved to CandidateBuilder with the code that uses it, and re-exported
        // so callers and tests keep one name for one thing.
        const val MIN_JOINT_RUNS = CandidateBuilder.MIN_JOINT_RUNS
    }
}
