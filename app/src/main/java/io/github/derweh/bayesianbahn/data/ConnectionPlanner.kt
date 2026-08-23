package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.model.ConnectionModel
import io.github.derweh.bayesianbahn.model.DeutschlandTicket
import io.github.derweh.bayesianbahn.model.EmpiricalDelay
import java.time.Instant
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

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

        val candidates = board
            .asSequence()
            .filter { it.departure?.plannedTime != null }
            .filter { stop ->
                !(stop.label.category == feeder.label.category &&
                    stop.label.number == feeder.label.number)
            }
            .filter { stop -> stop.departure!!.plannedPath.any(isDestination) }
            .filter { !deutschlandTicketOnly || DeutschlandTicket.covers(it.label.category) }
            // Include trains departing up to 30 min before the feeder's planned
            // arrival: usually missed (shown near 0%), but visible — and a
            // delayed one is sometimes exactly the connection that works.
            .filter { it.departure!!.plannedTime!! >= feederPlanned - 30 * 60_000 }
            .sortedBy { it.departure!!.plannedTime }
            .take(MAX_CANDIDATES)
            .toList()
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
        val live = departure.liveDelayMinutes
        val cancelledLive = departure.cancelled

        val history = historyRepository.load(stop.label.category, stop.label.number, stop.label.line)
        val transferHistory = history?.stations?.entries?.firstOrNull { (name, sh) ->
            sh.eva == transfer.eva || StationNames.matches(name, transfer.name)
        }?.value
        // Shards carry the eva, so prefer it and fall back to the name only
        // for entries that predate it.
        val destinationHistory = history?.stations?.entries?.firstOrNull { (name, sh) ->
            (destination != null && sh.eva == destination.eva) ||
                StationNames.matches(name, destinationName)
        }?.value

        val depHhmm = Instant.ofEpochMilli(plannedDep).atZone(ZONE).format(HHMM)
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
        if (joint.size < MIN_JOINT_RUNS && live == null && !cancelledLive) return null

        // Planned arrival at the destination is not part of the IRIS board;
        // recover it from the most recent historical run's planned time of day.
        val arrivalTod = arrivalByDate.entries.maxByOrNull { it.key }?.value?.plannedTimeOfDay
            ?: return null
        val plannedArr = arrivalMillis(plannedDep, arrivalTod) ?: return null

        return ConnectionModel.Candidate(
            id = stop.id,
            label = "${stop.label.display} → ${stop.destination ?: "?"}",
            plannedDepartureMillis = plannedDep,
            plannedArrivalMillis = plannedArr,
            runs = joint,
            liveDepartureDelay = live,
            cancelledLive = cancelledLive,
            cancelRate = cancelRate,
        )
    }

    companion object {
        private val ZONE = ZoneId.of("Europe/Berlin")
        private val HHMM = DateTimeFormatter.ofPattern("HH:mm")
        const val MAX_CANDIDATES = 6
        const val MIN_JOINT_RUNS = 5

        /** Absolute planned arrival: departure date + arrival time of day (may wrap midnight). */
        fun arrivalMillis(plannedDepMillis: Long, arrivalHhmm: String): Long? {
            val tod = runCatching { LocalTime.parse(arrivalHhmm, HHMM) }.getOrNull() ?: return null
            val dep = Instant.ofEpochMilli(plannedDepMillis).atZone(ZONE)
            var arr = dep.with(tod)
            if (arr.isBefore(dep)) arr = arr.plusDays(1)
            return arr.toInstant().toEpochMilli()
        }
    }
}
