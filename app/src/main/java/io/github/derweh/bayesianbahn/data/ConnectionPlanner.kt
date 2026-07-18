package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.model.ConnectionModel
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
) {

    sealed interface Outcome {
        data class Error(val message: String) : Outcome
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
        today: LocalDate = LocalDate.now(ZONE),
    ): Outcome {
        val transfer = stationRepository.search(transferQuery).firstOrNull()
            ?: return Outcome.Error("Transfer station \"$transferQuery\" not found.")
        val destinationName = stationRepository.search(destinationQuery).firstOrNull()?.name
            ?: destinationQuery.trim()

        val board = try {
            irisClient.board(transfer.eva, hours = 4)
        } catch (e: Exception) {
            return Outcome.Error(e.message ?: "network error")
        }

        // The feeder's own stop at the transfer station.
        val feederThere = board.firstOrNull {
            it.label.category == feeder.label.category && it.label.number == feeder.label.number
        }
        val feederArrival = feederThere?.arrival
            ?: return Outcome.Error(
                "${feeder.label.display} does not reach ${transfer.name} within the next hours.",
            )
        val feederPlanned = feederArrival.plannedTime
            ?: return Outcome.Error("No planned arrival time at ${transfer.name}.")

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
            .filter { stop -> stop.departure!!.plannedPath.any { matches(it, destinationName) } }
            .filter { it.departure!!.plannedTime!! >= feederPlanned - 15 * 60_000 }
            .sortedBy { it.departure!!.plannedTime }
            .take(MAX_CANDIDATES)
            .toList()
        if (candidates.isEmpty()) {
            return Outcome.Error(
                "No trains towards $destinationName found at ${transfer.name} in the next hours.",
            )
        }

        val modelCandidates = candidates.mapNotNull { stop ->
            buildCandidate(stop, transfer, destinationName, today)
        }
        val result = ConnectionModel.propagate(
            feederArrival = feederForecast.distribution,
            feederPlannedArrivalMillis = feederPlanned,
            transferMinutes = transferMinutes,
            candidates = modelCandidates,
        ) ?: return Outcome.Error(
            "Not enough delay history for the trains towards $destinationName to " +
                "evaluate this connection.",
        )
        return Outcome.Success(result, transfer, destinationName, feederForecast, feederPlanned)
    }

    /** Joins a candidate's historical delays at the transfer and destination. */
    private suspend fun buildCandidate(
        stop: TimetableStop,
        transfer: Station,
        destinationName: String,
        today: LocalDate,
    ): ConnectionModel.Candidate? {
        val departure = stop.departure ?: return null
        val plannedDep = departure.plannedTime ?: return null
        val live = departure.liveDelayMinutes
        val cancelledLive = departure.cancelled

        val history = historyRepository.load(stop.label.category, stop.label.number, stop.label.line)
        val transferHistory = history?.stations?.entries?.firstOrNull { (name, sh) ->
            sh.eva == transfer.eva || name.equals(transfer.name, ignoreCase = true)
        }?.value
        val destinationHistory = history?.stations?.entries?.firstOrNull { (name, _) ->
            matches(name, destinationName)
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

        private fun matches(pathStation: String, destination: String): Boolean =
            pathStation.equals(destination, ignoreCase = true) ||
                pathStation.contains(destination, ignoreCase = true)

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
