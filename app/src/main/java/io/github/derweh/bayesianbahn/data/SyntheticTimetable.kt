package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.EventInfo
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.api.TrainLabel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

/**
 * Reconstructs a station board for dates beyond IRIS's ~1-day plan horizon
 * from per-station history boards (`pipeline/build_boards.py`): every train
 * that recently called there, with typical times and a weekday pattern.
 * Routes come from the trains' own shards. No live data exists that far
 * ahead, so predictions run blind — and the UI says so.
 */
class SyntheticTimetable(
    private val fetcher: CachedFetcher,
    private val historyRepository: HistoryRepository,
    private val boardUrl: String = BOARD_URL,
) {

    /** One train from a station's history board. */
    data class Entry(
        val label: TrainLabel,
        val arrTod: Int?,
        val depTod: Int?,
        val weekdayMask: Int,
        val lastSeenEpochDay: Long,
    )

    suspend fun board(eva: String, startMillis: Long, hours: Int): List<TimetableStop> =
        withContext(Dispatchers.IO) {
            val bytes = fetcher.bytes("boards", eva, "$boardUrl$eva.jgz", BOARD_TTL_MILLIS)
                ?: return@withContext emptyList()
            val entries = parseBoard(bytes.decodeToString())
            val stops = plan(entries, startMillis, hours)
            // Attach onward routes from the trains' shards so destination
            // matching and transfer picking work as with a live board.
            stops.take(MAX_STOPS).mapNotNull { (stop, depTod) ->
                val history = historyRepository.load(
                    stop.label.category, stop.label.number, stop.label.line,
                )
                val path = pathAfter(history, eva, depTod ?: return@mapNotNull stop to null)
                stop to path
            }.map { (stop, path) ->
                if (path == null || stop.departure == null) stop
                else stop.copy(departure = stop.departure!!.copy(plannedPath = path))
            }
        }

    companion object {
        const val BOARD_URL =
            "https://raw.githubusercontent.com/DerWeh/BayesianBahn/refs/heads/shards/boards/"
        const val BOARD_TTL_MILLIS = 7 * 24 * 60 * 60 * 1000L

        /** Trains discontinued this long before today are dropped. */
        const val MAX_LAST_SEEN_AGE_DAYS = 60L

        /** Bound on shard fetches per synthetic board. */
        const val MAX_STOPS = 40

        /** A stop this many minutes after the reference is "later on the route". */
        const val ROUTE_WINDOW_MIN = 600

        private val ZONE = ZoneId.of("Europe/Berlin")

        fun parseBoard(json: String): List<Entry> {
            val root = runCatching { Json.parseToJsonElement(json).jsonObject }.getOrNull()
                ?: return emptyList()
            return root["trains"]?.jsonArray?.mapNotNull { el ->
                val arr = runCatching { el.jsonArray }.getOrNull() ?: return@mapNotNull null
                if (arr.size < 7) return@mapNotNull null
                Entry(
                    label = TrainLabel(
                        category = arr[0].jsonPrimitive.content,
                        number = arr[1].jsonPrimitive.content,
                        line = arr[2].jsonPrimitive.contentOrNull,
                    ),
                    arrTod = arr[3].jsonPrimitive.intOrNull,
                    depTod = arr[4].jsonPrimitive.intOrNull,
                    weekdayMask = arr[5].jsonPrimitive.intOrNull ?: 0,
                    lastSeenEpochDay = arr[6].jsonPrimitive.intOrNull?.toLong() ?: 0L,
                )
            } ?: emptyList()
        }

        /** Board entries expanded onto concrete dates within the window. */
        fun plan(
            entries: List<Entry>,
            startMillis: Long,
            hours: Int,
            today: LocalDate = LocalDate.now(ZONE),
        ): List<Pair<TimetableStop, Int?>> {
            val start = Instant.ofEpochMilli(startMillis).atZone(ZONE)
            val endMillis = startMillis + hours * 3_600_000L
            val recentCutoff = today.toEpochDay() - MAX_LAST_SEEN_AGE_DAYS
            val stops = mutableListOf<Pair<TimetableStop, Int?>>()
            // The window may cross midnight: try the start date and the next.
            for (dayOffset in 0..1) {
                val date = start.toLocalDate().plusDays(dayOffset.toLong())
                val weekdayBit = 1 shl (date.dayOfWeek.value - 1)
                for (entry in entries) {
                    if (entry.weekdayMask and weekdayBit == 0) continue
                    if (entry.lastSeenEpochDay < recentCutoff) continue
                    fun at(tod: Int?): Long? = tod?.let {
                        date.atStartOfDay(ZONE).plusMinutes(it.toLong())
                            .toInstant().toEpochMilli()
                    }
                    val dep = at(entry.depTod)
                    val arrv = at(entry.arrTod)
                    val anchor = dep ?: arrv ?: continue
                    if (anchor < startMillis || anchor >= endMillis) continue
                    stops += TimetableStop(
                        id = "syn-${entry.label.category}-${entry.label.number}-$date",
                        label = entry.label,
                        arrival = arrv?.let {
                            EventInfo(it, null, null, null, emptyList(), false)
                        },
                        departure = dep?.let {
                            EventInfo(it, null, null, null, emptyList(), false)
                        },
                    ) to entry.depTod
                }
            }
            return stops
                .distinctBy { it.first.id }
                .sortedBy { it.first.departure?.plannedTime ?: it.first.arrival?.plannedTime }
        }

        /**
         * Onward route of a train after the stop at [originTod] minutes:
         * the shard's stations ordered by how soon after the reference
         * planned time they are reached (within [ROUTE_WINDOW_MIN]).
         */
        fun pathAfter(history: TrainHistory?, originEva: String, originTod: Int): List<String> {
            if (history == null) return emptyList()
            return history.stations.entries
                .filter { (_, sh) -> sh.eva != originEva }
                .mapNotNull { (name, sh) ->
                    val diff = sh.runs.asSequence()
                        .mapNotNull { toMinutes(it.plannedTimeOfDay) }
                        .map { (it - originTod + 1440) % 1440 }
                        .filter { it in 1 until ROUTE_WINDOW_MIN }
                        .minOrNull() ?: return@mapNotNull null
                    name to diff
                }
                .sortedBy { it.second }
                .map { it.first }
        }

        private fun toMinutes(hhmm: String): Int? {
            val parts = hhmm.split(':')
            if (parts.size != 2) return null
            val h = parts[0].toIntOrNull() ?: return null
            val m = parts[1].toIntOrNull() ?: return null
            return h * 60 + m
        }
    }
}
