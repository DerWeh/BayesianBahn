package io.github.derweh.bayesianbahn.api

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.net.URLEncoder
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit

/**
 * Client for the keyless IRIS timetable API that also powers DB's own
 * station departure boards. Planned data is served per station and hour,
 * real-time changes as one document per station.
 */
class IrisClient(
    private val parser: IrisParser,
    private val baseUrl: String = "https://iris.noncd.db.de/iris-tts/timetable",
    client: OkHttpClient? = null,
) {
    private val http = client ?: OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .build()

    /**
     * Fetches the merged real-time board for a station: planned stops for
     * [hours] hour slices starting at [startMillis] (default: now — IRIS
     * serves plan data days ahead, so future trips work too) with all
     * currently known changes applied.
     *
     * On [Dispatchers.Default] as a whole, not just around the requests. Only
     * the requests used to leave the caller's thread, and parsing is the
     * larger half: `fchg` for Ulm Hbf is 168 KB against 10.6 KB for an hour of
     * `plan`, because a station's disruption messages are most of the
     * document. Every `viewModelScope.launch` that reaches this names no
     * dispatcher, so all of that ran on the main thread and a journey search —
     * which opens several boards — froze the spinner it had just started.
     */
    suspend fun board(
        eva: String,
        hours: Int = 2,
        startMillis: Long? = null,
    ): List<TimetableStop> = withContext(Dispatchers.Default) {
        coroutineScope {
            val start = startMillis
                ?.let { ZonedDateTime.ofInstant(java.time.Instant.ofEpochMilli(it), ZONE) }
                ?: ZonedDateTime.now(ZONE)
            val plans = (0 until hours).map { offset ->
                async { fetchPlan(eva, start.plusHours(offset.toLong())) }
            }
            val changes = async { fetchChanges(eva) }
            val stops = plans.flatMap { it.await() }.distinctBy { it.id }
            IrisParser.merge(stops, changes.await())
        }
    }

    /**
     * IRIS's own name for a station, which is what its route lists (`ppth`)
     * contain — "Türkheim(Bay)Bf" where the bundled list says "Türkheim (Bay)
     * Bahnhof". Resolving the name once per destination turns route matching
     * from a spelling comparison into an identity comparison.
     *
     * Null when the station is unknown or IRIS cannot be reached; callers fall
     * back to comparing names.
     */
    suspend fun stationName(eva: String): String? {
        val wanted = eva.trimStart('0')
        return stations(eva).firstOrNull { it.eva.trimStart('0') == wanted }?.name
    }

    /**
     * The EVA number IRIS gives a station it names [name] — the inverse lookup,
     * for turning a route entry into a station we know.
     */
    suspend fun stationEva(name: String): String? {
        val found = stations(name)
        return (found.firstOrNull { it.name.equals(name, ignoreCase = true) }
            ?: found.singleOrNull())?.eva
    }

    /** Raw `station/{query}` lookup; the query may be an EVA number or a name. */
    suspend fun stations(query: String): List<IrisStation> =
        withContext(Dispatchers.Default) {
            val encoded = URLEncoder.encode(query, "UTF-8").replace("+", "%20")
            val xml = get("$baseUrl/station/$encoded", notFoundAsEmpty = true)
                ?: return@withContext emptyList()
            parser.parseStations(xml)
        }

    private suspend fun fetchPlan(eva: String, slice: ZonedDateTime): List<TimetableStop> {
        val date = slice.format(DateTimeFormatter.ofPattern("yyMMdd"))
        val hour = slice.format(DateTimeFormatter.ofPattern("HH"))
        // IRIS publishes plan data only ~a day ahead; a missing future slice
        // is not an error, just an empty board.
        val xml = get("$baseUrl/plan/$eva/$date/$hour", notFoundAsEmpty = true) ?: return emptyList()
        return parser.parsePlan(xml)
    }

    private suspend fun fetchChanges(eva: String): Map<String, StopChange> =
        parser.parseChanges(get("$baseUrl/fchg/$eva")!!)

    private suspend fun get(url: String, notFoundAsEmpty: Boolean = false): String? =
        withContext(Dispatchers.IO) {
            val request = Request.Builder()
                .url(url)
                .header("User-Agent", "BayesianBahn/0.1 (F-Droid; FOSS delay prediction)")
                .build()
            http.newCall(request).execute().use { response ->
                if (notFoundAsEmpty && response.code == 404) return@withContext null
                if (!response.isSuccessful) throw IOException("HTTP ${response.code} for $url")
                response.body?.string() ?: throw IOException("Empty body for $url")
            }
        }

    private companion object {
        val ZONE: ZoneId = ZoneId.of("Europe/Berlin")
    }
}
