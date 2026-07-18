package io.github.derweh.bayesianbahn.api

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
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
     * Fetches the merged real-time board for a station: planned stops for the
     * next [hours] hour slices with all currently known changes applied.
     */
    suspend fun board(eva: String, hours: Int = 2): List<TimetableStop> = coroutineScope {
        val now = ZonedDateTime.now(ZONE)
        val plans = (0 until hours).map { offset ->
            async { fetchPlan(eva, now.plusHours(offset.toLong())) }
        }
        val changes = async { fetchChanges(eva) }
        val stops = plans.flatMap { it.await() }.distinctBy { it.id }
        IrisParser.merge(stops, changes.await())
    }

    private suspend fun fetchPlan(eva: String, slice: ZonedDateTime): List<TimetableStop> {
        val date = slice.format(DateTimeFormatter.ofPattern("yyMMdd"))
        val hour = slice.format(DateTimeFormatter.ofPattern("HH"))
        return parser.parsePlan(get("$baseUrl/plan/$eva/$date/$hour"))
    }

    private suspend fun fetchChanges(eva: String): Map<String, StopChange> =
        parser.parseChanges(get("$baseUrl/fchg/$eva"))

    private suspend fun get(url: String): String = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url(url)
            .header("User-Agent", "BayesianBahn/0.1 (F-Droid; FOSS delay prediction)")
            .build()
        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("HTTP ${response.code} for $url")
            response.body?.string() ?: throw IOException("Empty body for $url")
        }
    }

    private companion object {
        val ZONE: ZoneId = ZoneId.of("Europe/Berlin")
    }
}
