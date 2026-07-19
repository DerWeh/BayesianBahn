package io.github.derweh.bayesianbahn

import io.github.derweh.bayesianbahn.data.HistoryRepository
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.OkHttpClient
import okhttp3.Request
import org.junit.Assume.assumeNotNull
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.time.OffsetDateTime
import java.util.zip.GZIPInputStream
import kotlin.math.abs

/**
 * Opt-in cross-check against DB's official journey planner (via the
 * community HAFAS proxy v6.db.transport.rest): for each leg of each
 * suggested journey, our shard's planned time at the involved stations must
 * match the navigator's planned times within a couple of minutes —
 * catching wrong-arrival-time bugs like a mismatched station block.
 *
 * Run with:  E2E=1 ./gradlew testDebugUnitTest --tests '*NavigatorCompareE2E*'
 * Skips silently when the proxy is unreachable (it often is).
 */
class NavigatorCompareE2E {

    private val client = OkHttpClient()
    private val shardHost = HistoryRepository.SHARD_URL

    @Test
    fun `shard planned times agree with the navigator`() {
        assumeNotNull(System.getenv("E2E"))
        val json = fetch(
            "https://v6.db.transport.rest/journeys?from=8000144&to=8000170&results=3",
        )
        assumeTrue("transport.rest unreachable", json != null)

        var checked = 0
        val journeys = Json.parseToJsonElement(json!!).jsonObject["journeys"]?.jsonArray
            ?: return
        for (journey in journeys) {
            for (leg in journey.jsonObject["legs"]?.jsonArray ?: continue) {
                val obj = leg.jsonObject
                if (obj["walking"]?.jsonPrimitive?.content == "true") continue
                val lineName = obj["line"]?.jsonObject?.get("name")
                    ?.jsonPrimitive?.content ?: continue
                val key = HistoryRepository.shardKey(lineName)
                val shard = fetchShard(key) ?: continue
                val history = HistoryRepository.parseShard(shard) ?: continue

                for ((field, stationField) in listOf(
                    "plannedDeparture" to "origin",
                    "plannedArrival" to "destination",
                )) {
                    val planned = obj[field]?.jsonPrimitive?.content ?: continue
                    val stationName = obj[stationField]?.jsonObject?.get("name")
                        ?.jsonPrimitive?.content ?: continue
                    val time = OffsetDateTime.parse(planned)
                    val tod = time.hour * 60 + time.minute
                    val runs = history.stations.entries
                        .firstOrNull { (n, _) -> n.equals(stationName, ignoreCase = true) }
                        ?.value?.runs ?: continue
                    val nearest = runs.mapNotNull { run ->
                        run.plannedTimeOfDay.split(':').takeIf { it.size == 2 }
                            ?.let { it[0].toInt() * 60 + it[1].toInt() }
                    }.minByOrNull { abs(it - tod) } ?: continue
                    val diff = abs(nearest - tod)
                    check(diff <= 3) {
                        "$lineName at $stationName: navigator $tod vs shard $nearest ($diff min off)"
                    }
                    checked++
                }
            }
        }
        println("navigator compare: $checked planned times cross-checked OK")
        assumeTrue("nothing comparable", checked > 0)
    }

    private fun fetch(url: String): String? = runCatching {
        client.newCall(Request.Builder().url(url).build()).execute().use { r ->
            if (!r.isSuccessful) return null
            r.body?.string()
        }
    }.getOrNull()

    private fun fetchShard(key: String): String? = runCatching {
        client.newCall(Request.Builder().url("$shardHost$key.jgz").build()).execute().use { r ->
            if (!r.isSuccessful) return null
            GZIPInputStream(ByteArrayInputStream(r.body!!.bytes()))
                .readBytes().decodeToString()
        }
    }.getOrNull()

}
