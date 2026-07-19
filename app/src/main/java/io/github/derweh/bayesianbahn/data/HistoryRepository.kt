package io.github.derweh.bayesianbahn.data

import android.content.Context
import io.github.derweh.bayesianbahn.model.HistoricalRun
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.intOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.time.LocalDate
import java.util.zip.GZIPInputStream

/** Delay history of one train identity across all stations it calls at. */
data class TrainHistory(
    val trainName: String,
    val trainType: String,
    /** Station name → historical runs. */
    val stations: Map<String, StationHistory>,
)

data class StationHistory(val eva: String?, val runs: List<HistoricalRun>)

/**
 * Loads per-train history shards produced by `pipeline/build_shards.py`.
 *
 * Lookup order: downloaded base + recent overlay (or the bundled snapshot),
 * then an *on-demand* fetch from the repo's `shards` branch for trains
 * outside the local data — cached on disk, so a commuter's usual
 * connections are fetched once and refreshed at most daily.
 */
class HistoryRepository(
    private val context: Context,
    private val client: OkHttpClient = OkHttpClient(),
    private val shardUrl: String = SHARD_URL,
    private val recentShardUrl: String = RECENT_SHARD_URL,
) {

    /**
     * Finds the shard for a train, trying category+number ("ICE 512") first
     * and category+line ("RE 9") second, matching the pipeline's naming.
     */
    suspend fun load(category: String, number: String, line: String?): TrainHistory? =
        withContext(Dispatchers.IO) {
            val keys = candidateKeys(category, number, line)
            keys.firstNotNullOfOrNull { readShard(it) }
                ?: keys.firstNotNullOfOrNull { onDemand(it) }
        }

    /**
     * Cached network fetch of one train's history: the country-wide base
     * shard (rebuilt monthly, cached a week) merged with its small recent
     * overlay (rebuilt daily, cached [TTL_MILLIS]).
     */
    private fun onDemand(key: String): TrainHistory? {
        val base = fetchCached(ONDEMAND_DIR, "$shardUrl$key.jgz", key, BASE_TTL_MILLIS)
        val recent = fetchCached(
            "$ONDEMAND_DIR-recent", "$recentShardUrl$key.jgz", key, TTL_MILLIS,
        )
        return mergeHistories(base, recent)
    }

    private fun fetchCached(dirName: String, url: String, key: String, ttl: Long): TrainHistory? {
        val dir = File(context.filesDir, dirName).apply { mkdirs() }
        val cached = File(dir, "$key.jgz")
        val miss = File(dir, "$key.miss")
        val now = System.currentTimeMillis()
        fun fresh(f: File) = f.isFile && now - f.lastModified() < ttl

        if (fresh(cached)) return readFile(cached)
        if (fresh(miss)) return null
        try {
            val request = Request.Builder().url(url).build()
            client.newCall(request).execute().use { response ->
                when {
                    response.isSuccessful -> {
                        cached.writeBytes(response.body!!.bytes())
                        miss.delete()
                    }
                    response.code == 404 -> {
                        // Unknown train: remember, don't re-ask every query.
                        miss.writeBytes(ByteArray(0))
                        cached.delete()
                        return null
                    }
                    else -> throw IOException("HTTP ${response.code}")
                }
            }
        } catch (_: IOException) {
            // Offline or flaky: a stale cached shard beats nothing.
        }
        return readFile(cached)
    }

    private fun candidateKeys(category: String, number: String, line: String?): List<String> {
        val keys = mutableListOf<String>()
        if (number.isNotBlank()) keys += shardKey("$category $number")
        if (line != null && line.isNotBlank()) {
            keys += shardKey(if (line.startsWith(category)) line else "$category $line")
        }
        return keys.distinct()
    }

    private fun readShard(key: String): TrainHistory? {
        // Base data: downloaded monthly build if present, else the bundled
        // snapshot; the small daily "recent" overlay is merged on top.
        val base = readFile(File(DataUpdater.baseDir(context), "$key.jgz"))
            ?: readAsset(key)
        val recent = readFile(File(DataUpdater.recentDir(context), "$key.jgz"))
        return mergeHistories(base, recent)
    }

    private fun readFile(file: File): TrainHistory? {
        if (!file.isFile) return null
        val bytes = try {
            file.inputStream().use { GZIPInputStream(it).readBytes() }
        } catch (_: IOException) {
            return null
        }
        return parseShard(bytes.decodeToString())
    }

    private fun readAsset(key: String): TrainHistory? {
        val bytes = try {
            // .jgz, not .json.gz: aapt silently gunzips and renames *.gz
            // assets, which would break the lookup and the F-Droid build.
            context.assets.open("history/$key.jgz").use { stream ->
                GZIPInputStream(stream).readBytes()
            }
        } catch (_: IOException) {
            return null
        }
        return parseShard(bytes.decodeToString())
    }

    companion object {
        /** Country-wide base shards: `shards` branch, rebuilt monthly. */
        const val SHARD_URL =
            "https://raw.githubusercontent.com/DerWeh/BayesianBahn/refs/heads/shards/"

        /** Small recent-days overlays: `shards-recent` branch, rebuilt daily. */
        const val RECENT_SHARD_URL =
            "https://raw.githubusercontent.com/DerWeh/BayesianBahn/refs/heads/shards-recent/"
        const val ONDEMAND_DIR = "ondemand"

        /** Recent overlays are refreshed at most this often. */
        const val TTL_MILLIS = 18 * 60 * 60 * 1000L

        /** Base shards change monthly; a week of cache is plenty fresh. */
        const val BASE_TTL_MILLIS = 7 * 24 * 60 * 60 * 1000L

        /** Mirrors `train_key` in build_shards.py. */
        fun shardKey(trainName: String): String =
            trainName.trim().replace(Regex("[^A-Za-z0-9]+"), "_").trim('_').uppercase()

        /**
         * Overlays [recent] runs onto [base]; where both cover the same
         * (date, planned time) at a station, the recent run wins — it was
         * built from fresher raw data.
         */
        fun mergeHistories(base: TrainHistory?, recent: TrainHistory?): TrainHistory? {
            if (base == null) return recent
            if (recent == null) return base
            val stations = (base.stations.keys + recent.stations.keys).associateWith { name ->
                val b = base.stations[name]
                val r = recent.stations[name]
                when {
                    b == null -> r!!
                    r == null -> b
                    else -> {
                        val covered = r.runs.mapTo(HashSet()) { it.date to it.plannedTimeOfDay }
                        StationHistory(
                            eva = b.eva ?: r.eva,
                            runs = b.runs.filter { (it.date to it.plannedTimeOfDay) !in covered } +
                                r.runs,
                        )
                    }
                }
            }
            return TrainHistory(base.trainName, base.trainType, stations)
        }

        fun parseShard(json: String): TrainHistory? {
            val root = runCatching { Json.parseToJsonElement(json).jsonObject }.getOrNull()
                ?: return null
            val stations = root["stations"]?.jsonObject ?: return null
            return TrainHistory(
                trainName = root["train"]?.jsonPrimitive?.content ?: "?",
                trainType = root["type"]?.jsonPrimitive?.content ?: "?",
                stations = stations.entries.associate { (name, value) ->
                    val obj = value.jsonObject
                    name to StationHistory(
                        eva = obj["eva"]?.jsonPrimitive?.content,
                        runs = if ("days" in obj) parseColumnarRuns(obj) else {
                            obj["runs"]?.jsonArray?.mapNotNull { parseRun(it) } ?: emptyList()
                        },
                    )
                },
            )
        }

        /**
         * v2 columnar station block: delta-coded epoch days, deduplicated
         * planned times, arrival/prev arrays, sparse departure ("d" null or
         * absent means "same as arrival" — the consumers fall back to the
         * arrival delay either way) and cancelled indices.
         */
        private fun parseColumnarRuns(
            obj: kotlinx.serialization.json.JsonObject,
        ): List<HistoricalRun> {
            val days = obj["days"]?.jsonArray ?: return emptyList()
            val tods = obj["tod"]?.jsonArray?.map { it.jsonPrimitive.intOrNull ?: 0 }
                ?: return emptyList()
            val t = obj["t"]?.jsonArray
            val a = obj["a"]?.jsonArray
            val d = obj["d"]?.jsonArray
            val p = obj["p"]?.jsonArray
            val cancelled = obj["c"]?.jsonArray
                ?.mapNotNullTo(HashSet()) { it.jsonPrimitive.intOrNull } ?: emptySet()

            fun int(arr: kotlinx.serialization.json.JsonArray?, i: Int): Int? =
                arr?.getOrNull(i)?.jsonPrimitive?.intOrNull

            var epochDay = 0L
            return List(days.size) { i ->
                epochDay += days[i].jsonPrimitive.intOrNull?.toLong() ?: 0L
                val tod = tods.getOrElse(int(t, i) ?: 0) { 0 }
                val arr = int(a, i)
                HistoricalRun(
                    date = LocalDate.ofEpochDay(epochDay),
                    plannedTimeOfDay = "%02d:%02d".format(tod / 60, tod % 60),
                    arrivalDelay = arr,
                    departureDelay = int(d, i) ?: arr,
                    previousStopDelay = int(p, i),
                    cancelled = i in cancelled,
                )
            }
        }

        private fun parseRun(element: kotlinx.serialization.json.JsonElement): HistoricalRun? {
            val arr = runCatching { element.jsonArray }.getOrNull() ?: return null
            if (arr.size < 6) return null
            fun int(i: Int): Int? = arr[i].jsonPrimitive.intOrNull
            val date = runCatching { LocalDate.parse(arr[0].jsonPrimitive.content) }.getOrNull()
                ?: return null
            return HistoricalRun(
                date = date,
                plannedTimeOfDay = arr[1].jsonPrimitive.content,
                arrivalDelay = int(2),
                departureDelay = int(3),
                previousStopDelay = int(4),
                cancelled = int(5) == 1,
            )
        }
    }
}
