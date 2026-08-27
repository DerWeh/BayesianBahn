package io.github.derweh.bayesianbahn.data

import android.content.Context
import io.github.derweh.bayesianbahn.model.HistoricalRun
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
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
 * The one thing the planners need from history. Narrowing it to this lets them
 * be tested without an Android `Context`, which is why [SyntheticTimetable]
 * takes this rather than the repository itself.
 */
fun interface HistorySource {
    suspend fun load(category: String, number: String, line: String?): TrainHistory?
}

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
    private val cache: HistoryCache = HistoryCache(),
) : HistorySource {

    /**
     * Finds the shard for a train, trying category+number ("ICE 512") first
     * and category+line ("RE 9") second, matching the pipeline's naming.
     *
     * Repeats within a session are served from [cache]; see there for why one
     * search asks for the same train many times.
     */
    override suspend fun load(category: String, number: String, line: String?): TrainHistory? {
        val keys = candidateKeys(category, number, line)
        if (keys.isEmpty()) return null
        // shardKey() maps every key to [A-Z0-9_], so "|" cannot occur inside
        // one and the joined string identifies the candidate list uniquely.
        return cache.get(keys.joinToString("|")) {
            withContext(Dispatchers.IO) {
                keys.firstNotNullOfOrNull { readShard(it) }
                    ?: keys.firstNotNullOfOrNull { onDemand(it) }
            }
        }
    }

    /**
     * Drops the in-memory memo. Call after the downloaded history changes —
     * otherwise a refresh reports new data while the planners keep answering
     * from what it replaced.
     */
    fun invalidate() = cache.invalidate()

    private val fetcher = CachedFetcher(context, client)

    /**
     * Cached network fetch of one train's history: the country-wide base
     * shard (rebuilt monthly, cached a week) merged with its small recent
     * overlay (rebuilt daily, cached [TTL_MILLIS]).
     */
    private fun onDemand(key: String): TrainHistory? {
        val base = fetcher.bytes(ONDEMAND_DIR, key, "$shardUrl$key.jgz", BASE_TTL_MILLIS)
            ?.let { parseShard(it.decodeToString()) }
        val recent = fetcher
            .bytes("$ONDEMAND_DIR-recent", key, "$recentShardUrl$key.jgz", TTL_MILLIS)
            ?.let { parseShard(it.decodeToString()) }
        return mergeHistories(base, recent)
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

        /**
         * Decoded straight into typed fields rather than walked as a JSON tree.
         *
         * The tree route spent its time in `jsonPrimitive.intOrNull`, which is
         * a *string* parse per field per run: a median shard holds 900 runs of
         * five fields, so one train cost several thousand `String.toInt` calls
         * and the allocations under them. Building the tree itself was never
         * the expensive part — 88 ms against 1,424 ms for the whole parse over
         * 600 shards. The decoder reads the integers out of the character
         * stream once.
         *
         * Every field is optional with a default: a station block omits `t`
         * when all its runs share one planned time, `c` when nothing was
         * cancelled, and `d`/`p` whenever they would be all-null.
         */
        @Serializable
        private data class ShardDto(
            val train: String = "?",
            val type: String = "?",
            val stations: Map<String, StationDto> = emptyMap(),
        )

        @Serializable
        private data class StationDto(
            val eva: String? = null,
            /** Distinct planned times, minutes past midnight; `t` indexes it. */
            val tod: List<Int> = emptyList(),
            /** Delta-coded epoch days: the first is absolute, the rest are steps. */
            val days: List<Int> = emptyList(),
            val t: List<Int>? = null,
            val a: List<Int?>? = null,
            val d: List<Int?>? = null,
            val p: List<Int?>? = null,
            val c: List<Int>? = null,
            /** v1: one array per run. No shard has been written this way since
             *  the columnar format landed, but old caches on disk outlive it. */
            val runs: List<JsonElement>? = null,
        )

        private val decoder = Json { ignoreUnknownKeys = true; isLenient = true }

        fun parseShard(json: String): TrainHistory? {
            val root = runCatching { decoder.decodeFromString<ShardDto>(json) }.getOrNull()
                ?: return null
            if (root.stations.isEmpty()) {
                // An empty object is a shard with no stations; anything that
                // failed to look like one at all is not a shard.
                if (!json.contains("\"stations\"")) return null
            }
            return TrainHistory(
                trainName = root.train,
                trainType = root.type,
                stations = root.stations.mapValues { (_, station) ->
                    StationHistory(
                        eva = station.eva,
                        runs = if (station.days.isNotEmpty()) columnarRuns(station)
                        else station.runs?.mapNotNull { parseRun(it) } ?: emptyList(),
                    )
                },
            )
        }

        /** "HH:mm" without `String.format`, which cost 259 ms per 750,000 runs. */
        private fun hhmm(minutes: Int): String {
            val h = minutes / 60
            val m = minutes % 60
            val out = CharArray(5)
            out[0] = ('0' + h / 10); out[1] = ('0' + h % 10)
            out[2] = ':'
            out[3] = ('0' + m / 10); out[4] = ('0' + m % 10)
            return String(out)
        }

        /**
         * v2 columnar station block: delta-coded epoch days, deduplicated
         * planned times, arrival/prev arrays, sparse departure ("d" null or
         * absent means "same as arrival" — the consumers fall back to the
         * arrival delay either way) and cancelled indices.
         */
        private fun columnarRuns(station: StationDto): List<HistoricalRun> {
            if (station.tod.isEmpty()) return emptyList()
            val cancelled = station.c?.toHashSet()
            var epochDay = 0L
            return List(station.days.size) { i ->
                epochDay += station.days[i]
                val arrival = station.a?.getOrNull(i)
                HistoricalRun(
                    date = LocalDate.ofEpochDay(epochDay),
                    plannedTimeOfDay = hhmm(
                        station.tod.getOrElse(station.t?.getOrNull(i) ?: 0) { 0 },
                    ),
                    arrivalDelay = arrival,
                    departureDelay = station.d?.getOrNull(i) ?: arrival,
                    previousStopDelay = station.p?.getOrNull(i),
                    cancelled = cancelled != null && i in cancelled,
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
