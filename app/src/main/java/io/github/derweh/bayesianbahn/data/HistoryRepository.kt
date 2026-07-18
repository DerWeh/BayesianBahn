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
 * Draft: shards are bundled as assets under `history/`; a remote shard host
 * with monthly updates replaces this later (same file format).
 */
class HistoryRepository(private val context: Context) {

    /**
     * Finds the shard for a train, trying category+number ("ICE 512") first
     * and category+line ("RE 9") second, matching the pipeline's naming.
     */
    suspend fun load(category: String, number: String, line: String?): TrainHistory? =
        withContext(Dispatchers.IO) {
            candidateKeys(category, number, line).firstNotNullOfOrNull { key ->
                readShard(key)
            }
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
        /** Mirrors `train_key` in build_shards.py. */
        fun shardKey(trainName: String): String =
            trainName.trim().replace(Regex("[^A-Za-z0-9]+"), "_").trim('_').uppercase()

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
                        runs = obj["runs"]?.jsonArray?.mapNotNull { parseRun(it) } ?: emptyList(),
                    )
                },
            )
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
