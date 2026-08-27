package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.HistoricalRun
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.time.LocalDate
import java.util.zip.GZIPInputStream

/**
 * The rewritten shard parser against the one it replaced, on real shards.
 *
 * The rewrite was for speed — a JSON tree walked with `jsonPrimitive.intOrNull`
 * is a string parse per field per run, and a median shard holds nine hundred
 * runs — so the only thing that matters is that it changed nothing else. The
 * old implementation is kept here as the reference: hand-written examples would
 * only cover the shapes someone thought of, and the shapes that bite are the
 * ones the pipeline emits and nobody remembers (a station block with no `t`
 * because every run shares a planned time, no `c` because nothing was
 * cancelled, `d` absent because it never differed from `a`).
 */
class ShardParseEquivalenceTest {

    @Test
    fun `matches the old parser on every shard on this machine`() {
        val shards = File("../tools/.shards").walkTopDown()
            .filter { it.isFile && it.name.endsWith(".jgz") }
            .take(4000)
            .toList()
        assumeTrue("no shard cache; run the evaluation first", shards.size > 100)

        var runs = 0
        for (file in shards) {
            val json = file.inputStream().use { GZIPInputStream(it).readBytes() }
                .decodeToString()
            val expected = referenceParse(json)
            val actual = HistoryRepository.parseShard(json)
            assertEquals(file.name, expected?.trainName, actual?.trainName)
            assertEquals(file.name, expected?.trainType, actual?.trainType)
            assertEquals(file.name, expected?.stations?.keys, actual?.stations?.keys)
            expected?.stations?.forEach { (name, station) ->
                val got = actual!!.stations.getValue(name)
                assertEquals("${file.name} $name eva", station.eva, got.eva)
                assertEquals("${file.name} $name", station.runs, got.runs)
                runs += station.runs.size
            }
        }
        assertTrue("nothing compared", runs > 10_000)
        println("compared ${shards.size} shards, $runs runs")
    }

    @Test
    fun `a station whose runs all share one planned time has no index array`() {
        // No "t": every run takes tod[0]. The pipeline omits it, and a parser
        // that required it would return the wrong time for every such run.
        val json = """
            {"v":2,"train":"RE 1","type":"RE","stations":{"A":{"eva":"1",
              "tod":[501],"days":[20544,1],"a":[0,3],"d":[0,0],"p":[null,null]}}}
        """.trimIndent()
        val runs = HistoryRepository.parseShard(json)!!.stations.getValue("A").runs
        assertEquals(listOf("08:21", "08:21"), runs.map { it.plannedTimeOfDay })
        assertEquals(referenceParse(json)!!.stations.getValue("A").runs, runs)
    }

    @Test
    fun `a station where nothing was cancelled has no cancelled array`() {
        val json = """
            {"v":2,"train":"RE 1","type":"RE","stations":{"A":{"eva":"1",
              "tod":[60],"days":[20544],"a":[1]}}}
        """.trimIndent()
        val runs = HistoryRepository.parseShard(json)!!.stations.getValue("A").runs
        assertEquals(listOf(false), runs.map { it.cancelled })
        assertEquals(1, runs[0].departureDelay, )
        assertEquals(referenceParse(json)!!.stations.getValue("A").runs, runs)
    }

    @Test
    fun `not a shard at all is null, not an empty history`() {
        assertEquals(null, HistoryRepository.parseShard("not json"))
        assertEquals(null, HistoryRepository.parseShard("""{"nope":1}"""))
    }

    @Test
    fun `unknown keys from a future pipeline are ignored`() {
        val json = """
            {"v":3,"train":"RE 1","type":"RE","weather":"rain","stations":{"A":{
              "eva":"1","tod":[60],"days":[20544],"a":[1],"mood":"grim"}}}
        """.trimIndent()
        assertEquals(1, HistoryRepository.parseShard(json)!!.stations.getValue("A").runs.size)
    }

    // --- the implementation this replaced, verbatim ---------------------------

    private fun referenceParse(json: String): TrainHistory? {
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
                    runs = if ("days" in obj) referenceColumnar(obj) else emptyList(),
                )
            },
        )
    }

    private fun referenceColumnar(obj: JsonObject): List<HistoricalRun> {
        val days = obj["days"]?.jsonArray ?: return emptyList()
        val tods = obj["tod"]?.jsonArray?.map { it.jsonPrimitive.intOrNull ?: 0 }
            ?: return emptyList()
        val t = obj["t"]?.jsonArray
        val a = obj["a"]?.jsonArray
        val d = obj["d"]?.jsonArray
        val p = obj["p"]?.jsonArray
        val cancelled = obj["c"]?.jsonArray
            ?.mapNotNullTo(HashSet()) { it.jsonPrimitive.intOrNull } ?: emptySet()

        fun int(arr: JsonArray?, i: Int): Int? = arr?.getOrNull(i)?.jsonPrimitive?.intOrNull

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
}
