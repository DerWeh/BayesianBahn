package io.github.derweh.bayesianbahn

import io.github.derweh.bayesianbahn.data.CandidateBuilder
import io.github.derweh.bayesianbahn.data.Predictor
import io.github.derweh.bayesianbahn.model.ConnectionModel
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.time.LocalDate

/**
 * Scores a complete two-leg journey: how late the passenger actually arrives at
 * the far end of a change, against the arrival that happened.
 *
 * The arrival harness asks *when does this train get in*, and the connection
 * scorer asks *does the change work*. Neither answers the question a passenger
 * with a change actually has, and the two cannot be compared with each other
 * because they are not in the same units. This one is: a distribution over the
 * final arrival, scored with CRPS in minutes, exactly as a direct journey is.
 *
 * DB's answer is the arrival of whichever train *its* forecasts say the
 * passenger catches, read at the same moment ours is. Both forecasters see the
 * same candidates, and neither is handed a delay the other was denied — the
 * event builder in `tools/score_events.py` resolves both boardings and this
 * harness only rebases them onto the model's own reference.
 *
 * Like [ForecastHarness] it drives the shipping code rather than a description
 * of it: candidates come from [CandidateBuilder], which the app itself uses,
 * and the mixture from [ConnectionModel.propagate].
 *
 * Run with:
 *   HARNESS_JOURNEYS=journeys.jsonl HARNESS_SHARDS=tools/.shards \
 *   HARNESS_OUT=scored.jsonl HARNESS_DAY=2026-08-25 \
 *   pixi run ./gradlew testDebugUnitTest --tests '*JourneyHarness'
 */
class JourneyHarness {

    @Test
    fun `score recorded journeys with the shipping model`() {
        val journeys = System.getenv("HARNESS_JOURNEYS")
        assumeTrue("set HARNESS_JOURNEYS to run the harness", journeys != null)
        val events = File(requireNotNull(journeys))
        val shards = File(requireNotNull(System.getenv("HARNESS_SHARDS")))
        require(shards.isDirectory) {
            "HARNESS_SHARDS=${shards.absolutePath} is not a directory " +
                "(use an absolute path; tests run with the working directory at app/)"
        }
        val out = File(requireNotNull(System.getenv("HARNESS_OUT")))
        val day = LocalDate.parse(requireNotNull(System.getenv("HARNESS_DAY")))
        val blind = System.getenv("HARNESS_BLIND") != null

        val histories = ForecastHarness.ShardStore(shards, day)
        val predictor = Predictor()
        var scored = 0
        var noCandidates = 0
        var noFeeder = 0

        out.bufferedWriter().use { writer ->
            events.forEachLine { line ->
                if (line.isBlank()) return@forEachLine
                val event = Json.parseToJsonElement(line) as JsonObject
                val truthArrival = event.int("truth_arrival")
                val dbArrival = event.int("db_arrival")
                if (truthArrival == null || dbArrival == null) return@forEachLine

                // Already trimmed to before `day` by the store, once per shard.
                val feederHistory = histories.load(
                    event.str("cat")!!, event.str("num")!!, event.str("line"),
                )
                val feederPlanned = ForecastHarness.wallMinutesToMillis(
                    event.int("planned")!!,
                )
                val feederForecast = predictor.forecast(
                    history = feederHistory,
                    stationEva = event.str("eva")!!,
                    stationName = "",
                    trainCategory = event.str("cat")!!,
                    plannedTimeMillis = feederPlanned,
                    liveDelayMinutes = if (blind) null else event.dbl("db"),
                    today = day,
                )

                val candidates = (event["candidates"] as JsonArray).jsonArray
                    .map { it as JsonObject }
                    .mapNotNull { candidate ->
                        CandidateBuilder.build(
                            history = histories.load(
                                candidate.str("cat")!!, candidate.str("num")!!,
                                candidate.str("line"),
                            ),
                            id = candidate.str("id")!!,
                            label = candidate.str("id")!!,
                            transferEva = event.str("eva")!!,
                            transferName = "",
                            destinationEva = event.str("dest_eva"),
                            destinationName = event.str("dest")!!,
                            plannedDepartureMillis = ForecastHarness.wallMinutesToMillis(
                                candidate.int("planned_dep")!!,
                            ),
                            liveDepartureDelay =
                                if (blind) null else candidate.dbl("live_dep"),
                            cancelledLive =
                                !blind && candidate.bool("cancelled_live") == true,
                            today = day,
                        )
                    }
                if (candidates.isEmpty()) {
                    noCandidates++
                    return@forEachLine
                }

                val result = ConnectionModel.propagate(
                    feederArrival = feederForecast.distribution,
                    feederPlannedArrivalMillis = feederPlanned,
                    transferMinutes = TRANSFER_MINUTES,
                    candidates = candidates,
                )
                if (result == null) {
                    noFeeder++
                    return@forEachLine
                }

                // The model answers in minutes relative to the planned arrival
                // of the first candidate it kept, which it picks itself. Both
                // the truth and DB's answer are absolute wall-clock minutes, so
                // they are rebased here rather than guessed at in Python.
                val reference = result.referenceArrivalMillis
                val truth = minutesFrom(reference, truthArrival)
                val db = minutesFrom(reference, dbArrival)
                val d = result.distribution
                writer.write(
                    """{"eva":${ForecastHarness.q(event.str("eva"))},""" +
                        """"cat":${ForecastHarness.q(event.str("cat"))},""" +
                        """"num":${ForecastHarness.q(event.str("num"))},""" +
                        """"dest":${ForecastHarness.q(event.str("dest"))},""" +
                        """"tau":0,"lead":${event.dbl("lead")},""" +
                        """"read_at":${event.dbl("read_at")},""" +
                        """"planned":${event.int("planned")},""" +
                        """"planned_dep":${event.int("planned_dep")},""" +
                        """"candidates":${candidates.size},""" +
                        // Whether the passenger's train sat past the end of the
                        // list either forecaster was offered — the journeys the
                        // old cap dropped, and the ones the app got most wrong.
                        """"beyond_list":${event.bool("beyond_list")},""" +
                        // The model reconstructs the planned arrival at the far
                        // end from history; the event builder holds the real
                        // one. Emitting the reference makes any gap between the
                        // two visible instead of turning into a delay.
                        """"reference":${millisToWallMinutes(reference)},""" +
                        """"reference_id":${ForecastHarness.q(
                            result.candidates.firstOrNull {
                                it.candidate.plannedArrivalMillis == reference
                            }?.candidate?.id,
                        )},""" +
                        """"miss_p":${result.missProbability},""" +
                        """"db":$db,"truth":$truth,""" +
                        """"crps":${ForecastHarness.crps(d, truth)},""" +
                        """"cdf_at":${d.cdf(truth)},"cdf_below":${d.cdf(truth - 1.0)},""" +
                        """"q10":${d.quantile(0.1)},"q50":${d.quantile(0.5)},""" +
                        """"q90":${d.quantile(0.9)},""" +
                        """"source":${ForecastHarness.q(feederForecast.source.name)},""" +
                        """"runs":${feederForecast.runCount}}""",
                )
                writer.newLine()
                scored++
            }
        }
        println("journeys: scored $scored, $noCandidates with no usable candidate, " +
            "$noFeeder the model declined")
        assumeTrue("nothing scoreable yet", scored > 0)
    }

    companion object {
        /** ConnectionPlanner's default, and what the connection scorer assumes. */
        const val TRANSFER_MINUTES = 5

        /** Real epoch millis back to the wall-clock minutes the journal stores. */
        fun millisToWallMinutes(millis: Long): Long =
            java.time.Instant.ofEpochMilli(millis).atZone(ForecastHarness.BERLIN)
                .toLocalDateTime().atZone(java.time.ZoneId.of("UTC")).toEpochSecond() / 60

        /** Wall-clock minutes at the destination, as minutes from the reference. */
        fun minutesFrom(referenceMillis: Long, wallMinutes: Int): Double =
            (ForecastHarness.wallMinutesToMillis(wallMinutes) - referenceMillis) / 60_000.0

        fun JsonObject.int(key: String) = with(ForecastHarness) { int(key) }
        fun JsonObject.dbl(key: String) = with(ForecastHarness) { dbl(key) }
        fun JsonObject.str(key: String) = with(ForecastHarness) { str(key) }
        fun JsonObject.bool(key: String) = with(ForecastHarness) { bool(key) }
    }
}
