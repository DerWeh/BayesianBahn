package io.github.derweh.bayesianbahn

import io.github.derweh.bayesianbahn.data.HistoryRepository
import io.github.derweh.bayesianbahn.data.Predictor
import io.github.derweh.bayesianbahn.data.TrainHistory
import io.github.derweh.bayesianbahn.data.StationHistory
import io.github.derweh.bayesianbahn.model.DelayDistribution
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.util.zip.GZIPInputStream

/**
 * Scores the *real* model over recorded DB forecasts — Phase B of the
 * comparison against DB's own predictions.
 *
 * `pipeline/backtest.py` mirrors the model in Python, and that mirror has
 * already drifted once (its long-distance set is missing `WB`). A published
 * claim about accuracy must come from the code that ships, so this drives
 * [Predictor] itself, fed by shards read exactly as [HistoryRepository] reads
 * them. Nothing here reimplements the model; the only new arithmetic is CRPS,
 * which is tested separately in [ForecastHarnessTest].
 *
 * It is a JUnit test rather than a Gradle module so that an app F-Droid builds
 * reproducibly gains no build surface. It is opt-in for the same reason
 * `NavigatorCompareE2E` is: it needs data the repository does not carry.
 *
 * Run with:
 *   HARNESS_EVENTS=events.jsonl HARNESS_SHARDS=tools/.shards \
 *   HARNESS_OUT=scored.jsonl HARNESS_DAY=2026-08-17 \
 *   pixi run ./gradlew testDebugUnitTest --tests '*ForecastHarness'
 */
class ForecastHarness {

    @Test
    fun `score recorded events with the shipping model`() {
        val eventsPath = System.getenv("HARNESS_EVENTS")
        assumeTrue("set HARNESS_EVENTS to run the harness", eventsPath != null)
        val shards = File(requireNotNull(System.getenv("HARNESS_SHARDS")))
        // Gradle runs unit tests with the working directory at the module, so a
        // relative path here resolves under app/ and finds nothing — every
        // event would then fall back to the prior and the run would look
        // successful. Refuse instead.
        require(shards.isDirectory) {
            "HARNESS_SHARDS=${shards.absolutePath} is not a directory " +
                "(use an absolute path; tests run with the working directory at app/)"
        }
        require(File(shards, "base").isDirectory) {
            "no base/ under ${shards.absolutePath}; run tools/fetch_shards.py first"
        }
        val out = File(requireNotNull(System.getenv("HARNESS_OUT")))
        val day = LocalDate.parse(requireNotNull(System.getenv("HARNESS_DAY")))
        val blind = System.getenv("HARNESS_BLIND") != null

        val histories = ShardStore(shards)
        val predictor = Predictor()
        var scored = 0
        var skipped = 0
        var withHistory = 0

        out.bufferedWriter().use { writer ->
            File(eventsPath).forEachLine { line ->
                if (line.isBlank()) return@forEachLine
                val event = Json.parseToJsonElement(line) as JsonObject
                val truth = event.int("archive") ?: event.int("settled")
                if (truth == null || event.bool("cancelled") == true) {
                    skipped++
                    return@forEachLine
                }
                val history = histories.load(
                    event.str("cat")!!, event.str("num")!!, event.str("line"),
                )?.let { asOf(it, day) }
                history?.let {
                    // Belt and braces: asOf should have made this impossible.
                    requireNoRunsOnOrAfter(it, day)
                    withHistory++
                }

                val plannedMillis = wallMinutesToMillis(event.int("planned")!!)
                val forecast = predictor.forecast(
                    history = history,
                    stationEva = event.str("eva")!!,
                    stationName = "",
                    trainCategory = event.str("cat")!!,
                    plannedTimeMillis = plannedMillis,
                    // HARNESS_BLIND drops the live signal, isolating what the
                    // history alone predicts. The app's live path feeds DB's
                    // forecast for *this* station into a model trained on the
                    // actual delay at the *previous* stop — a documented
                    // approximation whose cost is worth measuring.
                    liveDelayMinutes = if (blind) null else event.dbl("db"),
                    // Pinned: recency weighting depends on it, so leaving it as
                    // "now" would make the same input score differently on a
                    // rerun and destroy the regression yardstick.
                    today = day,
                )
                val d = forecast.distribution
                writer.write(
                    """{"eva":${q(event.str("eva"))},"cat":${q(event.str("cat"))},""" +
                        """"num":${q(event.str("num"))},"tau":${event.int("tau")},""" +
                        """"lead":${event.dbl("lead")},"db":${event.int("db")},""" +
                        """"truth":$truth,"crps":${crps(d, truth.toDouble())},""" +
                        """"cdf_at":${d.cdf(truth.toDouble())},""" +
                        """"cdf_below":${d.cdf(truth - 1.0)},""" +
                        """"q10":${d.quantile(0.1)},"q50":${d.quantile(0.5)},""" +
                        """"q90":${d.quantile(0.9)},"source":${q(forecast.source.name)},""" +
                        """"runs":${forecast.runCount}}""",
                )
                writer.newLine()
                scored++
            }
        }
        println("harness: scored $scored events, skipped $skipped without truth")
        assumeTrue("nothing scoreable yet", scored > 0)
        // Everything falling back to the prior means the history never loaded,
        // not that the model has nothing to say.
        require(withHistory > 0) {
            "no event found any history under ${shards.absolutePath} — check the " +
                "shard keys match HistoryRepository.shardKey"
        }
        println("harness: $withHistory of $scored events had history")
    }

    /** Shards on disk, read the way HistoryRepository reads them. */
    private class ShardStore(private val root: File) {
        private val cache = mutableMapOf<String, TrainHistory?>()

        fun load(category: String, number: String, line: String?): TrainHistory? {
            for (key in candidateKeys(category, number, line)) {
                val found = cache.getOrPut(key) {
                    HistoryRepository.mergeHistories(read("base", key), read("recent", key))
                }
                if (found != null) return found
            }
            return null
        }

        private fun read(tier: String, key: String): TrainHistory? {
            val file = File(File(root, tier), "$key.jgz")
            if (!file.isFile) return null
            val bytes = file.inputStream().use { GZIPInputStream(it).readBytes() }
            return HistoryRepository.parseShard(bytes.decodeToString())
        }

        private fun candidateKeys(category: String, number: String, line: String?) =
            buildList {
                if (number.isNotBlank()) add(HistoryRepository.shardKey("$category $number"))
                if (!line.isNullOrBlank()) {
                    add(
                        HistoryRepository.shardKey(
                            if (line.startsWith(category)) line else "$category $line",
                        ),
                    )
                }
            }.distinct()
    }

    companion object {
        val BERLIN: ZoneId = ZoneId.of("Europe/Berlin")

        /**
         * Journal times are wall clock stored as if UTC; the model wants real
         * epoch millis. Crossing this boundary wrongly shifts every prediction
         * by the German offset, which changes the time band and the
         * time-of-day key without looking broken.
         */
        fun wallMinutesToMillis(minutes: Int): Long =
            Instant.ofEpochSecond(minutes * 60L).atZone(ZoneId.of("UTC")).toLocalDateTime()
                .atZone(BERLIN).toInstant().toEpochMilli()

        /**
         * History as the app would have held it the evening before.
         *
         * The published `shards-recent` overlay is rebuilt daily and now covers
         * the evaluated day itself, so a shard fetched today carries the answer.
         * Trimming here rather than relying on when the shards were downloaded
         * makes the score a function of its inputs alone — which is the whole
         * point of a yardstick meant to compare one commit against another.
         */
        fun asOf(history: TrainHistory, day: LocalDate): TrainHistory =
            history.copy(
                stations = history.stations.mapValues { (_, station) ->
                    station.copy(runs = station.runs.filter { it.date.isBefore(day) })
                },
            )

        fun requireNoRunsOnOrAfter(history: TrainHistory, day: LocalDate) {
            val leak = history.stations.values
                .flatMap { it.runs }
                .maxByOrNull { it.date }
                ?.date
            require(leak == null || leak.isBefore(day)) {
                "history contains a run on $leak, on or after the evaluated day $day"
            }
        }

        /**
         * CRPS by integrating (F(x) - 1{x >= y})^2 over whole minutes. Delays
         * are reported in minutes, so a one-minute grid over the range the
         * model is defined on loses nothing worth having.
         */
        fun crps(d: DelayDistribution, y: Double, lo: Int = -60, hi: Int = 600): Double {
            var total = 0.0
            for (x in lo..hi) {
                val indicator = if (x >= y) 1.0 else 0.0
                val diff = d.cdf(x.toDouble()) - indicator
                total += diff * diff
            }
            return total
        }

        fun q(s: String?) = if (s == null) "null" else "\"$s\""

        fun JsonObject.int(key: String): Int? = this[key]?.jsonPrimitive?.intOrNull
        fun JsonObject.dbl(key: String): Double? = this[key]?.jsonPrimitive?.doubleOrNull
        fun JsonObject.str(key: String): String? = this[key]?.jsonPrimitive?.contentOrNull
        fun JsonObject.bool(key: String): Boolean? = this[key]?.jsonPrimitive?.booleanOrNull
    }
}
