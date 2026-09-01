package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.EmpiricalDelay
import io.github.derweh.bayesianbahn.model.HistoricalRun
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * How long one forecast takes, at the sizes the shards really reach.
 *
 * Pooling a train's history with its line's changed what a forecast touches by
 * an order of magnitude. A run number calls at a station about once a day, so
 * its shard holds a few hundred runs; a line shard holds every run of the line
 * there, and the worst one across the evaluation's 62 stations — S41 at Berlin
 * Hermannstraße — holds 8,488 in its published 45-day window. The time-of-day
 * filter is the only thing that touches all of them, and it used to split and
 * parse two "HH:mm" strings per run on every prediction.
 *
 * These run on the JVM, not on a phone, so the budgets are deliberately loose:
 * a mid-range Android device on ART is commonly several times slower than a
 * desktop JVM on allocation-heavy code like this. What they are for is the
 * *shape* — a forecast that stays well under a millisecond at the worst shard
 * size anyone will meet leaves room for a large device penalty and for the
 * dozens of forecasts a journey search makes. A regression that puts the parse
 * back inside the filter shows up here as an order of magnitude, not as noise.
 */
class ForecastLatencyTest {

    private val zone = ZoneId.of("Europe/Berlin")
    private val today = LocalDate.of(2026, 8, 20)
    private val planned =
        ZonedDateTime.of(today, java.time.LocalTime.of(8, 0), zone).toInstant().toEpochMilli()
    private val station = "Hermannstraße"

    private fun shard(name: String, runs: List<HistoricalRun>) =
        TrainHistory(name, "S", mapOf(station to StationHistory("8089105", runs)), line = "S41")

    private fun run(daysAgo: Int, minute: Int, i: Int) = HistoricalRun(
        date = today.minusDays(daysAgo.toLong() + 1),
        plannedTimeOfDay = "%02d:%02d".format(minute / 60, minute % 60),
        arrivalDelay = i % 11,
        departureDelay = i % 11,
        previousStopDelay = i % 7,
        cancelled = i % 97 == 0,
    )

    /** A run number's shard: one call a day at the same slot, as IRIS numbers them. */
    private fun byNumber(days: Int) =
        shard("S 41454", List(days) { run(it, 8 * 60, it) })

    /** A line's shard: every run of the line there, spread across the day. */
    private fun byLine(runsPerDay: Int, days: Int) = shard(
        "S41",
        List(runsPerDay * days) { i ->
            run(i / runsPerDay, 5 * 60 + (i % runsPerDay) * (19 * 60) / runsPerDay, i)
        },
    )

    private fun time(
        own: TrainHistory?,
        line: TrainHistory?,
        live: Double? = null,
        rounds: Int = 300,
        onFetch: () -> Unit = {},
    ): Double = runBlocking {
        val predictor = Predictor()
        suspend fun once() = predictor.forecast(
            history = own,
            stationEva = "8089105",
            stationName = station,
            trainCategory = "S",
            plannedTimeMillis = planned,
            liveDelayMinutes = live,
            today = today,
            // Already-loaded shards: the cost of *fetching* one is the network's
            // and is measured by the cache, not here.
            lineHistory = { onFetch(); line },
        )
        repeat(rounds / 4) { once() }   // let the JIT settle
        val start = System.nanoTime()
        repeat(rounds) { once() }
        (System.nanoTime() - start) / 1e6 / rounds
    }

    @Test
    fun `a pooled forecast stays fast at the worst line shard in the data`() {
        val line = byLine(runsPerDay = 189, days = 45)   // 8,505 runs, as S41 has
        val thin = byNumber(days = 3)                    // a freshly renumbered train
        val blind = time(thin, line)
        val live = time(thin, line, live = 12.0)
        println("pooled over ${line.stations.getValue(station).runs.size} line runs: " +
            "%.3f ms blind, %.3f ms live".format(blind, live))
        assertTrue(
            "a pooled forecast took $blind ms blind / $live ms live — the " +
                "time-of-day filter has probably gone back to parsing strings",
            maxOf(blind, live) < 3.0,
        )
    }

    @Test
    fun `a pooled forecast at the median line shard size`() {
        val line = byLine(runsPerDay = 20, days = 45)
        val blind = time(byNumber(days = 3), line)
        println("pooled over ${line.stations.getValue(station).runs.size} line runs: " +
            "%.3f ms blind".format(blind))
        assertTrue("a median pooled forecast took $blind ms", blind < 1.0)
    }

    @Test
    fun `a train with its own history never pays for the line at all`() {
        // Above LINE_CEILING_N the line shard is not asked for, so this costs
        // exactly what a forecast cost before pooling existed.
        val own = byNumber(days = 150)   // base + recent, the usual case
        var asked = 0
        val line = byLine(runsPerDay = 189, days = 45)
        val elapsed = time(own, line) { asked++ }
        println("number-only over 150 runs: %.3f ms".format(elapsed))
        assertEquals(
            "the line shard was asked for despite ${EmpiricalDelay.LINE_CEILING_N}" +
                " effective runs of the train's own",
            0, asked,
        )
        assertTrue("a number-only forecast took $elapsed ms", elapsed < 1.0)
    }

    @Test
    fun `a whole journey search's worth of forecasts fits in a frame or two`() {
        // A two-leg search forecasts the feeder at the transfer and scores a
        // handful of candidates; thirty pooled forecasts is a generous upper
        // bound for one search, and the worst shard in the country for each.
        val line = byLine(runsPerDay = 189, days = 45)
        val per = time(byNumber(days = 3), line)
        val search = 30 * per
        println("30 pooled forecasts: %.1f ms".format(search))
        assertTrue("a search's forecasts took $search ms", search < 60.0)
    }

    @Test
    fun `reading the worst line shard is a one-off, not a per-forecast cost`() {
        // The runs are decoded once when the shard is read and reused for every
        // prediction of the session, so this is paid at most once per line and
        // station per 18-hour cache window — never inside a search's inner loop.
        val json = buildString {
            append("""{"v":2,"train":"S41","type":"S","line":"S41","stations":""")
            append("""{"$station":{"eva":"8089105","tod":[""")
            append((0 until 189).joinToString(",") { (300 + it * 6).toString() })
            append("""],"days":[20320""")
            repeat(8_504) { append(if (it % 189 == 188) ",1" else ",0") }
            append("""],"t":[""")
            append((0 until 8_505).joinToString(",") { (it % 189).toString() })
            append("""],"a":[""")
            append((0 until 8_505).joinToString(",") { (it % 11).toString() })
            append("""],"p":[""")
            append((0 until 8_505).joinToString(",") { (it % 7).toString() })
            append("]}}}")
        }
        repeat(3) { HistoryRepository.parseShard(json) }
        val start = System.nanoTime()
        val parsed = HistoryRepository.parseShard(json)!!
        val elapsed = (System.nanoTime() - start) / 1e6
        println("decoding an 8,505-run line shard: %.1f ms".format(elapsed))
        assertEquals(8_505, parsed.stations.getValue(station).runs.size)
        assertTrue("decoding took $elapsed ms", elapsed < 200.0)
    }
}
