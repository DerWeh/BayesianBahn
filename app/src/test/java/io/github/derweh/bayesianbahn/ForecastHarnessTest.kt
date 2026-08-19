package io.github.derweh.bayesianbahn

import io.github.derweh.bayesianbahn.data.StationHistory
import io.github.derweh.bayesianbahn.data.TrainHistory
import io.github.derweh.bayesianbahn.model.DelayDistribution
import io.github.derweh.bayesianbahn.model.HistoricalRun
import java.io.File
import java.time.LocalDate
import java.util.zip.GZIPOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.time.Instant
import java.time.ZoneId
import kotlin.math.abs

/**
 * The harness deliberately reimplements nothing of the model — but it does add
 * two pieces of arithmetic of its own, and both are the silent kind. A wrong
 * CRPS would rank model changes wrongly forever, and a wrong clock conversion
 * would shift every prediction by the German UTC offset while still producing
 * plausible-looking numbers.
 */
class ForecastHarnessTest {

    /** All mass at [at]: the degenerate case where CRPS has a closed form. */
    private fun pointMass(at: Double) = object : DelayDistribution {
        override fun cdf(x: Double) = if (x >= at) 1.0 else 0.0
        override fun quantile(p: Double) = at
    }

    /** Uniform on [lo, hi]; CRPS against a point inside it is analytic. */
    private fun uniform(lo: Double, hi: Double) = object : DelayDistribution {
        override fun cdf(x: Double) = ((x - lo) / (hi - lo)).coerceIn(0.0, 1.0)
        override fun quantile(p: Double) = lo + p * (hi - lo)
    }

    @Test
    fun `a point forecast that is right scores zero`() {
        assertEquals(0.0, ForecastHarness.crps(pointMass(5.0), 5.0), 1e-9)
    }

    @Test
    fun `a point forecast scores its absolute error`() {
        // CRPS of a degenerate distribution reduces to |y - m|.
        assertEquals(7.0, ForecastHarness.crps(pointMass(3.0), 10.0), 1e-9)
        assertEquals(7.0, ForecastHarness.crps(pointMass(10.0), 3.0), 1e-9)
    }

    @Test
    fun `a sharper forecast beats a vaguer one when both are centred`() {
        val tight = ForecastHarness.crps(uniform(-2.0, 2.0), 0.0)
        val loose = ForecastHarness.crps(uniform(-20.0, 20.0), 0.0)
        assertTrue("sharper must score lower: $tight vs $loose", tight < loose)
    }

    @Test
    fun `spreading probability beats being confidently wrong`() {
        val hedged = ForecastHarness.crps(uniform(0.0, 20.0), 18.0)
        val confident = ForecastHarness.crps(pointMass(0.0), 18.0)
        assertTrue("$hedged should beat $confident", hedged < confident)
    }

    @Test
    fun `crps is never negative`() {
        for (y in listOf(-30.0, 0.0, 4.0, 120.0)) {
            assertTrue(ForecastHarness.crps(uniform(-5.0, 30.0), y) >= 0.0)
        }
    }

    @Test
    fun `the uniform case matches the closed form`() {
        // For U(lo, hi) with y inside, CRPS integrates to
        // [(y-lo)^3 + (hi-y)^3] / 3(hi-lo)^2.
        //
        // The tolerance is about this toy distribution, not about the harness:
        // a continuous CDF sampled on whole minutes carries rectangle-rule
        // error. Real forecasts are step functions that jump at integer
        // minutes, where the same sum is exact — which is why the grid is one
        // minute wide in the first place.
        val lo = 0.0
        val hi = 10.0
        val y = 4.0
        val span = hi - lo
        val exact = ((y - lo) * (y - lo) * (y - lo) + (hi - y) * (hi - y) * (hi - y)) /
            (3 * span * span)
        assertEquals(exact, ForecastHarness.crps(uniform(lo, hi), y), 0.15)
    }

    @Test
    fun `the grid is exact for a step distribution that jumps on whole minutes`() {
        // The real case: an empirical CDF over integer-minute delays.
        val steps = object : DelayDistribution {
            override fun cdf(x: Double) = when {
                x < 2.0 -> 0.0
                x < 5.0 -> 0.5
                else -> 1.0
            }
            override fun quantile(p: Double) = if (p <= 0.5) 2.0 else 5.0
        }
        // y = 5: (0-0)^2 below 2, (0.5)^2 over [2,5) = 3 * 0.25, 0 above.
        assertEquals(0.75, ForecastHarness.crps(steps, 5.0), 1e-9)
    }

    @Test
    fun `history is trimmed to what was known the day before`() {
        // The published recent overlay is rebuilt daily and now covers the day
        // under evaluation, so a shard downloaded today carries the answer.
        val day = LocalDate.of(2026, 8, 17)
        val runs = listOf(
            HistoricalRun(day.minusDays(2), "18:40", 3, null, null, false),
            HistoricalRun(day.minusDays(1), "18:40", 5, null, null, false),
            HistoricalRun(day, "18:40", 41, null, null, false),
            HistoricalRun(day.plusDays(1), "18:40", 2, null, null, false),
        )
        val history = TrainHistory("RE 1", "RE", mapOf("A" to StationHistory("1", runs)))
        val trimmed = ForecastHarness.asOf(history, day)
        assertEquals(2, trimmed.stations.getValue("A").runs.size)
        assertTrue(trimmed.stations.getValue("A").runs.all { it.date.isBefore(day) })
    }

    @Test
    fun `a history that still leaks is refused`() {
        val day = LocalDate.of(2026, 8, 17)
        val history = TrainHistory(
            "RE 1", "RE",
            mapOf("A" to StationHistory("1",
                listOf(HistoricalRun(day, "18:40", 41, null, null, false)))),
        )
        try {
            ForecastHarness.requireNoRunsOnOrAfter(history, day)
            throw AssertionError("expected the leak check to fire")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("on or after"))
        }
    }

    @Test
    fun `wall clock minutes become the right instant in summer`() {
        // 2026-08-17 18:00 German time is 16:00 UTC.
        val minutes = (Instant.parse("2026-08-17T18:00:00Z").epochSecond / 60).toInt()
        val millis = ForecastHarness.wallMinutesToMillis(minutes)
        val utc = Instant.ofEpochMilli(millis).atZone(ZoneId.of("UTC"))
        assertEquals(16, utc.hour)
    }

    @Test
    fun `wall clock minutes become the right instant in winter`() {
        val minutes = (Instant.parse("2026-01-17T18:00:00Z").epochSecond / 60).toInt()
        val utc = Instant.ofEpochMilli(ForecastHarness.wallMinutesToMillis(minutes))
            .atZone(ZoneId.of("UTC"))
        assertEquals("one hour offset outside DST", 17, utc.hour)
    }

    @Test
    fun `the local time the model sees is the one the timetable says`() {
        val minutes = (Instant.parse("2026-08-17T18:00:00Z").epochSecond / 60).toInt()
        val berlin = Instant.ofEpochMilli(ForecastHarness.wallMinutesToMillis(minutes))
            .atZone(ZoneId.of("Europe/Berlin"))
        assertEquals(18, berlin.hour)
        assertTrue(abs(berlin.minute) == 0)
    }

    // --- the shard cache -----------------------------------------------------
    //
    // An unbounded cache scored one evening fine and then died of
    // OutOfMemoryError on the first full day, because nothing here scaled with
    // the number of trains until the data did. These tests pin the bound.

    @get:Rule
    val temp = TemporaryFolder()

    /** A minimal v2 columnar shard, gzipped the way the branches publish it. */
    private fun writeShard(root: File, tier: String, key: String) {
        val dir = File(root, tier).apply { mkdirs() }
        val json = """
            {"train":"$key","type":"RE","stations":{"Somewhere":
            {"eva":"8000001","days":[20000],"tod":[480],"t":[0],"a":[3],"p":[2]}}}
        """.trimIndent()
        GZIPOutputStream(File(dir, "$key.jgz").outputStream()).use {
            it.write(json.toByteArray())
        }
    }

    private fun store(keys: List<String>, capacity: Int): ForecastHarness.ShardStore {
        val root = temp.newFolder()
        keys.forEach { writeShard(root, "base", it) }
        // The harness refuses a shard root with no base/, so create it even when
        // the test writes no shards at all.
        File(root, "base").mkdirs()
        return ForecastHarness.ShardStore(root, capacity)
    }

    @Test
    fun `a shard is parsed once when a train's events arrive together`() {
        val s = store(listOf("RE_1", "RE_2"), capacity = 8)
        repeat(5) { s.load("RE", "1", null) }
        repeat(5) { s.load("RE", "2", null) }
        assertEquals("grouped access should parse each shard exactly once", 2, s.parses)
    }

    @Test
    fun `the cache does not grow past its capacity`() {
        val keys = (1..5).map { "RE_$it" }
        val s = store(keys, capacity = 2)
        // Walk all five, then come back to the first. With a capacity of two it
        // must have been evicted — which is the whole point: an unbounded cache
        // would still be holding it, and would hold all four thousand.
        (1..5).forEach { s.load("RE", "$it", null) }
        assertEquals(5, s.parses)
        s.load("RE", "1", null)
        assertEquals("the evicted history should be re-read, not still resident", 6, s.parses)
    }

    @Test
    fun `an evicted history is still returned correctly`() {
        val s = store((1..5).map { "RE_$it" }, capacity = 2)
        (1..5).forEach { s.load("RE", "$it", null) }
        val again = s.load("RE", "1", null)
        assertEquals("RE_1", again?.trainName)
        assertEquals(1, again?.stations?.size)
    }

    @Test
    fun `a train with no shard is not re-read on every event`() {
        val s = store(emptyList(), capacity = 8)
        repeat(5) { assertNull(s.load("RE", "999", null)) }
        // The bug this pins: `getOrPut` treats a cached null as a miss, so every
        // event of every history-less train went back to the filesystem.
        assertEquals("a missing shard should be remembered as missing", 1, s.parses)
    }

    @Test
    fun `the line key is tried when the number has no shard`() {
        val s = store(listOf("RE_9"), capacity = 8)
        // Number first, then line — so this costs two parses and finds the line.
        val found = s.load("RE", "12345", "RE 9")
        assertEquals("RE_9", found?.trainName)
        assertEquals(2, s.parses)
    }
}
