package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.DeutschlandTicket
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * How a synthetic board loads its shards, which is where a future-date search
 * spent its five minutes: one shard per stop, up to forty stops, each up to
 * four round trips on a miss — in sequence, and repeated for every board the
 * planner opened.
 */
class SyntheticBoardLoadingTest {

    private val zone = ZoneId.of("Europe/Berlin")
    private val today: LocalDate = LocalDate.now(zone)

    /** 08:00 local today; the board window starts here. */
    private val start: Long =
        ZonedDateTime.of(today.atTime(8, 0), zone).toInstant().toEpochMilli()

    /**
     * [count] regional trains departing from 08:05 at five-minute spacing, plus
     * one ICE. Weekday mask 127 runs every day and `lastSeen` is today, so the
     * board does not rot as the calendar moves.
     */
    private fun boardJson(count: Int): ByteArray {
        val seen = today.toEpochDay()
        val trains = (1..count).joinToString(",") { i ->
            """["RE","${1000 + i}","RE$i",${8 * 60 + i * 5},${8 * 60 + i * 5},127,$seen]"""
        }
        val ice = """["ICE","500","ICE500",${8 * 60 + 2},${8 * 60 + 2},127,$seen]"""
        return """{"name":"Test","trains":[$trains,$ice]}""".toByteArray()
    }

    private fun timetable(count: Int, history: HistorySource) = SyntheticTimetable(
        ByteSource { _, _, _, _ -> boardJson(count) },
        history,
        boardUrl = "http://localhost/",
    )

    @Test
    fun `the shards of one board are loaded concurrently`() {
        // If the loads were sequential this latch would never open and every
        // await would time out, which is exactly the old behaviour.
        val permits = SyntheticTimetable.MAX_CONCURRENT_SHARDS
        val latch = CountDownLatch(permits)
        val reachedTogether = AtomicInteger()
        val timetable = timetable(12, { _, _, _ ->
            latch.countDown()
            if (latch.await(10, TimeUnit.SECONDS)) reachedTogether.incrementAndGet()
            null
        })

        runBlocking { timetable.board("8000001", start, hours = 4) }
        assertTrue(
            "expected $permits shard loads to be in flight at once",
            reachedTogether.get() >= permits,
        )
    }

    @Test
    fun `no more shards are in flight than the permit count allows`() {
        val live = AtomicInteger()
        val peak = AtomicInteger()
        val timetable = timetable(20, { _, _, _ ->
            val now = live.incrementAndGet()
            peak.updateAndGet { maxOf(it, now) }
            Thread.sleep(5)
            live.decrementAndGet()
            null
        })

        runBlocking { timetable.board("8000001", start, hours = 4) }
        assertTrue("nothing ran concurrently at all", peak.get() > 1)
        assertTrue(
            "a public file host should not see ${peak.get()} parallel requests",
            peak.get() <= SyntheticTimetable.MAX_CONCURRENT_SHARDS,
        )
    }

    @Test
    fun `a train the caller will discard is never fetched`() {
        val asked = ConcurrentLinkedQueue<String>()
        val timetable = timetable(4, { category, number, _ ->
            asked += "$category $number"
            null
        })

        runBlocking {
            timetable.board("8000001", start, hours = 4) {
                DeutschlandTicket.covers(it.category)
            }
        }
        assertTrue("the ICE was filtered out, so its shard is wasted work",
            asked.none { it.startsWith("ICE") })
        assertEquals(4, asked.size)
    }

    @Test
    fun `without a filter every train on the board is fetched`() {
        val asked = ConcurrentLinkedQueue<String>()
        val timetable = timetable(4, { category, number, _ ->
            asked += "$category $number"
            null
        })
        runBlocking { timetable.board("8000001", start, hours = 4) }
        assertEquals(5, asked.size)
        assertTrue(asked.any { it.startsWith("ICE") })
    }

    @Test
    fun `the stop budget is spent on trains the filter keeps`() {
        // The filter runs before the cap, so a long-distance-heavy board does
        // not burn the budget on trains a Deutschland-Ticket search cannot use.
        val timetable = timetable(SyntheticTimetable.MAX_STOPS + 10, { _, _, _ -> null })
        val stops = runBlocking {
            timetable.board("8000001", start, hours = 24) {
                DeutschlandTicket.covers(it.category)
            }
        }
        assertEquals(SyntheticTimetable.MAX_STOPS, stops.size)
        assertTrue("the cap must not be spent on filtered-out trains",
            stops.none { it.label.category == "ICE" })
    }

    @Test
    fun `routes land on the stop they belong to`() {
        // The concurrent version must not shuffle results onto other stops.
        val timetable = timetable(6, { category, number, _ ->
            TrainHistory(
                "$category $number", category,
                mapOf(
                    "Somewhere" to StationHistory("8000002", listOf(run("09:00"))),
                    "Destination $number" to StationHistory("8000003", listOf(run("10:00"))),
                ),
            )
        })

        val stops = runBlocking { timetable.board("8000001", start, hours = 4) }
        assertEquals("six regional trains plus the ICE", 7, stops.size)
        for (stop in stops) {
            val path = stop.departure!!.plannedPath
            assertTrue(
                "${stop.label.display} got a route belonging to another train: $path",
                path.contains("Destination ${stop.label.number}"),
            )
        }
    }

    @Test
    fun `stops stay in departure order`() {
        val timetable = timetable(8, { _, _, _ -> null })
        val stops = runBlocking { timetable.board("8000001", start, hours = 4) }
        val times = stops.map { it.departure!!.plannedTime!! }
        assertEquals(times.sorted(), times)
    }

    @Test
    fun `a board that cannot be fetched is empty, not a crash`() {
        val timetable = SyntheticTimetable(
            ByteSource { _, _, _, _ -> null },
            { _, _, _ -> null },
            boardUrl = "http://localhost/",
        )
        assertEquals(emptyList<Any>(), runBlocking { timetable.board("8000001", start, 4) })
    }

    private fun run(tod: String) = io.github.derweh.bayesianbahn.model.HistoricalRun(
        date = today.minusDays(1),
        plannedTimeOfDay = tod,
        arrivalDelay = 0,
        departureDelay = 0,
        previousStopDelay = 0,
        cancelled = false,
    )
}
