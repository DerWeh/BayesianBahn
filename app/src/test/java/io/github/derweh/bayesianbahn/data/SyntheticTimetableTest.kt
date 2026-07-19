package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.HistoricalRun
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime

class SyntheticTimetableTest {

    private val zone = ZoneId.of("Europe/Berlin")

    // Friday 2026-07-24 (weekday bit 1 shl 4 = 16); Mon-Fri mask = 31.
    private val friday: LocalDate = LocalDate.of(2026, 7, 24)
    private val boardJson = """
        {"name":"Türkheim (Bay)","trains":[
          ["RE","57520","RE71",514,520,31,20649],
          ["ARV","78905","RE72",600,601,96,20649],
          ["RB","1","RB6",null,510,127,20500]
        ]}
    """.trimIndent()

    private fun millis(date: LocalDate, hour: Int, minute: Int): Long =
        ZonedDateTime.of(date.atTime(hour, minute), zone).toInstant().toEpochMilli()

    @Test
    fun `plans weekday-matching entries within the window`() {
        val entries = SyntheticTimetable.parseBoard(boardJson)
        assertEquals(3, entries.size)
        val stops = SyntheticTimetable.plan(
            entries,
            startMillis = millis(friday, 8, 0),
            hours = 3,
            today = LocalDate.of(2026, 7, 19),
        )
        // RE71 (Mon-Fri, dep 08:40) matches; RE72 (Sat/Sun mask 96) does not;
        // RB 1 was last seen too long ago.
        assertEquals(1, stops.size)
        val (stop, depTod) = stops.single()
        assertEquals("RE", stop.label.category)
        assertEquals(millis(friday, 8, 40), stop.departure!!.plannedTime)
        assertEquals(millis(friday, 8, 34), stop.arrival!!.plannedTime)
        assertEquals(520, depTod)
    }

    @Test
    fun `weekend entry appears on Saturday`() {
        val entries = SyntheticTimetable.parseBoard(boardJson)
        val saturday = friday.plusDays(1)
        val stops = SyntheticTimetable.plan(
            entries,
            startMillis = millis(saturday, 8, 0),
            hours = 4,
            today = LocalDate.of(2026, 7, 19),
        )
        assertEquals(listOf("ARV"), stops.map { it.first.label.category })
    }

    @Test
    fun `route derives from the shard ordered by time after the stop`() {
        fun run(tod: String) = HistoricalRun(
            date = LocalDate.of(2026, 7, 1),
            plannedTimeOfDay = tod,
            arrivalDelay = 0,
            departureDelay = 0,
            previousStopDelay = 0,
            cancelled = false,
        )
        val history = TrainHistory(
            "RE 1", "RE",
            stations = mapOf(
                "Türkheim (Bay)" to StationHistory("8000144", listOf(run("08:40"))),
                "Mindelheim" to StationHistory("8000338", listOf(run("08:50"))),
                "Memmingen" to StationHistory("8000249", listOf(run("09:10"))),
                "Buchloe" to StationHistory("8000057", listOf(run("08:25"))),
            ),
        )
        val path = SyntheticTimetable.pathAfter(history, "8000144", 520)
        // Buchloe lies before the stop (08:25) — excluded; rest ordered.
        assertEquals(listOf("Mindelheim", "Memmingen"), path)
        assertTrue("origin not in its own path", "Türkheim (Bay)" !in path)
    }
}
