package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate

class GermanCalendarTest {

    @Test
    fun `easter computus is correct`() {
        assertEquals(LocalDate.of(2026, 4, 5), GermanCalendar.easterSunday(2026))
        assertEquals(LocalDate.of(2025, 4, 20), GermanCalendar.easterSunday(2025))
        assertEquals(LocalDate.of(2027, 3, 28), GermanCalendar.easterSunday(2027))
    }

    @Test
    fun `nationwide holidays are recognised`() {
        for (d in listOf(
            "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01",
            "2026-05-14", "2026-05-25", "2026-10-03", "2026-12-25", "2026-12-26",
        )) {
            assertTrue(d, GermanCalendar.isNationwideHoliday(LocalDate.parse(d)))
        }
        assertFalse(GermanCalendar.isNationwideHoliday(LocalDate.parse("2026-07-24")))
        // Fronleichnam is regional, deliberately not folded.
        assertFalse(GermanCalendar.isNationwideHoliday(LocalDate.parse("2026-06-04")))
    }

    @Test
    fun `next timetable change is the coming second Sunday of June or December`() {
        assertEquals(
            LocalDate.of(2026, 12, 13),
            GermanCalendar.nextTimetableChange(LocalDate.of(2026, 7, 19)),
        )
        assertEquals(
            LocalDate.of(2026, 6, 14),
            GermanCalendar.nextTimetableChange(LocalDate.of(2026, 6, 1)),
        )
        assertEquals(
            LocalDate.of(2027, 6, 13),
            GermanCalendar.nextTimetableChange(LocalDate.of(2026, 12, 13)),
        )
    }

    @Test
    fun `holiday runs the Sunday timetable in the synthetic board`() {
        // 2026-10-03 is a Saturday and a nationwide holiday: a Mon-Fri train
        // must not appear, a Sunday train must.
        val json = """{"name":"X","trains":[
            ["RE","1","R1",null,510,31,20720],
            ["RE","2","R2",null,520,64,20720]]}"""
        val entries = io.github.derweh.bayesianbahn.data.SyntheticTimetable.parseBoard(json)
        val start = LocalDate.parse("2026-10-03").atStartOfDay(java.time.ZoneId.of("Europe/Berlin"))
            .plusHours(8).toInstant().toEpochMilli()
        val stops = io.github.derweh.bayesianbahn.data.SyntheticTimetable.plan(
            entries, start, 2, today = LocalDate.parse("2026-09-20"),
        )
        assertEquals(listOf("2"), stops.map { it.first.label.number })
    }
}
