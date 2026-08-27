package io.github.derweh.bayesianbahn.ui

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * What the departure the search starts from actually is.
 *
 * The screen had no test for this at all, and the gap hid a rule nothing in
 * the UI disclosed: leaving the time on "now" and picking another date searched
 * from 06:00 on that date. Nothing looked broken — the app answered with a real
 * train, just the first one of a morning nobody had asked about.
 */
class DepartureTimeTest {

    private val zone = ZoneId.of("Europe/Berlin")
    private val now = ZonedDateTime.of(LocalDateTime.of(2026, 8, 27, 15, 20, 43), zone)
    private val saturday = LocalDate.of(2026, 8, 29).toEpochDay()

    private fun at(y: Int, mo: Int, d: Int, h: Int, mi: Int) =
        ZonedDateTime.of(LocalDateTime.of(y, mo, d, h, mi), zone).toInstant().toEpochMilli()

    @Test
    fun `now on another date is that date at the current time`() {
        assertEquals(
            at(2026, 8, 29, 15, 20),
            departMillis(hour = null, minute = null, epochDay = saturday, now = now),
        )
    }

    @Test
    fun `now on today is today at the current time`() {
        assertEquals(
            at(2026, 8, 27, 15, 20),
            departMillis(hour = null, minute = null, epochDay = null, now = now),
        )
    }

    @Test
    fun `now is truncated to the minute the button shows`() {
        // 15:20:43 must search from 15:20, not from 15:20:43 — a train leaving
        // at 15:20 has not gone yet as far as the timetable is concerned.
        assertEquals(
            at(2026, 8, 27, 15, 20),
            departMillis(null, null, null, now),
        )
    }

    @Test
    fun `a picked time wins on any date`() {
        assertEquals(
            at(2026, 8, 29, 7, 45),
            departMillis(hour = 7, minute = 45, epochDay = saturday, now = now),
        )
        assertEquals(
            at(2026, 8, 27, 7, 45),
            departMillis(hour = 7, minute = 45, epochDay = null, now = now),
        )
    }

    @Test
    fun `a picked hour without a minute is the full hour`() {
        assertEquals(
            at(2026, 8, 27, 7, 0),
            departMillis(hour = 7, minute = null, epochDay = null, now = now),
        )
    }

    @Test
    fun `midnight on another date is that date, not the next one`() {
        val midnight = ZonedDateTime.of(LocalDateTime.of(2026, 8, 27, 0, 5), zone)
        assertEquals(
            at(2026, 8, 29, 0, 5),
            departMillis(null, null, saturday, midnight),
        )
    }
}
