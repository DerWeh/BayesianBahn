package io.github.derweh.bayesianbahn.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate

/**
 * From the F-Droid review: a future-date search took about four minutes behind
 * a bare spinner, "which is hard to tell apart from a hang". The wait itself is
 * the historical-timetable fallback; this decides when to warn about it.
 */
class LoadingHintTest {

    private val today = LocalDate.of(2026, 8, 8)

    @Test
    fun `today and tomorrow are served by the live plan`() {
        assertFalse(beyondLivePlan(null, today))
        assertFalse(beyondLivePlan(today.toEpochDay(), today))
        assertFalse(beyondLivePlan(today.plusDays(1).toEpochDay(), today))
    }

    @Test
    fun `further out falls back to the historical timetable`() {
        assertTrue(beyondLivePlan(today.plusDays(2).toEpochDay(), today))
        assertTrue(beyondLivePlan(today.plusDays(30).toEpochDay(), today))
    }
}
