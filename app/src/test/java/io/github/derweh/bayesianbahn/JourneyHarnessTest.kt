package io.github.derweh.bayesianbahn

import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * Tests for the one piece of arithmetic the journey harness adds.
 *
 * The model answers in minutes relative to a reference it picks itself — the
 * planned arrival of the first candidate it kept — while the truth and DB's
 * answer arrive as wall-clock minutes at the destination. Rebasing them crosses
 * the boundary that has already broken this pipeline once: journal times are
 * German wall clock stored as if UTC, and real epoch millis are not. Getting it
 * wrong shifts every journey by the German offset without looking broken, and
 * two hours of error would swamp the difference being measured.
 */
class JourneyHarnessTest {

    private val zone = ZoneId.of("Europe/Berlin")

    private fun reference(hour: Int, minute: Int): Long =
        ZonedDateTime.of(2026, 8, 25, hour, minute, 0, 0, zone).toInstant().toEpochMilli()

    /** Wall-clock minutes as the journal stores them: minutes since the epoch. */
    private fun wall(hour: Int, minute: Int, day: Int = 25): Int =
        ((ZonedDateTime.of(2026, 8, day, hour, minute, 0, 0, ZoneId.of("UTC"))
            .toEpochSecond()) / 60).toInt()

    @Test
    fun `an arrival exactly on the reference is zero`() {
        assertEquals(
            0.0,
            JourneyHarness.minutesFrom(reference(13, 10), wall(13, 10)),
            1e-9,
        )
    }

    @Test
    fun `a late arrival is positive minutes`() {
        assertEquals(
            7.0,
            JourneyHarness.minutesFrom(reference(13, 10), wall(13, 17)),
            1e-9,
        )
    }

    @Test
    fun `catching an earlier train than the reference is negative`() {
        // The model allows it: a delayed earlier candidate that is still at the
        // platform gets boarded, and it arrives before the reference does.
        assertEquals(
            -25.0,
            JourneyHarness.minutesFrom(reference(13, 10), wall(12, 45)),
            1e-9,
        )
    }

    @Test
    fun `the summer offset is applied once and only once`() {
        // Reading the wall clock as real UTC would put this two hours out, and
        // the number would still look like a plausible delay.
        assertEquals(
            0.0,
            JourneyHarness.minutesFrom(reference(2, 30), wall(2, 30)),
            1e-9,
        )
    }

    @Test
    fun `an arrival after midnight is not read as the same morning`() {
        assertEquals(
            50.0,
            JourneyHarness.minutesFrom(
                ZonedDateTime.of(2026, 8, 25, 23, 40, 0, 0, zone).toInstant().toEpochMilli(),
                wall(0, 30, day = 26),
            ),
            1e-9,
        )
    }

    @Test
    fun `the transfer time matches the one the event builder assumed`() {
        // Python resolves which train was caught; Kotlin resolves what the
        // model predicts. Two different transfer times would have them scoring
        // different journeys, with nothing in the output to say so.
        assertEquals(5, JourneyHarness.TRANSFER_MINUTES)
    }
}
