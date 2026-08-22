package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The rule deciding when DB's live number is evidence. Both the arrival
 * forecast and the connection model enforce it themselves, so this is the one
 * place the boundary is stated.
 */
class LiveReportTest {

    @Test
    fun `no report is not evidence`() {
        assertNull(LiveReport.informative(null))
    }

    @Test
    fun `a report of on time is not evidence`() {
        assertNull(LiveReport.informative(0.0))
    }

    @Test
    fun `a report of running early is not evidence`() {
        // Trains DB called early averaged 1.4 minutes late over 2026-08-17..19.
        assertNull(LiveReport.informative(-1.0))
        assertNull(LiveReport.informative(-30.0))
    }

    @Test
    fun `a delay below a whole minute is not evidence`() {
        assertNull(LiveReport.informative(0.9))
    }

    @Test
    fun `a delay of exactly the threshold is evidence`() {
        assertEquals(
            LiveReport.MIN_INFORMATIVE_DELAY_MINUTES,
            LiveReport.informative(LiveReport.MIN_INFORMATIVE_DELAY_MINUTES)!!,
            1e-9,
        )
    }

    @Test
    fun `a reported delay is passed through unchanged`() {
        assertEquals(30.0, LiveReport.informative(30.0)!!, 1e-9)
    }
}
