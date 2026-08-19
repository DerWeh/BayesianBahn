package io.github.derweh.bayesianbahn.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * A train one minute early was labelled "+-1" on the arrivals board: the "+"
 * was part of the template, so a negative delay carried both signs. Trains do
 * run early, and the board has to say so.
 */
class DelayFormatTest {

    @Test
    fun `a late train keeps its plus`() {
        assertEquals("+1", signedMinutes(1.0))
        assertEquals("+12", signedMinutes(12.0))
    }

    @Test
    fun `an early train reads as negative, not as plus minus`() {
        assertEquals("-1", signedMinutes(-1.0))
        assertEquals("-4", signedMinutes(-4.0))
    }

    @Test
    fun `no delay is written without a sign`() {
        assertEquals("0", signedMinutes(0.0))
    }

    @Test
    fun `minutes are rounded, not truncated`() {
        // Truncation would show a train 1.6 minutes late as "+1" and one 0.6
        // minutes early as "0".
        assertEquals("+2", signedMinutes(1.6))
        assertEquals("-1", signedMinutes(-0.6))
    }

    @Test
    fun `the chip is hidden when the train is on time`() {
        assertNull(delayChip(0.0))
        assertNull(delayChip(null))
    }

    @Test
    fun `a delay that rounds away is on time, not plus zero`() {
        assertNull(delayChip(0.4))
        assertNull(delayChip(-0.4))
    }

    @Test
    fun `the chip carries the sign of the delay`() {
        assertEquals("+3", delayChip(3.0))
        assertEquals("-1", delayChip(-1.0))
    }
}
