package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DeutschlandTicketTest {

    @Test
    fun `regional trains are covered`() {
        for (cat in listOf("RE", "RB", "IRE", "S", "MEX", "BRB", "ALX", "ag", "ME", "HLB")) {
            assertTrue("$cat should be covered", DeutschlandTicket.covers(cat))
        }
    }

    @Test
    fun `long-distance trains are not covered`() {
        for (cat in listOf("ICE", "IC", "EC", "ECE", "RJ", "RJX", "NJ", "EN", "FLX", "TGV", "D", "WB")) {
            assertFalse("$cat should not be covered", DeutschlandTicket.covers(cat))
        }
    }

    @Test
    fun `coverage is case insensitive`() {
        assertFalse(DeutschlandTicket.covers("ice"))
        assertTrue(DeutschlandTicket.covers("re"))
    }
}
