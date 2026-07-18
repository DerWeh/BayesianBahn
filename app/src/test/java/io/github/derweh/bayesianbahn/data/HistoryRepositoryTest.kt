package io.github.derweh.bayesianbahn.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate

class HistoryRepositoryTest {

    @Test
    fun `shard keys mirror the pipeline`() {
        assertEquals("ICE_512", HistoryRepository.shardKey("ICE 512"))
        assertEquals("RB_86", HistoryRepository.shardKey(" RB 86 "))
        assertEquals("S_31648", HistoryRepository.shardKey("S 31648"))
    }

    @Test
    fun `parses a pipeline shard`() {
        val json = """
            {"stations":{"Augsburg Hbf":{"eva":"8000013","runs":[
                ["2026-06-28","17:59",12,11,11,0],
                ["2026-06-29","17:59",null,4,null,1]
            ]}},"train":"ICE 512","type":"ICE"}
        """.trimIndent()
        val history = HistoryRepository.parseShard(json)!!
        assertEquals("ICE 512", history.trainName)
        assertEquals("ICE", history.trainType)

        val station = history.stations.getValue("Augsburg Hbf")
        assertEquals("8000013", station.eva)
        assertEquals(2, station.runs.size)

        val first = station.runs[0]
        assertEquals(LocalDate.of(2026, 6, 28), first.date)
        assertEquals("17:59", first.plannedTimeOfDay)
        assertEquals(12, first.arrivalDelay)
        assertEquals(11, first.previousStopDelay)

        val second = station.runs[1]
        assertNull(second.arrivalDelay)
        assertTrue(second.cancelled)
    }

    @Test
    fun `garbage json returns null`() {
        assertNull(HistoryRepository.parseShard("not json"))
        assertNull(HistoryRepository.parseShard("{}"))
    }
}
