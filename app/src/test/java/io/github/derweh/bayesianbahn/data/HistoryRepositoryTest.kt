package io.github.derweh.bayesianbahn.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
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
    fun `line keys mirror the pipeline, and carry the station`() {
        // IRIS writes the product into the line, so it is not repeated...
        assertEquals("S7_8089105", HistoryRepository.lineKey("S", "S7", "8089105"))
        assertEquals("RE9_8000013", HistoryRepository.lineKey("RE", "RE9", "8000013"))
        // ...but the train's own product is a different thing: a replacement
        // bus on the S7 must not read the trains' history.
        assertEquals("BUS_S7_8089105", HistoryRepository.lineKey("Bus", "S7", "8089105"))
        assertEquals("HLB_RB90_8000037", HistoryRepository.lineKey("HLB", "RB90", "8000037"))
        // A line shard is per station: "S1" alone names eight networks.
        assertNotEquals(
            HistoryRepository.lineKey("S", "S1", "8089105"),
            HistoryRepository.lineKey("S", "S1", "8000261"),
        )
    }

    @Test
    fun `a shard records the line it runs`() {
        // The board names a line for about a sixth of stops; this is where the
        // fallback gets one for the rest.
        val json = """
            {"train":"RE 4711","type":"RE","line":"RE9","stations":{"Ulm Hbf":
            {"eva":"8000170","days":[20000],"tod":[480],"a":[3],"p":[2]}}}
        """.trimIndent()
        assertEquals("RE9", HistoryRepository.parseShard(json)!!.line)
        // Older shards have no line field and must still parse.
        assertNull(
            HistoryRepository.parseShard(
                """{"train":"RE 4711","type":"RE","stations":{}}""",
            )?.line,
        )
    }

    @Test
    fun `merging keeps the line whichever tier records it`() {
        val base = TrainHistory("RE 4711", "RE", emptyMap())
        val recent = TrainHistory("RE 4711", "RE", emptyMap(), line = "RE9")
        assertEquals("RE9", HistoryRepository.mergeHistories(base, recent)?.line)
        assertEquals("RE9", HistoryRepository.mergeHistories(recent, base)?.line)
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
