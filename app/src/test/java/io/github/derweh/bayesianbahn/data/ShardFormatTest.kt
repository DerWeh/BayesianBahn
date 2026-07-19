package io.github.derweh.bayesianbahn.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate

class ShardFormatTest {

    @Test
    fun `parses v2 columnar shard`() {
        // Two runs on consecutive days at 06:23 (epoch day 20500 = 2026-02-17),
        // one at 17:59; second run cancelled; dep differs from arr once.
        val json = """
            {"v":2,"train":"RE 1","type":"RE","stations":{
              "Augsburg Hbf":{"eva":"8000013","tod":[383,1079],
                "days":[20500,1,0],"t":[0,0,1],
                "a":[3,null,7],"d":[5,null,null],"p":[2,null,6],"c":[1]}}}
        """.trimIndent()
        val history = HistoryRepository.parseShard(json)!!
        val runs = history.stations.getValue("Augsburg Hbf").runs
        assertEquals(3, runs.size)

        assertEquals(LocalDate.ofEpochDay(20500), runs[0].date)
        assertEquals("06:23", runs[0].plannedTimeOfDay)
        assertEquals(3, runs[0].arrivalDelay)
        assertEquals(5, runs[0].departureDelay)
        assertEquals(2, runs[0].previousStopDelay)
        assertTrue(!runs[0].cancelled)

        assertEquals(LocalDate.ofEpochDay(20501), runs[1].date)
        assertTrue(runs[1].cancelled)
        assertNull(runs[1].arrivalDelay)

        assertEquals(LocalDate.ofEpochDay(20501), runs[2].date)
        assertEquals("17:59", runs[2].plannedTimeOfDay)
        // "d" null falls back to the arrival delay.
        assertEquals(7, runs[2].departureDelay)
    }

    @Test
    fun `single planned time omits the t array`() {
        val json = """
            {"v":2,"train":"S 1","type":"S","stations":{
              "München Ost":{"eva":"8000262","tod":[500],
                "days":[20500,7],"a":[1,2],"p":[0,1]}}}
        """.trimIndent()
        val runs = HistoryRepository.parseShard(json)!!
            .stations.getValue("München Ost").runs
        assertEquals(2, runs.size)
        assertEquals("08:20", runs[0].plannedTimeOfDay)
        assertEquals("08:20", runs[1].plannedTimeOfDay)
        assertEquals(LocalDate.ofEpochDay(20507), runs[1].date)
        // no "d": departure falls back to arrival
        assertEquals(2, runs[1].departureDelay)
        assertEquals(0.0, runs.count { it.cancelled }.toDouble(), 0.0)
    }

    @Test
    fun `still parses the legacy v1 row format`() {
        val json = """
            {"train":"RE 1","type":"RE","stations":{
              "Ulm Hbf":{"eva":"8000170","runs":[["2026-03-02","06:23",1,2,0,0]]}}}
        """.trimIndent()
        val runs = HistoryRepository.parseShard(json)!!.stations.getValue("Ulm Hbf").runs
        assertEquals(1, runs.size)
        assertEquals(LocalDate.parse("2026-03-02"), runs[0].date)
        assertEquals("06:23", runs[0].plannedTimeOfDay)
        assertEquals(2, runs[0].departureDelay)
    }
}
