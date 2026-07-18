package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.HistoricalRun
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test
import java.time.LocalDate

class HistoryMergeTest {

    private fun run(date: String, tod: String, delay: Int) = HistoricalRun(
        date = LocalDate.parse(date),
        plannedTimeOfDay = tod,
        arrivalDelay = delay,
        departureDelay = delay,
        previousStopDelay = delay,
        cancelled = false,
    )

    private fun history(vararg stations: Pair<String, List<HistoricalRun>>) = TrainHistory(
        trainName = "ICE 1",
        trainType = "ICE",
        stations = stations.associate { (name, runs) -> name to StationHistory("8000013", runs) },
    )

    @Test
    fun `recent runs are appended to base runs`() {
        val base = history("Augsburg Hbf" to listOf(run("2026-06-01", "10:00", 3)))
        val recent = history("Augsburg Hbf" to listOf(run("2026-07-17", "10:00", 8)))
        val merged = HistoryRepository.mergeHistories(base, recent)!!
        assertEquals(2, merged.stations.getValue("Augsburg Hbf").runs.size)
    }

    @Test
    fun `recent wins on the same date and planned time`() {
        val base = history("Augsburg Hbf" to listOf(run("2026-07-17", "10:00", 3)))
        val recent = history("Augsburg Hbf" to listOf(run("2026-07-17", "10:00", 8)))
        val merged = HistoryRepository.mergeHistories(base, recent)!!
        val runs = merged.stations.getValue("Augsburg Hbf").runs
        assertEquals(1, runs.size)
        assertEquals(8, runs.single().arrivalDelay)
    }

    @Test
    fun `stations only in one side survive`() {
        val base = history("Augsburg Hbf" to listOf(run("2026-06-01", "10:00", 3)))
        val recent = history("München Hbf" to listOf(run("2026-07-17", "11:00", 8)))
        val merged = HistoryRepository.mergeHistories(base, recent)!!
        assertEquals(setOf("Augsburg Hbf", "München Hbf"), merged.stations.keys)
    }

    @Test
    fun `null sides pass through`() {
        val base = history("Augsburg Hbf" to listOf(run("2026-06-01", "10:00", 3)))
        assertSame(base, HistoryRepository.mergeHistories(base, null))
        assertSame(base, HistoryRepository.mergeHistories(null, base))
        assertEquals(null, HistoryRepository.mergeHistories(null, null))
    }
}
