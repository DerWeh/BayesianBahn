package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.model.HistoricalRun
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * Tests for the candidate construction the app and the journey harness share.
 *
 * It was lifted out of [ConnectionPlanner] so the harness would not describe it
 * a second time; a second description is how the Python mirror of the arrival
 * model came to be missing `WB` while still producing plausible numbers. What
 * is pinned here is what a second description would get subtly wrong: that
 * departure and arrival come from the *same* historical run, that a candidate
 * with too little history is refused rather than guessed at, and that a station
 * is matched by eva before name.
 */
class CandidateBuilderTest {

    private val zone = ZoneId.of("Europe/Berlin")
    private val today = LocalDate.of(2026, 8, 25)

    private fun millis(hour: Int, minute: Int, day: Int = 25): Long =
        ZonedDateTime.of(2026, 8, day, hour, minute, 0, 0, zone).toInstant().toEpochMilli()

    private fun run(dayBack: Long, hhmm: String, dep: Int?, arr: Int?,
                    cancelled: Boolean = false) = HistoricalRun(
        date = today.minusDays(dayBack),
        plannedTimeOfDay = hhmm,
        arrivalDelay = arr,
        departureDelay = dep,
        previousStopDelay = null,
        cancelled = cancelled,
    )

    private fun history(
        transfer: List<HistoricalRun>,
        destination: List<HistoricalRun>,
        transferEva: String? = "8000001",
        destinationEva: String? = "8000999",
    ) = TrainHistory(
        trainName = "RE 2", trainType = "RE",
        stations = mapOf(
            "Transferhausen" to StationHistory(transferEva, transfer),
            "Musterstadt" to StationHistory(destinationEva, destination),
        ),
    )

    private fun build(
        history: TrainHistory?,
        live: Double? = null,
        cancelledLive: Boolean = false,
    ) = CandidateBuilder.build(
        history = history,
        id = "c1", label = "RE 2",
        transferEva = "8000001", transferName = "Transferhausen",
        destinationEva = "8000999", destinationName = "Musterstadt",
        plannedDepartureMillis = millis(12, 10),
        liveDepartureDelay = live,
        cancelledLive = cancelledLive,
        today = today,
    )

    private fun fullHistory(pairs: List<Pair<Int, Int>>) = history(
        transfer = pairs.mapIndexed { i, (dep, _) ->
            run(i + 1L, "12:10", dep, dep)
        },
        destination = pairs.mapIndexed { i, (_, arr) ->
            run(i + 1L, "13:10", arr, arr)
        },
    )

    @Test
    fun `departure and arrival come from the same run`() {
        // Pairing across runs would keep the marginals and destroy the
        // correlation, which is the whole reason a two-leg journey is not two
        // one-leg journeys.
        val got = build(fullHistory(listOf(0 to 0, 5 to 9, 1 to 2, 12 to 20, 3 to 4)))!!
        assertEquals(
            listOf(0.0 to 0.0, 5.0 to 9.0, 1.0 to 2.0, 12.0 to 20.0, 3.0 to 4.0),
            got.runs.map { it.departureDelay to it.arrivalDelay },
        )
    }

    @Test
    fun `a run missing at the destination contributes no pair`() {
        val got = build(
            history(
                transfer = (1..6).map { run(it.toLong(), "12:10", it, it) },
                destination = (1..6).mapNotNull {
                    if (it == 3) null else run(it.toLong(), "13:10", it, it)
                },
            ),
        )!!
        assertEquals(5, got.runs.size)
        assertTrue(got.runs.none { it.departureDelay == 3.0 })
    }

    @Test
    fun `too little joint history and no live report is refused`() {
        // Not "predicted from four runs": a candidate the model cannot speak
        // for has to be dropped, or the mixture quietly rests on nothing.
        assertNull(build(fullHistory(listOf(0 to 0, 1 to 1, 2 to 2, 3 to 3))))
    }

    @Test
    fun `a live report carries a candidate with no history at all`() {
        val got = build(history(transfer = emptyList(), destination = emptyList()),
            live = 4.0)
        assertNull("no historical arrival means no planned arrival to recover", got)
    }

    @Test
    fun `a cancelled historical run is not a usable pair`() {
        val runs = (1..6).map { run(it.toLong(), "12:10", it, it, cancelled = it == 2) }
        val got = build(
            history(transfer = runs,
                destination = (1..6).map { run(it.toLong(), "13:10", it, it) }),
        )!!
        assertEquals(5, got.runs.size)
    }

    @Test
    fun `the cancellation rate comes from the transfer runs`() {
        // Ten runs, five of them cancelled: the rate is a half, and the five
        // that ran are exactly MIN_JOINT_RUNS, so the candidate still stands.
        val runs = (1..10).map { run(it.toLong(), "12:10", it, it, cancelled = it <= 5) }
        val got = build(
            history(transfer = runs,
                destination = (1..10).map { run(it.toLong(), "13:10", it, it) }),
        )!!
        assertEquals(0.5, got.cancelRate, 1e-9)
        assertEquals(CandidateBuilder.MIN_JOINT_RUNS, got.runs.size)
    }

    @Test
    fun `the station is matched by eva even when the name differs`() {
        val got = CandidateBuilder.build(
            history = fullHistory(listOf(0 to 0, 1 to 1, 2 to 2, 3 to 3, 4 to 4)),
            id = "c1", label = "RE 2",
            transferEva = "8000001", transferName = "quite another name",
            destinationEva = "8000999", destinationName = "and another",
            plannedDepartureMillis = millis(12, 10),
            liveDepartureDelay = null, cancelledLive = false, today = today,
        )
        assertEquals(5, got!!.runs.size)
    }

    @Test
    fun `no history at all is refused rather than invented`() {
        assertNull(build(null))
    }

    @Test
    fun `the planned arrival is the departure plus the leg the runs took`() {
        val got = build(fullHistory(listOf(0 to 0, 1 to 1, 2 to 2, 3 to 3, 4 to 4)))!!
        assertEquals(millis(13, 10), got.plannedArrivalMillis)
    }

    // --- the leg, and the day-late arrival it replaced ------------------------
    //
    // The old rule hung the destination's most recent *time of day* on today's
    // date, and rolled the date forward whenever that landed before the
    // departure. A timetable that shifted by half an hour was enough: on one
    // collected day, four candidates in six were published as arriving 24 hours
    // and 28 minutes later instead of 28 minutes later. Departure and arrival
    // now come from the same run, so a shifted schedule moves both.

    @Test
    fun `a schedule that shifted since the last run does not cost a day`() {
        // Every run left at 19:23 and arrived at 19:51; today's train leaves at
        // 19:51. The old rule read 19:51 - 19:23 as "already gone, so tomorrow".
        val runs = (1..6).map { run(it.toLong(), "19:23", 0, 0) }
        val dest = (1..6).map { run(it.toLong(), "19:51", 0, 0) }
        val got = CandidateBuilder.plannedArrival(runs, dest, millis(19, 51))!!
        assertEquals(millis(20, 19), got)
    }

    @Test
    fun `an overnight leg still crosses midnight`() {
        val runs = (1..6).map { run(it.toLong(), "23:40", 0, 0) }
        val dest = (1..6).map { run(it.toLong(), "00:20", 0, 0) }
        assertEquals(
            millis(0, 20, day = 26),
            CandidateBuilder.plannedArrival(runs, dest, millis(23, 40)),
        )
    }

    @Test
    fun `the leg is the median, so one odd run cannot set it`() {
        val runs = (1..5).map { run(it.toLong(), "12:10", 0, 0) }
        val dest = listOf(
            run(1, "13:10", 0, 0), run(2, "13:10", 0, 0), run(3, "13:10", 0, 0),
            run(4, "13:10", 0, 0), run(5, "22:10", 0, 0),
        )
        assertEquals(
            millis(13, 10),
            CandidateBuilder.plannedArrival(runs, dest, millis(12, 10)),
        )
    }

    @Test
    fun `a leg longer than any real one is refused rather than published`() {
        val runs = (1..6).map { run(it.toLong(), "12:10", 0, 0) }
        val dest = (1..6).map { run(it.toLong(), "11:10", 0, 0) }   // 23 hours
        assertNull(CandidateBuilder.plannedArrival(runs, dest, millis(12, 10)))
    }

    @Test
    fun `a leg is measured from the same run at both ends`() {
        assertEquals(28, CandidateBuilder.legMinutes("19:23", "19:51"))
        assertEquals(40, CandidateBuilder.legMinutes("23:40", "00:20"))
        assertEquals(0, CandidateBuilder.legMinutes("12:00", "12:00"))
    }

    @Test
    fun `no run pairs at both ends means no arrival to predict`() {
        val runs = listOf(run(1, "12:10", 0, 0))
        val dest = listOf(run(9, "13:10", 0, 0))
        assertNull(CandidateBuilder.plannedArrival(runs, dest, millis(12, 10)))
    }
}
