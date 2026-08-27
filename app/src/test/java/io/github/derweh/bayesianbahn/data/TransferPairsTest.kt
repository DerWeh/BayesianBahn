package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.EventInfo
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.api.TrainLabel
import io.github.derweh.bayesianbahn.data.JourneyPlanner.Companion.MAX_TRANSFER_ATTEMPTS
import io.github.derweh.bayesianbahn.data.JourneyPlanner.Companion.MAX_TRANSFER_RESULTS
import io.github.derweh.bayesianbahn.data.JourneyPlanner.Companion.rankTransferPairs
import io.github.derweh.bayesianbahn.data.JourneyPlanner.Companion.spendTransferBudget
import io.github.derweh.bayesianbahn.model.PointDistribution
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The order the attempt budget is spent in, and what it refuses to spend twice.
 *
 * The bug this replaces was invisible from inside the app: every itinerary it
 * showed was correct, and the ones it never looked for left no trace. At a big
 * station the first four departures consumed all eight attempts while forty
 * other trains were never considered, which `tools/route_bench.py` measures as
 * 57% recall against 73% for the order here.
 */
class TransferPairsTest {

    private val ulm = Station("8000170", "Ulm Hbf", 1000, 48.39965, 9.98240)
    private val turkheim = Station("8000144", "Türkheim (Bay) Bahnhof", 88, 48.06282, 10.63757)
    private val memmingen = Station("8003843", "Memmingen", 259, 47.99512, 10.18216)
    private val neuUlm = Station("8000271", "Neu-Ulm", 194, 48.39174, 9.99781)
    private val illertissen = Station("8003057", "Illertissen", 77, 48.22283, 10.09994)

    private fun feeder(id: String, minutes: Long) = TimetableStop(
        id = id,
        label = TrainLabel("RE", id, null),
        arrival = null,
        departure = EventInfo(
            plannedTime = minutes * 60_000,
            changedTime = null,
            plannedPlatform = null,
            changedPlatform = null,
            plannedPath = emptyList(),
            cancelled = false,
        ),
    )

    private fun itinerary(stop: TimetableStop, transfer: Station) = JourneyPlanner.Itinerary(
        feeder = stop,
        departureMillis = stop.departure!!.plannedTime!!,
        transferStation = transfer.name,
        distribution = PointDistribution(listOf(0.0 to 1.0)),
        referenceArrivalMillis = 0,
        catchProbability = null,
        missProbability = null,
        connection = null,
    )

    @Test
    fun `the nearest change wins, whichever train reaches it`() {
        val early = feeder("early", 100)
        val late = feeder("late", 160)
        val ranked = rankTransferPairs(
            routes = listOf(early to listOf(neuUlm), late to listOf(memmingen)),
            origin = ulm,
            destination = turkheim,
        )
        // The old order would have spent on Neu-Ulm first purely because its
        // train leaves an hour earlier.
        assertEquals(
            listOf("Memmingen" to "late", "Neu-Ulm" to "early"),
            ranked.map { (stop, transfer) -> transfer.name to stop.id },
        )
    }

    @Test
    fun `the same station is offered on its earliest train first`() {
        val early = feeder("early", 100)
        val late = feeder("late", 160)
        val ranked = rankTransferPairs(
            routes = listOf(late to listOf(memmingen), early to listOf(memmingen)),
            origin = ulm,
            destination = turkheim,
        )
        assertEquals(listOf("early", "late"), ranked.map { (stop, _) -> stop.id })
    }

    @Test
    fun `a station is kept for later trains, not collapsed to one pair`() {
        val early = feeder("early", 100)
        val late = feeder("late", 160)
        val ranked = rankTransferPairs(
            routes = listOf(early to listOf(memmingen, illertissen), late to listOf(memmingen)),
            origin = ulm,
            destination = turkheim,
        )
        assertEquals(3, ranked.size)
    }

    @Test
    fun `a board is opened once`() = runBlocking {
        val early = feeder("early", 100)
        val late = feeder("late", 160)
        val opened = mutableListOf<String>()
        val found = spendTransferBudget(
            listOf(early to memmingen, late to memmingen, late to illertissen),
        ) { _, transfer ->
            opened += transfer.name
            null
        }
        assertTrue(found.isEmpty())
        assertEquals(listOf("Memmingen", "Illertissen"), opened)
    }

    @Test
    fun `a feeder that already worked is not changed off twice`() = runBlocking {
        val stop = feeder("one", 100)
        val other = feeder("two", 160)
        val tried = mutableListOf<String>()
        val found = spendTransferBudget(
            listOf(stop to memmingen, stop to illertissen, other to neuUlm),
        ) { feeder, transfer ->
            tried += "${feeder.id}@${transfer.name}"
            itinerary(feeder, transfer)
        }
        assertEquals(listOf("one@Memmingen", "two@Neu-Ulm"), tried)
        assertEquals(listOf("Memmingen", "Neu-Ulm"), found.map { it.transferStation })
    }

    @Test
    fun `the budget bounds the evaluations, not the size of the list`() = runBlocking {
        val pairs = (0 until 50).map {
            feeder("f$it", 100 + it.toLong()) to Station("$it", "Station $it", 100, 48.0, 10.0)
        }
        var calls = 0
        val found = spendTransferBudget(pairs) { _, _ -> calls++; null }
        assertEquals(MAX_TRANSFER_ATTEMPTS, calls)
        assertTrue(found.isEmpty())
    }

    @Test
    fun `the search stops once it has enough itineraries`() = runBlocking {
        val pairs = (0 until 50).map {
            feeder("f$it", 100 + it.toLong()) to Station("$it", "Station $it", 100, 48.0, 10.0)
        }
        var calls = 0
        val found = spendTransferBudget(pairs) { stop, transfer ->
            calls++
            itinerary(stop, transfer)
        }
        assertEquals(MAX_TRANSFER_RESULTS, found.size)
        assertEquals(MAX_TRANSFER_RESULTS, calls)
    }
}
