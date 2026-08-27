package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.data.JourneyPlanner.Companion.transferCandidates
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Ulm Hbf → Türkheim (Bay) at 22:00 found one connection, via Memmingen, but
 * only on the fourth of six allowed attempts: transfers were ranked by station
 * size, so the budget went on Stuttgart Hbf (130 km the wrong way), Göppingen
 * and Neu-Ulm first. One more large station on any feeder's route and the
 * search would have reported "no plannable trains" for a route that runs.
 */
class TransferCandidatesTest {

    // Coordinates as shipped in stations.csv.
    private val ulm = Station("8000170", "Ulm Hbf", 693, 48.39944, 9.98223)
    private val turkheim = Station("8000144", "Türkheim (Bay) Bahnhof", 69, 48.04569, 10.61744)
    private val stuttgart = Station("8000096", "Stuttgart Hbf", 1009, 48.78408, 9.18163)
    private val goppingen = Station("8000127", "Göppingen", 263, 48.70003, 9.65197)
    private val neuUlm = Station("8006730", "Neu-Ulm", 260, 48.39304, 10.00482)
    private val memmingen = Station("8000249", "Memmingen", 122, 47.98558, 10.18685)
    private val illertissen = Station("8003057", "Illertissen", 77, 48.22283, 10.09994)

    @Test
    fun `the useful change is offered first, not the biggest station`() {
        val ranked = transferCandidates(
            path = listOf(stuttgart, goppingen, neuUlm, memmingen, illertissen),
            origin = ulm,
            destination = turkheim,
        )
        assertEquals("Memmingen", ranked.first().name)
    }

    @Test
    fun `stations leading away from the destination are dropped`() {
        val ranked = transferCandidates(
            path = listOf(stuttgart, goppingen, neuUlm, memmingen),
            origin = ulm,
            destination = turkheim,
        ).map { it.name }
        assertTrue("Stuttgart Hbf is 130 km the wrong way", "Stuttgart Hbf" !in ranked)
        assertTrue("Göppingen is the wrong way", "Göppingen" !in ranked)
        assertEquals(listOf("Memmingen", "Neu-Ulm"), ranked)
    }

    /**
     * The reason for a tolerance above 1.0: from a small station the only way
     * out is often a hub slightly behind you.
     */
    @Test
    fun `a hub just behind the origin stays available`() {
        val ranked = transferCandidates(
            path = listOf(neuUlm),
            origin = ulm,
            destination = turkheim,
        )
        assertEquals(listOf("Neu-Ulm"), ranked.map { it.name })
    }

    @Test
    fun `village halts are still skipped`() {
        val halt = Station("8000000", "Kleinkleckersdorf", 5, 48.05, 10.60)
        val ranked = transferCandidates(listOf(halt, memmingen), ulm, turkheim)
        assertEquals(listOf("Memmingen"), ranked.map { it.name })
    }

    @Test
    fun `the destination itself is not a transfer`() {
        assertTrue(transferCandidates(listOf(turkheim), ulm, turkheim).isEmpty())
    }

    @Test
    fun `without coordinates the old weight ranking is used`() {
        val a = Station("1", "Big", 900)
        val b = Station("2", "Small", 100)
        val noCoords = Station("3", "Nowhere", 50)
        assertEquals(
            listOf("Big", "Small"),
            transferCandidates(listOf(b, a), Station("4", "From", 100), noCoords).map { it.name },
        )
    }

    @Test
    fun `stations missing coordinates rank after the located ones`() {
        val unlocated = Station("9", "Unknown", 800)
        val ranked = transferCandidates(listOf(unlocated, memmingen), ulm, turkheim)
        assertEquals(listOf("Memmingen", "Unknown"), ranked.map { it.name })
    }

    @Test
    fun `every shipped station carries coordinates`() {
        val csv = File("src/main/assets/stations.csv")
        assertTrue(csv.exists())
        val bad = csv.readLines().filter { line ->
            val p = line.split(';')
            p.size < 5 || p[3].toDoubleOrNull() == null || p[4].toDoubleOrNull() == null
        }
        assertEquals("stations without usable coordinates: ${bad.take(3)}", 0, bad.size)
    }

    @Test
    fun `distance is a real great-circle distance`() {
        // Ulm - Memmingen is about 48 km as the crow flies (55 by road).
        val d = ulm.distanceKm(memmingen)!!
        assertTrue("got $d km", d in 45.0..52.0)
        assertEquals(0.0, ulm.distanceKm(ulm)!!, 1e-9)
        assertEquals(null, ulm.distanceKm(Station("x", "No coords", 10)))
    }
}
