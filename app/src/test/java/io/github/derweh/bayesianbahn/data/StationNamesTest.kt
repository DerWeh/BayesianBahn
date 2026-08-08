package io.github.derweh.bayesianbahn.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The journey search reported "No plannable trains from Ulm Hbf towards
 * Türkheim (Bay) Bahnhof found around that time" for a route served every
 * hour, because the three data sources spell that station three different ways
 * and the old matcher compared the raw strings.
 */
class StationNamesTest {

    // The exact strings the three sources produce for one station.
    private val fromStationList = "Türkheim (Bay) Bahnhof"  // bundled stations.csv
    private val fromIrisRoute = "Türkheim(Bay)Bf"           // IRIS `ppth` route entry
    private val fromHistory = "Türkheim (Bay)"              // delay-history shard

    @Test
    fun `the three spellings of one station all match`() {
        assertTrue(StationNames.matches(fromIrisRoute, fromStationList))
        assertTrue(StationNames.matches(fromHistory, fromStationList))
        assertTrue(StationNames.matches(fromIrisRoute, fromHistory))
    }

    @Test
    fun `matching is symmetric`() {
        assertTrue(StationNames.matches(fromStationList, fromIrisRoute))
        assertTrue(StationNames.matches(fromStationList, fromHistory))
    }

    @Test
    fun `run-together punctuation splits like spaced punctuation`() {
        assertEquals(listOf("turkheim", "bay", "bahnhof"), StationNames.tokens(fromIrisRoute))
        assertEquals(listOf("turkheim", "bay", "bahnhof"), StationNames.tokens(fromStationList))
    }

    @Test
    fun `designations are interchangeable with their abbreviations`() {
        assertTrue(StationNames.matches("Ulm Hbf", "Ulm Hauptbahnhof"))
        assertTrue(StationNames.matches("Ulm Hbf", "Ulm"))
        assertTrue(StationNames.matches("Stuttgart-Untertürkheim Pbf", "Stuttgart-Untertürkheim"))
        assertTrue(StationNames.matches("Nürnberg Hbf", "Nurnberg Hbf"))
    }

    /**
     * The reason this drops designations instead of matching substrings: every
     * one of these pairs is a *different* station, and `contains` matched them.
     */
    @Test
    fun `distinct stations that share a prefix do not match`() {
        assertFalse(StationNames.matches("Memmingen", "Memmingen Ost"))
        assertFalse(StationNames.matches("München Hbf", "München-Pasing"))
        assertFalse(StationNames.matches("Ulm Hbf", "Neu-Ulm"))
        assertFalse(StationNames.matches("Berlin Hbf", "Berlin Ostbahnhof"))
        assertFalse(StationNames.matches("Frankfurt(Main)Hbf", "Frankfurt(Main)Süd"))
        assertFalse(StationNames.matches("Türkheim (Bay) Bahnhof", "Türkheim (Bay) Ost"))
    }

    @Test
    fun `a bare designation matches nothing`() {
        assertFalse(StationNames.matches("Bahnhof", "Ulm Hbf"))
        assertFalse(StationNames.matches("", "Ulm Hbf"))
        assertFalse(StationNames.matches("", ""))
    }

    /**
     * Locks the fix against the real shipped data rather than a copy of it: the
     * bug was precisely that the asset and the API disagreed.
     */
    @Test
    fun `the bundled station list matches the live IRIS spelling`() {
        val csv = File("src/main/assets/stations.csv")
        assertTrue("missing asset: ${csv.absolutePath}", csv.exists())
        val names = csv.readLines().mapNotNull { line ->
            line.split(';').takeIf { it.size >= 3 }?.get(1)
        }
        val turkheim = names.firstOrNull { it.startsWith("Türkheim (Bay)") }
        assertEquals("Türkheim (Bay) Bahnhof", turkheim)
        assertTrue(StationNames.matches(fromIrisRoute, turkheim!!))

        // Ulm Hbf is spelled identically by both, and must stay distinct from
        // its neighbour across the Danube.
        val ulm = names.first { it == "Ulm Hbf" }
        assertTrue(StationNames.matches("Ulm Hbf", ulm))
        assertFalse(names.any { it != ulm && StationNames.matches(it, ulm) })
    }
}
