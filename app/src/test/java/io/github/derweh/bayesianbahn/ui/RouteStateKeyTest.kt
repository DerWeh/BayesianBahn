package io.github.derweh.bayesianbahn.ui

import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.api.TrainLabel
import io.github.derweh.bayesianbahn.data.Station
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Only the current route is composed, so each screen's `rememberSaveable` state
 * is retained against these keys and handed back when the screen returns.
 * Getting them wrong is invisible until a user notices — as one did, when
 * opening a station board emptied the From and To fields behind it.
 */
class RouteStateKeyTest {

    private fun station(eva: String, name: String = "Somewhere") = Station(eva, name, weight = 1)

    private fun stop(id: String) = TimetableStop(
        id = id,
        label = TrainLabel("RE", "1", "RE1"),
        arrival = null,
        departure = null,
    )

    @Test
    fun `the home screen keeps one key for the whole session`() {
        // If this changed between visits, From and To would be blank every time
        // — which is the bug this exists to prevent.
        assertEquals(Route.Journey.key, Route.Journey.key)
        assertEquals("journey", Route.Journey.key)
    }

    @Test
    fun `returning to the same station is the same screen`() {
        assertEquals(
            Route.Board(station("8000001")).key,
            Route.Board(station("8000001")).key,
        )
    }

    @Test
    fun `a different station is a different screen`() {
        assertNotEquals(
            Route.Board(station("8000001")).key,
            Route.Board(station("8000105")).key,
        )
    }

    @Test
    fun `the same station renamed is still the same screen`() {
        // Keys are built from the eva, not the display name, so IRIS spelling a
        // station differently does not orphan its state.
        assertEquals(
            Route.Board(station("8000001", "Aachen Hbf")).key,
            Route.Board(station("8000001", "Aachen Hauptbahnhof")).key,
        )
    }

    @Test
    fun `predictions for different trains do not share state`() {
        val here = station("8000001")
        assertNotEquals(
            Route.Prediction(here, stop("ice-512")).key,
            Route.Prediction(here, stop("re-1")).key,
        )
    }

    @Test
    fun `screens of different kinds never collide`() {
        val here = station("8000001")
        val one = stop("ice-512")
        val keys = listOf(
            Route.Journey.key,
            Route.Search.key,
            Route.Board(here).key,
            Route.Prediction(here, one).key,
            Route.Connection(here, one).key,
        )
        assertEquals("every screen needs its own state", keys.size, keys.toSet().size)
    }

    // --- forgetting -----------------------------------------------------------

    @Test
    fun `a screen still on the stack is kept`() {
        val stack = listOf(Route.Journey, Route.Board(station("8000001")))
        val stale = staleKeys(stack.mapTo(mutableSetOf()) { it.key }, stack)
        assertTrue(stale.isEmpty())
    }

    @Test
    fun `a screen that was popped is forgotten`() {
        val board = Route.Board(station("8000001"))
        val known = setOf(Route.Journey.key, board.key)
        assertEquals(setOf(board.key), staleKeys(known, listOf(Route.Journey)))
    }

    @Test
    fun `going back does not forget the screen being returned to`() {
        // The whole point: popping the board must leave the journey screen's
        // typed stations alone.
        val board = Route.Board(station("8000001"))
        val known = setOf(Route.Journey.key, board.key)
        assertTrue(Route.Journey.key !in staleKeys(known, listOf(Route.Journey)))
    }

    @Test
    fun `a long session does not accumulate every station ever opened`() {
        val visited = (1..50).map { Route.Board(station("800000$it")) }
        val known = (visited.map { it.key } + Route.Journey.key).toSet()
        val stale = staleKeys(known, listOf(Route.Journey, visited.last()))
        assertEquals(49, stale.size)
        assertTrue(visited.last().key !in stale)
    }
}
