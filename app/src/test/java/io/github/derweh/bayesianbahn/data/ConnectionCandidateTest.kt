package io.github.derweh.bayesianbahn.data

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Which connecting trains the app offers for a change.
 *
 * A train leaving shortly before the feeder is due is worth showing — usually
 * missed, but a delayed one is sometimes exactly the connection that works.
 * Taking the first six by departure time let those crowd out every train the
 * passenger could actually catch: where a station has a service every few
 * minutes, all six had gone before the feeder arrived, so the app offered six
 * impossible trains and no possible one. The evaluation measured that at 31%
 * of journeys with a change on one collected day, and it is invisible from
 * inside the app, which reports the six honestly as unreachable.
 */
class ConnectionCandidateTest {
    private fun min(x: Long) = x * 60_000L

    private fun pick(departures: List<Long>, feederPlanned: Long) =
        ConnectionPlanner.pickCandidates(departures.sorted(), feederPlanned) { it }

    @Test
    fun `a dense service does not fill the list with trains already gone`() {
        val feeder = min(600)
        val gone = (1..6).map { feeder - min(it * 4L) }.sorted()
        val ahead = (0..3).map { feeder + min(5) + min(it * 10L) }
        assertEquals(gone.takeLast(2) + ahead, pick(gone + ahead, feeder))
    }

    @Test
    fun `a just-missed train is still offered`() {
        val feeder = min(600)
        val list = listOf(feeder - min(5), feeder + min(10))
        assertEquals(list, pick(list, feeder))
    }

    @Test
    fun `a train gone longer than the look-back is not offered`() {
        val feeder = min(600)
        val stale = feeder - min(ConnectionPlanner.LOOK_BACK_MINUTES + 1L)
        val ahead = feeder + min(10)
        assertEquals(listOf(ahead), pick(listOf(stale, ahead), feeder))
    }

    @Test
    fun `the last trains of the day still fill the list`() {
        val feeder = min(600)
        val gone = (0..5).map { feeder - min(20) + min(it.toLong()) }
        val last = feeder + min(40)
        val got = pick(gone + listOf(last), feeder)
        assertEquals(ConnectionPlanner.MAX_CANDIDATES, got.size)
        assertEquals(last, got.last())
    }

    @Test
    fun `nothing ahead and nothing behind is an empty list`() {
        assertEquals(emptyList<Long>(), pick(emptyList(), min(600)))
    }
}
