package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConnectionModelTest {

    private val minute = 60_000L
    private val t0 = 1_700_000_000_000L // feeder planned arrival at the transfer

    /** Feeder that always arrives with exactly [delay] minutes. */
    private fun feeder(delay: Double) = PointDistribution(listOf(delay to 1.0))

    private fun candidate(
        id: String,
        depAfterFeeder: Long,
        rideMinutes: Long = 30,
        runs: List<ConnectionModel.JointRun> = List(20) { ConnectionModel.JointRun(0.0, 0.0, 1.0) },
        liveDep: Double? = null,
        cancelledLive: Boolean = false,
        cancelRate: Double = 0.0,
    ) = ConnectionModel.Candidate(
        id = id,
        label = id,
        plannedDepartureMillis = t0 + depAfterFeeder * minute,
        plannedArrivalMillis = t0 + (depAfterFeeder + rideMinutes) * minute,
        runs = runs,
        liveDepartureDelay = liveDep,
        cancelledLive = cancelledLive,
        cancelRate = cancelRate,
    )

    @Test
    fun `punctual feeder catches the first train`() {
        val result = ConnectionModel.propagate(
            feederArrival = feeder(0.0),
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(candidate("A", depAfterFeeder = 10), candidate("B", depAfterFeeder = 40)),
        )!!
        assertEquals(1.0, result.candidates[0].boardProbability, 1e-9)
        assertEquals(0.0, result.candidates[1].boardProbability, 1e-9)
        assertEquals(0.0, result.missProbability, 1e-9)
        // Arrives exactly when train A is planned to arrive.
        assertEquals(0.0, result.distribution.quantile(0.5), 1e-9)
    }

    @Test
    fun `late feeder falls through to the next train`() {
        // Feeder 30 late, A departs +10 punctually -> gone; B at +40 is caught.
        val result = ConnectionModel.propagate(
            feederArrival = feeder(30.0),
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(candidate("A", depAfterFeeder = 10), candidate("B", depAfterFeeder = 40)),
        )!!
        assertEquals(0.0, result.candidates[0].boardProbability, 1e-9)
        assertEquals(1.0, result.candidates[1].boardProbability, 1e-9)
        // Reference is A's planned arrival (+40); B arrives 30 min later.
        assertEquals(30.0, result.distribution.quantile(0.5), 1e-9)
    }

    @Test
    fun `a delayed earlier train can still be caught`() {
        // A is late half the time by 25 min: a feeder 20 late catches A in
        // exactly those runs and falls through to B otherwise.
        val runsA = List(10) { ConnectionModel.JointRun(25.0, 25.0, 1.0) } +
            List(10) { ConnectionModel.JointRun(0.0, 0.0, 1.0) }
        val result = ConnectionModel.propagate(
            feederArrival = feeder(20.0),
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(
                candidate("A", depAfterFeeder = 10, runs = runsA),
                candidate("B", depAfterFeeder = 60),
            ),
        )!!
        assertEquals(0.5, result.candidates[0].boardProbability, 1e-9)
        assertEquals(0.5, result.candidates[1].boardProbability, 1e-9)
    }

    @Test
    fun `live cancelled candidate is skipped`() {
        val result = ConnectionModel.propagate(
            feederArrival = feeder(0.0),
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(
                candidate("A", depAfterFeeder = 10, cancelledLive = true),
                candidate("B", depAfterFeeder = 40),
            ),
        )!!
        assertEquals(0.0, result.candidates[0].boardProbability, 1e-9)
        assertEquals(1.0, result.candidates[1].boardProbability, 1e-9)
        // Reference skips the cancelled train: B's own planned arrival.
        assertEquals(0.0, result.distribution.quantile(0.5), 1e-9)
    }

    @Test
    fun `live departure delay keeps a train catchable and shifts its arrival`() {
        // A reported +20: a feeder 15 late still catches it; arrival uses the
        // delta model on A's historical dep->arr residuals (+2 on the leg).
        val runsA = List(20) { ConnectionModel.JointRun(5.0, 7.0, 1.0) }
        val result = ConnectionModel.propagate(
            feederArrival = feeder(15.0),
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(candidate("A", depAfterFeeder = 10, runs = runsA, liveDep = 20.0)),
        )!!
        assertEquals(1.0, result.candidates[0].boardProbability, 1e-9)
        assertEquals(22.0, result.distribution.quantile(0.5), 1e-9)
    }

    @Test
    fun `missing every candidate is reported`() {
        val result = ConnectionModel.propagate(
            feederArrival = feeder(60.0),
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(candidate("A", depAfterFeeder = 10)),
        )
        // All mass misses -> no distribution to show.
        assertTrue(result == null)
    }

    @Test
    fun `uncertain feeder mixes both outcomes`() {
        // Feeder on time (75%) or 30 late (25%).
        val feeder = PointDistribution(listOf(0.0 to 0.75, 30.0 to 0.25))
        val result = ConnectionModel.propagate(
            feederArrival = feeder,
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(candidate("A", depAfterFeeder = 10), candidate("B", depAfterFeeder = 40)),
        )!!
        assertEquals(0.75, result.candidates[0].boardProbability, 0.02)
        assertEquals(0.25, result.candidates[1].boardProbability, 0.02)
        assertEquals(0.0, result.distribution.quantile(0.5), 1e-9)
        assertTrue(result.distribution.quantile(0.9) >= 29.0)
    }

    @Test
    fun `historical cancellation rate leaks probability to the next train`() {
        val result = ConnectionModel.propagate(
            feederArrival = feeder(0.0),
            feederPlannedArrivalMillis = t0,
            transferMinutes = 5,
            candidates = listOf(
                candidate("A", depAfterFeeder = 10, cancelRate = 0.2),
                candidate("B", depAfterFeeder = 40),
            ),
        )!!
        assertEquals(0.8, result.candidates[0].boardProbability, 1e-9)
        assertEquals(0.2, result.candidates[1].boardProbability, 1e-9)
    }
}
