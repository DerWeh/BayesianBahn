package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DelayModelTest {

    @Test
    fun `prior predictive matches prior parameters`() {
        val model = DelayModel()
        val predictive = model.posteriorPredictive(TrainClass.LONG_DISTANCE, TimeBand.MIDDAY)
        val prior = DelayModel.PRIORS.getValue(TrainClass.LONG_DISTANCE)
        assertEquals(prior.mu0, predictive.loc, 1e-9)
        assertEquals(2 * prior.alpha0, predictive.df, 1e-9)
    }

    @Test
    fun `posterior mean moves towards data`() {
        val model = DelayModel()
        repeat(50) { model.observe(TrainClass.REGIONAL, TimeBand.MIDDAY, 10.0) }
        val predictive = model.posteriorPredictive(TrainClass.REGIONAL, TimeBand.MIDDAY)
        // 50 observations of 10 min against prior mean 1.5 with kappa0=4
        assertTrue("expected loc near 10, was ${predictive.loc}", predictive.loc in 9.0..10.0)
    }

    @Test
    fun `bucket stats use welford updates`() {
        var stats = BucketStats()
        val xs = listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        xs.forEach { stats = stats.update(it) }
        assertEquals(8, stats.n)
        assertEquals(5.0, stats.mean, 1e-9)
        assertEquals(32.0, stats.m2, 1e-9) // population variance 4 * n
    }

    @Test
    fun `live report anchors the prediction`() {
        val model = DelayModel()
        val prediction = model.predict(TrainClass.LONG_DISTANCE, TimeBand.NIGHT, liveDelayMinutes = 23.0)
        assertEquals(23.0, prediction.expected, 1e-9)
        assertTrue(prediction.live)
        assertTrue(prediction.upper > prediction.lower)
    }

    @Test
    fun `category classification`() {
        assertEquals(TrainClass.LONG_DISTANCE, TrainClass.fromCategory("ICE"))
        assertEquals(TrainClass.SBAHN, TrainClass.fromCategory("S"))
        assertEquals(TrainClass.REGIONAL, TrainClass.fromCategory("RE"))
        assertEquals(TrainClass.REGIONAL, TrainClass.fromCategory("BRB"))
    }
}
