package io.github.derweh.bayesianbahn.model

import org.junit.Assert.assertEquals
import org.junit.Test

class StudentTTest {

    @Test
    fun `standard t with 1 dof is cauchy`() {
        val t = StudentT(df = 1.0, loc = 0.0, scale = 1.0)
        // Cauchy CDF: 0.5 + atan(x)/pi
        assertEquals(0.5, t.cdf(0.0), 1e-9)
        assertEquals(0.75, t.cdf(1.0), 1e-9)
        assertEquals(0.25, t.cdf(-1.0), 1e-9)
    }

    @Test
    fun `large dof approaches normal`() {
        val t = StudentT(df = 1e6, loc = 0.0, scale = 1.0)
        assertEquals(0.5, t.cdf(0.0), 1e-6)
        assertEquals(0.8413447, t.cdf(1.0), 1e-4)
        assertEquals(0.9772499, t.cdf(2.0), 1e-4)
    }

    @Test
    fun `known t quantiles`() {
        // t distribution with 5 dof: 95th percentile = 2.015048
        val t = StudentT(df = 5.0, loc = 0.0, scale = 1.0)
        assertEquals(2.015048, t.quantile(0.95), 1e-4)
        assertEquals(-2.015048, t.quantile(0.05), 1e-4)
        assertEquals(0.0, t.quantile(0.5), 1e-6)
    }

    @Test
    fun `location and scale shift the distribution`() {
        val t = StudentT(df = 8.0, loc = 3.0, scale = 2.0)
        assertEquals(0.5, t.cdf(3.0), 1e-9)
        val base = StudentT(df = 8.0, loc = 0.0, scale = 1.0)
        assertEquals(base.quantile(0.9) * 2.0 + 3.0, t.quantile(0.9), 1e-6)
    }

    @Test
    fun `quantile inverts cdf`() {
        val t = StudentT(df = 6.0, loc = 4.0, scale = 7.0)
        for (p in listOf(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, t.cdf(t.quantile(p)), 1e-7)
        }
    }
}
