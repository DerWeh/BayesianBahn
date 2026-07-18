package io.github.derweh.bayesianbahn.model

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin

/**
 * Location-scale Student-t distribution — the posterior predictive of the
 * Normal-inverse-gamma model. Self-contained implementation (Lanczos
 * log-gamma, continued-fraction incomplete beta) so the app needs no math
 * dependency.
 */
data class StudentT(val df: Double, val loc: Double, val scale: Double) {
    init {
        require(df > 0 && scale > 0) { "df and scale must be positive" }
    }

    /** P(X <= x). */
    fun cdf(x: Double): Double {
        val t = (x - loc) / scale
        val p = incompleteBeta(df / 2.0, 0.5, df / (df + t * t))
        return if (t > 0) 1.0 - 0.5 * p else 0.5 * p
    }

    /** Inverse CDF by bisection; accurate to ~1e-9 in standardized units. */
    fun quantile(p: Double): Double {
        require(p in 0.0..1.0) { "p must be in [0, 1]" }
        if (p == 0.0) return Double.NEGATIVE_INFINITY
        if (p == 1.0) return Double.POSITIVE_INFINITY
        var lo = -1e9
        var hi = 1e9
        repeat(200) {
            val mid = 0.5 * (lo + hi)
            if (cdf(mid) < p) lo = mid else hi = mid
            if (hi - lo < 1e-9 * scale) return 0.5 * (lo + hi)
        }
        return 0.5 * (lo + hi)
    }

    val mean: Double get() = if (df > 1) loc else Double.NaN

    companion object {
        /** Lanczos approximation of ln Γ(x), |error| < 1e-13 for x > 0. */
        fun logGamma(x: Double): Double {
            if (x < 0.5) {
                // Reflection formula keeps the approximation accurate near 0.
                return ln(PI / sin(PI * x)) - logGamma(1.0 - x)
            }
            val g = 7.0
            val coefficients = doubleArrayOf(
                0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                771.32342877765313, -176.61502916214059, 12.507343278686905,
                -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
            )
            val z = x - 1.0
            var sum = coefficients[0]
            for (i in 1 until coefficients.size) sum += coefficients[i] / (z + i)
            val t = z + g + 0.5
            return 0.5 * ln(2.0 * PI) + (z + 0.5) * ln(t) - t + ln(sum)
        }

        /** Regularized incomplete beta function I_x(a, b). */
        fun incompleteBeta(a: Double, b: Double, x: Double): Double {
            if (x <= 0.0) return 0.0
            if (x >= 1.0) return 1.0
            val logBeta = logGamma(a + b) - logGamma(a) - logGamma(b) +
                a * ln(x) + b * ln(1.0 - x)
            val front = exp(logBeta)
            return if (x < (a + 1.0) / (a + b + 2.0)) {
                front * betaContinuedFraction(a, b, x) / a
            } else {
                1.0 - front * betaContinuedFraction(b, a, 1.0 - x) / b
            }
        }

        /** Modified Lentz continued fraction for the incomplete beta. */
        private fun betaContinuedFraction(a: Double, b: Double, x: Double): Double {
            val tiny = 1e-300
            val eps = 1e-14
            var c = 1.0
            var d = 1.0 - (a + b) * x / (a + 1.0)
            if (abs(d) < tiny) d = tiny
            d = 1.0 / d
            var h = d
            for (m in 1..300) {
                val m2 = 2 * m
                var num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
                d = 1.0 + num * d
                if (abs(d) < tiny) d = tiny
                c = 1.0 + num / c
                if (abs(c) < tiny) c = tiny
                d = 1.0 / d
                h *= d * c
                num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
                d = 1.0 + num * d
                if (abs(d) < tiny) d = tiny
                c = 1.0 + num / c
                if (abs(c) < tiny) c = tiny
                d = 1.0 / d
                val delta = d * c
                h *= delta
                if (abs(delta - 1.0) < eps) break
            }
            return h
        }
    }
}
