package io.github.derweh.bayesianbahn.model

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * DB's report as a noisy observation instead of a fact.
 *
 * The app used to take a live delay report and treat it as exact, attaching a
 * fixed fraction of the historical spread to it. The report's own error spans
 * eleven minutes and the interval issued was two, so 42% of live-anchored
 * arrivals landed above their own 90th percentile. What replaces it is the
 * measured report-to-final residual: a width and a centre that grow with how
 * far ahead the question is asked, and a standardised shape.
 *
 * The shape is an asymmetric Laplace because delays are: a train DB has
 * already called late rarely makes the time back and often loses more, so the
 * right tail is the long one. Two rates rather than one, and both quantile and
 * cdf are closed forms — no sampling, no inversion, no table. That is what
 * makes it affordable in [ConnectionModel.propagate], which asks for a
 * probability once per candidate per feeder sample.
 *
 * Fitted by `tools/sensitivity_live.py` and frozen into `tools/anchor-model.json`;
 * `tools/tests/test_anchor_model.py` fails if these numbers drift from it.
 */
data class ResidualShape(
    /** Minutes of 10th-to-90th width at zero lead. */
    val widthIntercept: Double,
    /** Extra width per minute of lead. */
    val widthPerMinute: Double,
    /** Where the middle of the residual sits at zero lead. */
    val centreIntercept: Double,
    /** How the centre moves with the square root of the lead. */
    val centrePerSqrtMinute: Double,
    /** Rate of the left (recovering) tail, in units of the width. */
    val left: Double,
    /** Rate of the right (losing more) tail. Larger: delays grow more than they shrink. */
    val right: Double,
) {
    fun width(leadMinutes: Double): Double =
        widthIntercept + widthPerMinute * max(leadMinutes, 0.0)

    fun centre(leadMinutes: Double): Double =
        centreIntercept + centrePerSqrtMinute * sqrt(max(leadMinutes, 0.0))

    companion object {
        /** The feeder's own arrival: what a live report is worth about arriving. */
        val ARRIVAL = ResidualShape(7.470888, 0.244365, -0.76809, 0.344203, 0.155641, 0.534142)

        /** A connecting train DB has said something informative about. */
        val DEPARTURE_REPORTED = ResidualShape(22.525412, 0.08932, 2.832335, 0.199934, 0.290929, 0.49434)

        /**
         * A connecting train DB has said nothing about — four in five of them.
         * Silence is not "on time": these leave a median one to two minutes
         * late, and the tenth percentile is exactly zero because trains do not
         * leave early. That offset is the slack the old hard threshold threw
         * away.
         *
         * Both lead terms fit to zero. Not because the width is constant — by
         * lead band it runs 3, 9, 16, 8, 9 minutes, which is noisy and not
         * monotone — but because there is no trend for a linear term to find.
         * A constant is a summary here, not a discovered law, and the band
         * carrying two thirds of the sample is the one it sits nearest.
         */
        val DEPARTURE_SILENT = ResidualShape(15.0, 0.0, 1.0, 0.0, 0.034148, 0.733683)
    }
}

/**
 * A live report plus its measured error: an asymmetric Laplace about
 * `report + centre(lead)`, scaled by `width(lead)`.
 *
 * Construction does the only arithmetic there is — two multiplications and a
 * square root — so [quantile] and [cdf] are a branch, a logarithm or an
 * exponential, and an add.
 */
class AnchoredDelay(
    report: Double,
    leadMinutes: Double,
    shape: ResidualShape,
) : DelayDistribution {

    private val location = report + shape.centre(leadMinutes)
    private val scaleLeft: Double
    private val scaleRight: Double

    init {
        val width = max(shape.width(leadMinutes), MIN_WIDTH)
        scaleLeft = max(width * shape.left, MIN_SCALE)
        scaleRight = max(width * shape.right, MIN_SCALE)
    }

    override fun quantile(p: Double): Double {
        require(p in 0.0..1.0) { "p out of range: $p" }
        val clamped = p.coerceIn(TAIL, 1.0 - TAIL)
        return if (clamped < 0.5) {
            location + scaleLeft * ln(2.0 * clamped)
        } else {
            location - scaleRight * ln(2.0 * (1.0 - clamped))
        }
    }

    override fun cdf(x: Double): Double =
        if (x < location) {
            0.5 * exp((x - location) / scaleLeft)
        } else {
            1.0 - 0.5 * exp(-(x - location) / scaleRight)
        }

    override fun survival(x: Double): Double =
        if (x <= location) {
            1.0 - 0.5 * exp((x - location) / scaleLeft)
        } else {
            0.5 * exp(-(x - location) / scaleRight)
        }

    /** The centre, for the screens that want one number. */
    val median: Double get() = location

    override fun toString(): String =
        "AnchoredDelay(loc=%.2f, left=%.2f, right=%.2f)".format(
            location, scaleLeft, scaleRight,
        )

    private companion object {
        /**
         * Quantiles are unbounded at 0 and 1, so they are asked one part in a
         * million from the ends. At the widest lead this is about a hundred
         * minutes out, which is far past anything a screen shows.
         */
        const val TAIL = 1e-6

        /** A width can never be zero: [DEPARTURE_SILENT] has no lead term. */
        const val MIN_WIDTH = 0.5
        const val MIN_SCALE = 0.05
    }
}

/** Minutes from [nowMillis] until [plannedMillis], never negative. */
fun leadMinutes(nowMillis: Long, plannedMillis: Long): Double =
    max(0.0, (plannedMillis - nowMillis) / 60_000.0)

/** Guards against a caller passing milliseconds where minutes were wanted. */
internal fun sane(leadMinutes: Double): Boolean = abs(leadMinutes) < 60 * 24 * 7
