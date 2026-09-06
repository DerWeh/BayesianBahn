package io.github.derweh.bayesianbahn.model

/** Weighted empirical distribution over arbitrary (value, weight) points. */
/**
 * A weighted set of arrival times, as a distribution.
 *
 * Both lookups run on a prefix-summed copy rather than by walking the points.
 * The points are one per historical run per candidate train, so a change with
 * six candidates carries several hundred to a few thousand of them, and every
 * scan was the whole list: drawing a histogram asks for a cdf per bucket and
 * the interval asks for three quantiles, which turned one screen into hundreds
 * of thousands of comparisons. The evaluation felt it worse still — scoring a
 * day of two-leg journeys spent 88% of its time here — because CRPS integrates
 * the cdf over 661 whole minutes for every journey it scores.
 *
 * Building the prefix sum costs one pass, which the constructor was already
 * paying for the sort.
 *
 * The total is the last prefix sum rather than a separate pass over the input.
 * That is not only cheaper: it is what makes `cdf` reach exactly 1 at the top
 * and `quantile(1.0)` return exactly the last point. Summing the same weights
 * in a different order gave a total that could sit an ulp away from the running
 * one, and a quantile landing on a flat stretch of the cdf then picked its
 * point on the strength of that last bit. Over a day of scored journeys it
 * moved two medians out of eight thousand — real, and arbitrary either way,
 * since a discrete distribution genuinely has no single median there.
 */
class PointDistribution(points: List<Pair<Double, Double>>) : DelayDistribution {
    private val values: DoubleArray
    private val cumulative: DoubleArray
    private val totalWeight: Double

    init {
        val sorted = points.sortedBy { it.first }
        values = DoubleArray(sorted.size)
        cumulative = DoubleArray(sorted.size)
        var acc = 0.0
        for (i in sorted.indices) {
            values[i] = sorted[i].first
            acc += sorted[i].second
            cumulative[i] = acc
        }
        totalWeight = acc
    }

    /** Index of the last value <= [x], or -1 when every value is above it. */
    private fun lastAtOrBelow(x: Double): Int {
        var lo = 0
        var hi = values.size - 1
        var found = -1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (values[mid] <= x) {
                found = mid
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
        return found
    }

    override fun cdf(x: Double): Double {
        if (values.isEmpty()) return Double.NaN
        val i = lastAtOrBelow(x)
        return if (i < 0) 0.0 else cumulative[i] / totalWeight
    }

    override fun quantile(p: Double): Double {
        require(p in 0.0..1.0)
        if (values.isEmpty()) return Double.NaN
        val target = p * totalWeight
        // The first index whose cumulative weight reaches the target, which is
        // what the linear scan returned: `acc >= target` on the way up.
        var lo = 0
        var hi = values.size - 1
        var found = values.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (cumulative[mid] >= target) {
                found = mid
                hi = mid - 1
            } else {
                lo = mid + 1
            }
        }
        return values[found]
    }
}

/**
 * Propagates an arrival distribution through a transfer via the law of total
 * probability: the passenger boards the first candidate train (in planned
 * departure order) that has not yet left when they reach the platform, so
 *
 *   P(final arrival) = Σ_k P(board k) · P(arrival | board k),
 *
 * where both the feeder's arrival and each candidate's departure/arrival are
 * empirical distributions. A delayed *earlier* candidate that is still at the
 * platform is boarded — the model captures that missing your train sometimes
 * helps. Within a candidate, departure and arrival delays come from the same
 * historical run, so their correlation is preserved; independence is only
 * assumed *between* the feeder and the candidates (large-scale disruptions
 * violate this — predictions are then optimistic).
 *
 * A candidate with a live departure delay is treated as departing exactly
 * then, and its arrival applies the delta model (live + historical last-leg
 * residuals), matching [EmpiricalDelay]'s live handling.
 */
object ConnectionModel {

    /** Number of equal-mass samples drawn from the feeder's arrival distribution. */
    const val FEEDER_SAMPLES = 80

    /**
     * Weight below which a candidate's arrival points are dropped.
     *
     * They still count towards its board probability, which is exact and cheap;
     * what is dropped is their contribution to the shape of the arrival
     * distribution, and a millionth of the mass cannot move a quantile that is
     * read at one part in a hundred.
     */
    const val MASS_FLOOR = 1e-6

    /** One historical run of a candidate: delays at transfer and destination. */
    data class JointRun(
        val departureDelay: Double,
        val arrivalDelay: Double,
        val weight: Double,
    )

    data class Candidate(
        val id: String,
        val label: String,
        val plannedDepartureMillis: Long,
        val plannedArrivalMillis: Long,
        /** Joint (departure, arrival) delay samples; may be empty when live data exists. */
        val runs: List<JointRun>,
        /** Live departure delay in minutes, if IRIS reported one. */
        val liveDepartureDelay: Double?,
        val cancelledLive: Boolean,
        /** Historical cancellation rate of this candidate at the transfer. */
        val cancelRate: Double,
    )

    data class CandidateResult(val candidate: Candidate, val boardProbability: Double)

    data class Result(
        /**
         * Final arrival, minutes relative to [referenceArrivalMillis],
         * conditional on boarding one of the candidates.
         */
        val distribution: DelayDistribution,
        /** Planned arrival of the first (not live-cancelled) candidate. */
        val referenceArrivalMillis: Long,
        val candidates: List<CandidateResult>,
        /** Probability of catching none of the given candidates. */
        val missProbability: Double,
    )

    /**
     * P(the change works), composing both trains' distributions.
     *
     * Extracted so that nothing computes this twice. [propagate] uses the same
     * arithmetic inline over its candidate walk, and `ForecastHarness` calls
     * this directly — which is the point: the number in the study and the
     * number on the screen come from one piece of code, so they cannot drift.
     *
     * The change works when the feeder arrives early enough that the connecting
     * train has not yet left:
     *
     *     feeder arrival delay - slack <= the connecting train's departure delay
     *
     * so for each equal-mass sample of the feeder's arrival, the answer is the
     * departure's survival at that point. Independence between the two is
     * assumed and costs almost nothing — measured, they correlate at 0.1.
     */
    fun catchProbability(
        feederArrival: DelayDistribution,
        departure: DelayDistribution,
        slackMinutes: Double,
        samples: Int = FEEDER_SAMPLES,
    ): Double {
        var total = 0.0
        for (i in 0 until samples) {
            val p = (i + 0.5) / samples
            total += departure.survival(feederArrival.quantile(p) - slackMinutes)
        }
        return total / samples
    }

    fun propagate(
        feederArrival: DelayDistribution,
        feederPlannedArrivalMillis: Long,
        transferMinutes: Int,
        candidates: List<Candidate>,
        nowMillis: Long = System.currentTimeMillis(),
    ): Result? {
        // A change is two trains, and this used to model one of them. The
        // feeder's arrival was a distribution; the connecting train's departure
        // was whatever DB last said, believed exactly — `if (live >= threshold)`
        // boards it and anything else loses it. That is the same point mass the
        // arrival anchor was criticised for, one train along, and correcting
        // only the arrival made changes *worse*: the two errors had been partly
        // cancelling.
        //
        // Both are marginalised now. Whether a change works is
        //
        //     feeder arrival error - connecting departure error <= margin
        //
        // which holds exactly on the scored days, so the answer is a difference
        // of two residuals rather than one of them and a guess. Held out after
        // the blockade ended: Brier 0.077 against 0.092 as shipped, and 0.160
        // against 0.196 on the changes tight enough to be in doubt.
        //
        // [LiveReport.informative] still decides which residual applies. A
        // train DB has said nothing about is not on time — those leave a median
        // minute and a mean three minutes late, and that slack is what the old
        // threshold threw away.
        val usable = candidates
            .map { it.copy(liveDepartureDelay = LiveReport.informative(it.liveDepartureDelay)) }
            .filter { it.cancelledLive || it.runs.isNotEmpty() || it.liveDepartureDelay != null }
            .sortedBy { it.plannedDepartureMillis }
        val reference = usable.firstOrNull { !it.cancelledLive } ?: return null

        // One distribution per candidate, built here rather than inside the
        // loop: the loop asks each of them [FEEDER_SAMPLES] times, and building
        // is the only arithmetic there is — a square root and two products.
        // Asking is a branch and one exp.
        //
        // Only where DB has reported. A candidate without one keeps the
        // history path below, which was never the bug: it already spreads the
        // departure over the train's own runs.
        val departure = arrayOfNulls<AnchoredDelay>(usable.size)
        for ((k, cand) in usable.withIndex()) {
            val live = cand.liveDepartureDelay
            if (cand.cancelledLive || live == null) continue
            departure[k] = AnchoredDelay(
                report = live,
                leadMinutes = leadMinutes(nowMillis, cand.plannedDepartureMillis),
                shape = ResidualShape.DEPARTURE_REPORTED,
            )
        }

        val points = ArrayList<Pair<Double, Double>>()
        val boardProbability = DoubleArray(usable.size)
        var missTotal = 0.0
        val sampleWeight = 1.0 / FEEDER_SAMPLES

        for (i in 0 until FEEDER_SAMPLES) {
            val p = (i + 0.5) / FEEDER_SAMPLES
            val feederDelay = feederArrival.quantile(p)
            // Time the passenger is ready to depart from the transfer platform.
            val ready = feederPlannedArrivalMillis + ((feederDelay + transferMinutes) * 60_000).toLong()

            var reachMass = sampleWeight // P(this sample ∧ no earlier candidate boarded)
            for ((k, cand) in usable.withIndex()) {
                if (reachMass <= 1e-12) break
                // Departure threshold in delay-minutes of this candidate.
                val threshold = (ready - cand.plannedDepartureMillis) / 60_000.0
                val arrivalBase = (cand.plannedArrivalMillis - reference.plannedArrivalMillis) / 60_000.0

                val pGone: Double
                if (cand.cancelledLive) {
                    pGone = 1.0
                } else if (departure[k] != null) {
                    // Still at the platform with probability P(departure delay
                    // >= threshold), not with certainty either way.
                    val live = cand.liveDepartureDelay ?: 0.0
                    val pBoard = (1 - cand.cancelRate) * departure[k]!!.survival(threshold)
                    boardProbability[k] += reachMass * pBoard
                    // Points, unlike the probability above, are only worth
                    // carrying when they weigh something. A departure that is
                    // now a distribution never returns exactly zero, so the
                    // walk no longer stops at the first certain train and every
                    // later candidate would otherwise contribute a run apiece
                    // at a millionth of the weight — several thousand points to
                    // sort for a change in the sixth decimal.
                    // `points.isEmpty()` keeps the floor from ever being the
                    // reason there is no distribution at all: the first thing
                    // that can be boarded is always carried, however unlikely,
                    // and only what comes after it is pruned.
                    //
                    // `pBoard > 0` guards that exception. `survival` underflows
                    // to exactly zero for a threshold far enough out, and
                    // without this the first such candidate contributes its
                    // whole run list at weight zero -- a PointDistribution of
                    // total weight zero, whose cdf is NaN rather than absent.
                    if (pBoard > 0 && (reachMass * pBoard > MASS_FLOOR || points.isEmpty())) {
                        // Arrival given it was boarded: the delta model as
                        // before, around the departure the passenger caught.
                        val runs = cand.runs
                        if (runs.isEmpty()) {
                            points += (arrivalBase + live) to reachMass * pBoard
                        } else {
                            val w = runs.sumOf { it.weight }
                            for (r in runs) {
                                points += (arrivalBase + live + (r.arrivalDelay - r.departureDelay)) to
                                    reachMass * pBoard * r.weight / w
                            }
                        }
                    }
                    pGone = 1.0 - pBoard
                } else {
                    val total = cand.runs.sumOf { it.weight }
                    val staying = cand.runs.filter { it.departureDelay >= threshold }
                    val stayWeight = staying.sumOf { it.weight }
                    val pBoard = (1 - cand.cancelRate) * stayWeight / total
                    if (pBoard > 0) {
                        for (r in staying) {
                            points += (arrivalBase + r.arrivalDelay) to
                                reachMass * pBoard * r.weight / stayWeight
                        }
                        boardProbability[k] += reachMass * pBoard
                    }
                    pGone = 1.0 - pBoard
                }
                reachMass *= pGone
            }
            missTotal += reachMass
        }

        if (points.isEmpty()) return null
        return Result(
            distribution = PointDistribution(points),
            referenceArrivalMillis = reference.plannedArrivalMillis,
            candidates = usable.mapIndexed { k, c -> CandidateResult(c, boardProbability[k]) },
            missProbability = missTotal,
        )
    }
}
