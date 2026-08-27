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

    fun propagate(
        feederArrival: DelayDistribution,
        feederPlannedArrivalMillis: Long,
        transferMinutes: Int,
        candidates: List<Candidate>,
    ): Result? {
        // A live departure report is taken as fact below — if the train is
        // reported later than the passenger can arrive it is missed, otherwise
        // it is caught, with no distribution in between. That is only defensible
        // for a report that is an observation, and DB reports "on time" for
        // almost every train until shortly before departure (see [LiveReport]).
        // Believing those turned a train with a history of leaving late into a
        // certainty. The gate is applied here rather than in the callers so no
        // caller can omit it, and before the filter below so that a candidate
        // left with neither history nor a report is dropped rather than reaching
        // the weighting with an empty run list.
        val usable = candidates
            .map { it.copy(liveDepartureDelay = LiveReport.informative(it.liveDepartureDelay)) }
            .filter { it.cancelledLive || it.runs.isNotEmpty() || it.liveDepartureDelay != null }
            .sortedBy { it.plannedDepartureMillis }
        val reference = usable.firstOrNull { !it.cancelledLive } ?: return null

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
                } else if (cand.liveDepartureDelay != null) {
                    val live = cand.liveDepartureDelay
                    if (live >= threshold) {
                        // Known to still be there: board it; arrival = delta model.
                        val runs = cand.runs
                        if (runs.isEmpty()) {
                            points += (arrivalBase + live) to reachMass
                        } else {
                            val w = runs.sumOf { it.weight }
                            for (r in runs) {
                                points += (arrivalBase + live + (r.arrivalDelay - r.departureDelay)) to
                                    reachMass * r.weight / w
                            }
                        }
                        boardProbability[k] += reachMass
                        pGone = 0.0
                    } else {
                        pGone = 1.0
                    }
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
