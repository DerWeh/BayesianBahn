package io.github.derweh.bayesianbahn.model

/** Weighted empirical distribution over arbitrary (value, weight) points. */
class PointDistribution(points: List<Pair<Double, Double>>) : DelayDistribution {
    private val points = points.sortedBy { it.first }
    private val totalWeight = points.sumOf { it.second }

    override fun cdf(x: Double): Double {
        if (points.isEmpty()) return Double.NaN
        var acc = 0.0
        for ((value, weight) in points) {
            if (value > x) break
            acc += weight
        }
        return acc / totalWeight
    }

    override fun quantile(p: Double): Double {
        require(p in 0.0..1.0)
        if (points.isEmpty()) return Double.NaN
        val target = p * totalWeight
        var acc = 0.0
        for ((value, weight) in points) {
            acc += weight
            if (acc >= target) return value
        }
        return points.last().first
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
        val usable = candidates
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
