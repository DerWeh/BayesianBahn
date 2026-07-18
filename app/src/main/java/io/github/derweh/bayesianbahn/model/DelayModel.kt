package io.github.derweh.bayesianbahn.model

import kotlinx.serialization.Serializable
import java.time.Instant
import java.time.ZoneId

/** Coarse product class — delay behaviour differs strongly between these. */
enum class TrainClass {
    LONG_DISTANCE, REGIONAL, SBAHN, OTHER;

    companion object {
        private val LONG_DISTANCE_CATEGORIES = setOf(
            "ICE", "IC", "EC", "ECE", "RJ", "RJX", "NJ", "EN", "FLX", "TGV", "D", "IR", "WB",
        )

        fun fromCategory(category: String): TrainClass = when {
            category.uppercase() in LONG_DISTANCE_CATEGORIES -> LONG_DISTANCE
            category.uppercase() == "S" -> SBAHN
            category.uppercase() in setOf("RE", "RB", "IRE", "MEX", "TER", "BRB", "AG") -> REGIONAL
            category.isNotBlank() && category.first().isLetter() -> REGIONAL
            else -> OTHER
        }
    }
}

/** Time-of-day band; congestion (and thus delay) is strongly time dependent. */
enum class TimeBand {
    MORNING_PEAK, MIDDAY, EVENING_PEAK, NIGHT;

    companion object {
        private val ZONE = ZoneId.of("Europe/Berlin")

        fun fromEpochMillis(epochMillis: Long): TimeBand {
            val hour = Instant.ofEpochMilli(epochMillis).atZone(ZONE).hour
            return when (hour) {
                in 6..8 -> MORNING_PEAK
                in 9..15 -> MIDDAY
                in 16..18 -> EVENING_PEAK
                else -> NIGHT
            }
        }
    }
}

/** Sufficient statistics of observed delays in one bucket (Welford form). */
@Serializable
data class BucketStats(val n: Long = 0, val mean: Double = 0.0, val m2: Double = 0.0) {
    fun update(x: Double): BucketStats {
        val n1 = n + 1
        val delta = x - mean
        val mean1 = mean + delta / n1
        return BucketStats(n1, mean1, m2 + delta * (x - mean1))
    }
}

/** Prediction for one train event, all values in minutes of delay. */
data class DelayPrediction(
    /** Expected delay. */
    val expected: Double,
    /** Lower end of the 80% credible interval. */
    val lower: Double,
    /** Upper end of the 80% credible interval. */
    val upper: Double,
    /** Probability the delay stays within 5 minutes. */
    val probWithin5Min: Double,
    /** Number of observations that informed this bucket. */
    val observations: Long,
    /** Whether a live delay report anchored this prediction. */
    val live: Boolean,
)

/**
 * Bayesian delay model: an independent Normal-inverse-gamma conjugate model
 * per (train class × time band) bucket.
 *
 * The NIG prior encodes domain knowledge (long-distance trains are later and
 * more variable than S-Bahn trains); every observed final delay updates the
 * bucket's sufficient statistics, and the posterior predictive is a
 * location-scale Student-t, so inference is exact and closed form — no
 * sampling needed on the phone.
 */
class DelayModel(private val stats: MutableMap<String, BucketStats> = mutableMapOf()) {

    /** NIG prior: mean delay [mu0], pseudo-observations [kappa0], shape/rate [alpha0]/[beta0]. */
    data class Prior(val mu0: Double, val kappa0: Double, val alpha0: Double, val beta0: Double)

    fun bucketKey(trainClass: TrainClass, band: TimeBand) = "${trainClass.name}:${band.name}"

    fun observations(trainClass: TrainClass, band: TimeBand): Long =
        stats[bucketKey(trainClass, band)]?.n ?: 0

    /** Records one completed event's final delay (minutes) into its bucket. */
    fun observe(trainClass: TrainClass, band: TimeBand, delayMinutes: Double) {
        // Clamp: IRIS occasionally reports pathological multi-hour rewrites
        // that would poison the variance estimate.
        val clamped = delayMinutes.coerceIn(-15.0, 240.0)
        val key = bucketKey(trainClass, band)
        stats[key] = (stats[key] ?: BucketStats()).update(clamped)
    }

    /** Snapshot of the sufficient statistics for persistence. */
    fun snapshot(): Map<String, BucketStats> = stats.toMap()

    /** Posterior predictive distribution of the final delay for a bucket. */
    fun posteriorPredictive(trainClass: TrainClass, band: TimeBand): StudentT {
        val prior = PRIORS.getValue(trainClass)
        val s = stats[bucketKey(trainClass, band)] ?: BucketStats()
        val n = s.n.toDouble()

        val kappaN = prior.kappa0 + n
        val muN = (prior.kappa0 * prior.mu0 + n * s.mean) / kappaN
        val alphaN = prior.alpha0 + n / 2.0
        val betaN = prior.beta0 + s.m2 / 2.0 +
            prior.kappa0 * n * (s.mean - prior.mu0) * (s.mean - prior.mu0) / (2.0 * kappaN)

        val predictiveScale = kotlin.math.sqrt(betaN * (kappaN + 1.0) / (alphaN * kappaN))
        return StudentT(df = 2.0 * alphaN, loc = muN, scale = predictiveScale)
    }

    /**
     * Predicts the final delay of one event.
     *
     * Without live data the bucket's posterior predictive is returned as is.
     * With a live delay report the distribution is re-anchored at the reported
     * delay: DB's own report is the best point estimate, and the residual
     * uncertainty (can it still grow?) is taken as a fixed fraction of the
     * bucket's predictive spread — a documented heuristic until the model
     * learns report-to-final residuals (see README roadmap).
     */
    /** The predictive distribution, re-anchored at a live report when present. */
    fun predictiveFor(
        trainClass: TrainClass,
        band: TimeBand,
        liveDelayMinutes: Double?,
    ): StudentT {
        val predictive = posteriorPredictive(trainClass, band)
        return if (liveDelayMinutes == null) {
            predictive
        } else {
            StudentT(
                df = predictive.df,
                loc = liveDelayMinutes,
                scale = (predictive.scale * LIVE_SHRINKAGE).coerceAtLeast(MIN_LIVE_SCALE),
            )
        }
    }

    fun predict(
        trainClass: TrainClass,
        band: TimeBand,
        liveDelayMinutes: Double?,
    ): DelayPrediction {
        val dist = predictiveFor(trainClass, band, liveDelayMinutes)
        return DelayPrediction(
            expected = dist.loc,
            lower = dist.quantile(0.10),
            upper = dist.quantile(0.90),
            probWithin5Min = dist.cdf(5.0),
            observations = stats[bucketKey(trainClass, band)]?.n ?: 0,
            live = liveDelayMinutes != null,
        )
    }

    companion object {
        /**
         * Priors in minutes, from published long-run DB punctuality figures:
         * long-distance trains average ~4 min delay with heavy tails, regional
         * ~1.5 min, S-Bahn ~1 min. kappa0/alpha0 make the prior worth a
         * handful of observations so real data quickly dominates.
         */
        val PRIORS: Map<TrainClass, Prior> = mapOf(
            TrainClass.LONG_DISTANCE to Prior(mu0 = 4.0, kappa0 = 4.0, alpha0 = 3.0, beta0 = 128.0),
            TrainClass.REGIONAL to Prior(mu0 = 1.5, kappa0 = 4.0, alpha0 = 3.0, beta0 = 32.0),
            TrainClass.SBAHN to Prior(mu0 = 1.0, kappa0 = 4.0, alpha0 = 3.0, beta0 = 18.0),
            TrainClass.OTHER to Prior(mu0 = 2.0, kappa0 = 4.0, alpha0 = 3.0, beta0 = 50.0),
        )

        /** Fraction of predictive spread kept once a live report anchors the location. */
        const val LIVE_SHRINKAGE = 0.4

        /** Minutes; live reports are never treated as more precise than this. */
        const val MIN_LIVE_SCALE = 1.2

        fun fromSnapshot(snapshot: Map<String, BucketStats>) =
            DelayModel(snapshot.toMutableMap())
    }
}
