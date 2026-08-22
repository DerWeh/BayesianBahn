package io.github.derweh.bayesianbahn.model

/**
 * When DB's live number counts as evidence.
 *
 * DB states a stop in four shapes and three of them mean "on time": the
 * predicted time moved, it was confirmed unchanged, the stop is listed without
 * a time, or it is absent from the update entirely. Only the first is an
 * observation; the rest are the timetable, restated. Scored against the archive
 * over 2026-08-17..19, DB called a train on time for 61% of stops ten minutes
 * before departure and 99% of stops three hours out — and 31% of that last
 * group arrived more than two minutes late.
 *
 * Anchoring a forecast on that number cost 0.53 min of CRPS on trains that have
 * history, and left the stated 80% interval covering 55% of arrivals instead of
 * 80%. A report of "early" is no better: those trains averaged 1.4 minutes
 * *late*, so the threshold sits above zero rather than at it.
 *
 * This lives in `model` rather than next to its first caller because it is a
 * modelling rule, and because the models enforce it themselves — a caller that
 * forgets to apply it would otherwise reintroduce the bug silently.
 */
object LiveReport {

    /** Minutes. Below this, a live report is not treated as evidence. */
    const val MIN_INFORMATIVE_DELAY_MINUTES = 1.0

    /**
     * The report if it is evidence, null if it is the plan restated.
     *
     * Returning null is what makes the rest of the model ignore it: every live
     * path already handles "no live data", because that is the normal case for
     * a journey planned more than a day ahead.
     */
    fun informative(delayMinutes: Double?): Double? =
        delayMinutes?.takeIf { it >= MIN_INFORMATIVE_DELAY_MINUTES }
}
