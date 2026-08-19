package io.github.derweh.bayesianbahn.ui

import kotlin.math.roundToInt

/**
 * A delay in minutes, written the way a platform display writes it.
 *
 * Trains do run early, and their delay is negative. Prefixing a hard-coded "+"
 * put "+-1" on the arrivals board; the sign belongs to the number, not to the
 * template around it.
 */
fun signedMinutes(delayMinutes: Double): String {
    val minutes = delayMinutes.roundToInt()
    return if (minutes > 0) "+$minutes" else minutes.toString()
}

/**
 * The delay chip's text, or `null` when there is nothing worth showing.
 *
 * Rounding happens before the decision, so a delay that rounds to zero reads as
 * on time rather than as "+0" — truncating instead would have shown "+0" for
 * anything under a minute late.
 */
fun delayChip(delayMinutes: Double?): String? {
    if (delayMinutes == null) return null
    return if (delayMinutes.roundToInt() == 0) null else signedMinutes(delayMinutes)
}
