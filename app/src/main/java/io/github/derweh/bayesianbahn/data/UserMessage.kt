package io.github.derweh.bayesianbahn.data

/**
 * A failure the user can act on, named rather than written out.
 *
 * OkHttp's own message was being handed straight to the screen:
 *
 *     failed to connect to iris.noncd.db.de/81.200.197.7 (port 443)
 *     from /172.18.227.123 (port 52250) after 15000ms
 *
 * That names a host the user never chose and two addresses they cannot do
 * anything about, and never says what went wrong or what to try.
 *
 * The wording lives in `strings.xml`: these are raised deep in the planners,
 * which have no Android context and no way to know the user's language, so the
 * screen that shows a message is what turns it into text.
 */
sealed interface UserMessage {

    /** The live timetable could not be reached and nothing offline could stand in. */
    data object TimetableUnreachable : UserMessage

    /**
     * Same, but the downloaded history could not fill the gap either — worth
     * saying, because the app does normally work without a connection.
     */
    data object TimetableUnreachableNoHistory : UserMessage

    data object SameOriginAndDestination : UserMessage

    /**
     * A delay-history update failed. [detail] is the underlying exception's
     * message where it has one — untranslated, but more specific than anything
     * this app could say in its place.
     */
    data class UpdateFailed(val detail: String?) : UserMessage

    data class StationNotFound(val query: String) : UserMessage

    data class TransferStationNotFound(val query: String) : UserMessage

    data class NoTimetableData(val station: String) : UserMessage

    /**
     * Shown when the origin has departures but none of them lead anywhere
     * useful. The search only covers journeys with at most one change, so
     * "nothing found" must not be reported as "no connection exists" — that is
     * exactly the reading the old wording invited.
     */
    data class NoConnection(val from: String, val to: String) : UserMessage

    data class FeederDoesNotReach(val train: String, val station: String) : UserMessage

    data class NoPlannedArrival(val station: String) : UserMessage

    data class NoTrainsTowards(
        val destination: String,
        val station: String,
        val deutschlandTicketOnly: Boolean,
    ) : UserMessage

    data class NotEnoughHistory(val destination: String) : UserMessage
}
