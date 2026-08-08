package io.github.derweh.bayesianbahn.data

/**
 * Text for failures the user can act on.
 *
 * OkHttp's own message was being handed straight to the screen:
 *
 *     failed to connect to iris.noncd.db.de/81.200.197.7 (port 443)
 *     from /172.18.227.123 (port 52250) after 15000ms
 *
 * That names a host the user never chose and two addresses they cannot do
 * anything about, and never says what went wrong or what to try.
 */
object UserMessages {

    /** The live timetable could not be reached and nothing offline could stand in. */
    const val TIMETABLE_UNREACHABLE =
        "Could not reach DB's live timetable. Check your connection and try again."

    /**
     * Same, but the downloaded history could not fill the gap either — worth
     * saying, because the app does normally work without a connection.
     */
    const val TIMETABLE_UNREACHABLE_NO_HISTORY =
        "Could not reach DB's live timetable, and no downloaded timetable covers " +
            "this station and time. Check your connection and try again."

    /**
     * The search only covers journeys with at most one change, so "nothing
     * found" must not be reported as "no connection exists" — that is exactly
     * the reading the old wording invited.
     */
    const val ONE_CHANGE_ONLY =
        "Only direct journeys and journeys with one change are searched so far, " +
            "so DB's own apps may still find a route with more changes."

    /** Shown when the origin has departures but none of them lead anywhere useful. */
    fun noConnection(from: String, to: String) =
        "No direct or one-change connection from $from to $to found around that " +
            "time. $ONE_CHANGE_ONLY"
}
