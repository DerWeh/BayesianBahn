package io.github.derweh.bayesianbahn.api

/** Identity of a train as shown to passengers, e.g. ICE 512 or line RB86. */
data class TrainLabel(
    /** Category such as "ICE", "IC", "RE", "S". */
    val category: String,
    /** Train number, e.g. "512". */
    val number: String,
    /** Passenger-facing line, e.g. "RB86" or "3" for an S-Bahn line. */
    val line: String?,
) {
    val display: String
        get() = line?.let { if (it.startsWith(category)) it else "$category $it" }
            ?: "$category $number"
}

/** One arrival or departure event of a stop, planned data plus real-time changes. */
data class EventInfo(
    /** Planned time, epoch millis, null for added stops without plan data. */
    val plannedTime: Long?,
    /** Real-time changed time, epoch millis, null if no change reported. */
    val changedTime: Long?,
    val plannedPlatform: String?,
    val changedPlatform: String?,
    /** Route before (arrival) or after (departure) this station. */
    val plannedPath: List<String>,
    val cancelled: Boolean,
) {
    val bestTime: Long? get() = changedTime ?: plannedTime
    val platform: String? get() = changedPlatform ?: plannedPlatform

    /** Reported delay in minutes, null if either time is missing. */
    val liveDelayMinutes: Double?
        get() = if (changedTime != null && plannedTime != null) {
            (changedTime - plannedTime) / 60_000.0
        } else null
}

/** A train calling at the requested station. */
/**
 * Something IRIS says about a stop besides its times.
 *
 * A blocked section is not a cancellation: the trains keep their times and DB
 * reports the trip impossible through a notice like this one instead. Reading
 * only the times and the cancellation flag made such an evening look ordinary.
 *
 * [category] is set on HIM notices — "Störung", "Bauarbeiten", "Information" —
 * and [code] on the numeric delay and cancellation causes. [priority] is
 * DB's own ranking, 1 being the most urgent.
 */
data class StopMessage(
    val kind: Kind,
    val category: String? = null,
    val code: String? = null,
    val priority: Int? = null,
) {
    enum class Kind { HIM, CANCEL_CAUSE, DELAY_CAUSE, QUALITY, FREE_TEXT, OTHER }

    /** Whether this is DB reporting the route in trouble, not merely late. */
    val isDisruption: Boolean
        get() = kind == Kind.HIM && category == DISRUPTION

    companion object {
        /** DB's own word for it, as it appears in the `cat` attribute. */
        const val DISRUPTION = "Störung"

        fun kindOf(raw: String?): Kind = when (raw) {
            "h" -> Kind.HIM
            "c" -> Kind.CANCEL_CAUSE
            "d" -> Kind.DELAY_CAUSE
            "q" -> Kind.QUALITY
            "f" -> Kind.FREE_TEXT
            else -> Kind.OTHER
        }
    }
}

data class TimetableStop(
    val id: String,
    val label: TrainLabel,
    val arrival: EventInfo?,
    val departure: EventInfo?,
    val messages: List<StopMessage> = emptyList(),
) {
    /** DB is reporting trouble on this train's route, times notwithstanding. */
    val disrupted: Boolean
        get() = messages.any { it.isDisruption }

    val destination: String?
        get() = departure?.plannedPath?.lastOrNull()
    val origin: String?
        get() = arrival?.plannedPath?.firstOrNull()
}

/** Partial real-time update for a stop, joined onto the plan by [id]. */
data class StopChange(
    val id: String,
    val arrivalChangedTime: Long?,
    val arrivalChangedPlatform: String?,
    val arrivalCancelled: Boolean,
    val departureChangedTime: Long?,
    val departureChangedPlatform: String?,
    val departureCancelled: Boolean,
    val messages: List<StopMessage> = emptyList(),
)

/** One entry of a `station/{query}` document: IRIS's own name for an EVA number. */
data class IrisStation(val name: String, val eva: String)
