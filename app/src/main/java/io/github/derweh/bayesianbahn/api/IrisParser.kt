package io.github.derweh.bayesianbahn.api

import org.xmlpull.v1.XmlPullParser
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Parses the XML documents served by the IRIS timetable API
 * (`iris.noncd.db.de/iris-tts/timetable`).
 *
 * The parser factory is injected so production code can use Android's built-in
 * pull parser while JVM unit tests use kxml2 directly.
 */
class IrisParser(private val newPullParser: () -> XmlPullParser) {

    /** Parses a `plan/{eva}/{date}/{hour}` document into full stops. */
    fun parsePlan(xml: String): List<TimetableStop> {
        val parser = newPullParser().apply { setInput(xml.reader()) }
        val stops = mutableListOf<TimetableStop>()

        var id: String? = null
        var category = ""
        var number = ""
        var arrival: EventInfo? = null
        var departure: EventInfo? = null
        var line: String? = null

        var event = parser.eventType
        while (event != XmlPullParser.END_DOCUMENT) {
            when (event) {
                XmlPullParser.START_TAG -> when (parser.name) {
                    "s" -> {
                        id = parser.getAttributeValue(null, "id")
                        category = ""
                        number = ""
                        line = null
                        arrival = null
                        departure = null
                    }
                    "tl" -> {
                        category = parser.getAttributeValue(null, "c") ?: ""
                        number = parser.getAttributeValue(null, "n") ?: ""
                    }
                    "ar" -> {
                        line = line ?: parser.getAttributeValue(null, "l")
                        arrival = parsePlannedEvent(parser)
                    }
                    "dp" -> {
                        line = line ?: parser.getAttributeValue(null, "l")
                        departure = parsePlannedEvent(parser)
                    }
                }
                XmlPullParser.END_TAG -> if (parser.name == "s" && id != null) {
                    stops += TimetableStop(id, TrainLabel(category, number, line), arrival, departure)
                    id = null
                }
            }
            event = parser.next()
        }
        return stops
    }

    /** Parses a `fchg/{eva}` document into partial changes keyed by stop id. */
    fun parseChanges(xml: String): Map<String, StopChange> {
        val parser = newPullParser().apply { setInput(xml.reader()) }
        val changes = mutableMapOf<String, StopChange>()

        var id: String? = null
        var arTime: Long? = null
        var arPlatform: String? = null
        var arCancelled = false
        var dpTime: Long? = null
        var dpPlatform: String? = null
        var dpCancelled = false

        var event = parser.eventType
        while (event != XmlPullParser.END_DOCUMENT) {
            when (event) {
                XmlPullParser.START_TAG -> when (parser.name) {
                    "s" -> {
                        id = parser.getAttributeValue(null, "id")
                        arTime = null; arPlatform = null; arCancelled = false
                        dpTime = null; dpPlatform = null; dpCancelled = false
                    }
                    "ar" -> {
                        arTime = parseTime(parser.getAttributeValue(null, "ct"))
                        arPlatform = parser.getAttributeValue(null, "cp")
                        arCancelled = parser.getAttributeValue(null, "cs") == "c"
                    }
                    "dp" -> {
                        dpTime = parseTime(parser.getAttributeValue(null, "ct"))
                        dpPlatform = parser.getAttributeValue(null, "cp")
                        dpCancelled = parser.getAttributeValue(null, "cs") == "c"
                    }
                }
                XmlPullParser.END_TAG -> if (parser.name == "s" && id != null) {
                    changes[id] = StopChange(
                        id, arTime, arPlatform, arCancelled, dpTime, dpPlatform, dpCancelled,
                    )
                    id = null
                }
            }
            event = parser.next()
        }
        return changes
    }

    private fun parsePlannedEvent(parser: XmlPullParser): EventInfo = EventInfo(
        plannedTime = parseTime(parser.getAttributeValue(null, "pt")),
        changedTime = null,
        plannedPlatform = parser.getAttributeValue(null, "pp"),
        changedPlatform = null,
        plannedPath = parser.getAttributeValue(null, "ppth")
            ?.split('|')?.filter { it.isNotBlank() } ?: emptyList(),
        cancelled = false,
    )

    companion object {
        private val ZONE = ZoneId.of("Europe/Berlin")
        private val FORMAT = DateTimeFormatter.ofPattern("yyMMddHHmm")

        /** IRIS times are `yyMMddHHmm` in German local time. */
        fun parseTime(raw: String?): Long? {
            if (raw.isNullOrBlank()) return null
            return runCatching {
                LocalDateTime.parse(raw, FORMAT).atZone(ZONE).toInstant().toEpochMilli()
            }.getOrNull()
        }

        /** Applies real-time [changes] onto planned [stops]. */
        fun merge(stops: List<TimetableStop>, changes: Map<String, StopChange>): List<TimetableStop> =
            stops.map { stop ->
                val change = changes[stop.id] ?: return@map stop
                stop.copy(
                    arrival = stop.arrival?.copy(
                        changedTime = change.arrivalChangedTime,
                        changedPlatform = change.arrivalChangedPlatform,
                        cancelled = change.arrivalCancelled,
                    ),
                    departure = stop.departure?.copy(
                        changedTime = change.departureChangedTime,
                        changedPlatform = change.departureChangedPlatform,
                        cancelled = change.departureCancelled,
                    ),
                )
            }
    }
}
