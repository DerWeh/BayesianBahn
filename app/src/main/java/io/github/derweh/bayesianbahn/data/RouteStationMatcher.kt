package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.IrisClient
import java.util.concurrent.ConcurrentHashMap

/**
 * Decides whether an entry in a train's IRIS route list (`ppth`) is a given
 * station.
 *
 * Route lists carry names, not EVA numbers, so this comparison is unavoidable —
 * but it does not have to be a guess. IRIS will name a station itself:
 *
 *     GET /iris-tts/timetable/station/8000144
 *     <station name="Türkheim(Bay)Bf" eva="8000144" ds100="MTHB"/>
 *
 * One lookup per destination, cached, turns the comparison into an identity
 * check against the exact spelling IRIS uses everywhere else in its own data.
 * [StationNames] stays as the fallback for when that lookup is unavailable —
 * offline, or a station IRIS does not know — where a canonicalised comparison
 * is still much better than nothing.
 */
class RouteStationMatcher(
    private val irisClient: IrisClient,
    private val stationRepository: StationRepository,
) {

    /** eva -> IRIS's name, or [UNRESOLVED] when the lookup did not answer. */
    private val names = ConcurrentHashMap<String, String>()

    /** IRIS route name -> eva, for the inverse lookup. */
    private val evas = ConcurrentHashMap<String, String>()

    /**
     * Predicate for "this route entry is [station]". Resolving costs at most
     * one request, made here, so the returned predicate is a plain function and
     * can be used inside `filter`/`any` on a whole board.
     */
    suspend fun matcherFor(station: Station): (String) -> Boolean {
        val irisName = irisName(station.eva)
        return { entry ->
            (irisName != null && entry.equals(irisName, ignoreCase = true)) ||
                StationNames.matches(entry, station.name)
        }
    }

    /**
     * The station an IRIS route entry names.
     *
     * The bundled list is tried first because it costs nothing, and IRIS is
     * asked only when that fails — which it does for names like
     * "Frankfurt(M) Flughafen Regionalbf", spelled "Frankfurt (Main) Flughafen
     * Regionalbahnhof" in the station list. No string rule bridges that pair
     * ("M"/"Main", "Regionalbf"/"Regionalbahnhof"), so offering such a station
     * as a transfer and then failing to find it was unavoidable without asking
     * IRIS which station it means.
     */
    suspend fun station(routeName: String): Station? {
        stationRepository.byName(routeName)?.let { return it }
        return evaFor(routeName)?.let { stationRepository.byEva(it) }
    }

    /**
     * The stations named by a train's route, for picking a transfer.
     *
     * This is the inverse lookup and cannot be done with one request the way
     * [matcherFor] can, so it stays off the common path: names are resolved
     * against the bundled list first, and IRIS is only asked when that leaves
     * nothing to work with. Roughly one station in five is spelled too
     * differently to match locally ("Ostkreuz" vs "Berlin Ostkreuz"), and for a
     * route made only of those the alternative is finding no transfer at all.
     */
    suspend fun stationsOn(path: List<String>): List<Station> {
        val local = path.mapNotNull { stationRepository.byName(it) }
        if (local.isNotEmpty()) return local
        return path.take(MAX_INVERSE_LOOKUPS).mapNotNull { name ->
            evaFor(name)?.let { stationRepository.byEva(it) }
        }
    }

    /** IRIS's EVA number for a name it uses, asked once and remembered. */
    private suspend fun evaFor(routeName: String): String? {
        val eva = evas.getOrPut(routeName) {
            runCatching { irisClient.stationEva(routeName) }.getOrNull() ?: UNRESOLVED
        }
        return eva.takeIf { it != UNRESOLVED }
    }

    private suspend fun irisName(eva: String): String? {
        names[eva]?.let { return it.takeIf { name -> name != UNRESOLVED } }
        // A failure here is not worth retrying inside one search; it is cached
        // as unresolved so an unreachable IRIS costs one request, not one per
        // station on every route.
        val resolved = runCatching { irisClient.stationName(eva) }.getOrNull()
        names[eva] = resolved ?: UNRESOLVED
        return resolved
    }

    private companion object {
        /** Marks a lookup that did not answer; a real name or eva is never empty. */
        const val UNRESOLVED = ""

        /** A cap, so one odd route cannot turn into a burst against a public API. */
        const val MAX_INVERSE_LOOKUPS = 8
    }
}
