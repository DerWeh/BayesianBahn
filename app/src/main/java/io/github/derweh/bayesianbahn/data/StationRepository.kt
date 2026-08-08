package io.github.derweh.bayesianbahn.data

import android.content.Context
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

data class Station(
    val eva: String,
    val name: String,
    val weight: Int,
    val lat: Double? = null,
    val lon: Double? = null,
) {
    /** Great-circle distance in km, or null when either station lacks coordinates. */
    fun distanceKm(other: Station): Double? {
        val lat1 = lat ?: return null
        val lon1 = lon ?: return null
        val lat2 = other.lat ?: return null
        val lon2 = other.lon ?: return null
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2).pow(2) +
            cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2).pow(2)
        return 2 * EARTH_RADIUS_KM * asin(min(1.0, sqrt(a)))
    }

    private companion object {
        const val EARTH_RADIUS_KM = 6371.0
    }
}

/**
 * Offline station search over the bundled `stations.csv` asset
 * (derived from DB's CC BY 4.0 station dataset, sorted by importance).
 */
class StationRepository(private val context: Context) {

    private val stations: List<Station> by lazy {
        context.assets.open("stations.csv").bufferedReader().readLines().mapNotNull { line ->
            val parts = line.split(';')
            if (parts.size < 3) return@mapNotNull null
            Station(
                eva = parts[0],
                name = parts[1],
                weight = parts[2].toIntOrNull() ?: 0,
                lat = parts.getOrNull(3)?.toDoubleOrNull(),
                lon = parts.getOrNull(4)?.toDoubleOrNull(),
            )
        }
    }

    /**
     * Name lookup for stations named in an IRIS route.
     *
     * Uses [StationNames] rather than [normalize]: the latter treats
     * "Türkheim(Bay)Bf" and "Türkheim (Bay) Bahnhof" as different stations, so
     * IRIS route entries for anything with a designation resolved to null and
     * were dropped as transfer candidates without a trace.
     */
    fun byName(name: String): Station? {
        val core = StationNames.core(name)
        if (core.isEmpty()) return null
        return stations.firstOrNull { StationNames.core(it.name) == core }
    }

    fun byEva(eva: String): Station? {
        val e = eva.trimStart('0')
        return stations.firstOrNull { it.eva.trimStart('0') == e }
    }

    fun search(query: String, limit: Int = 30): List<Station> {
        val q = normalize(query)
        if (q.isBlank()) return stations.take(limit)
        return stations.asSequence()
            .filter { normalize(it.name).contains(q) }
            .sortedByDescending { (if (normalize(it.name).startsWith(q)) 1_000_000 else 0) + it.weight }
            .take(limit)
            .toList()
    }

    private fun normalize(s: String): String = s.lowercase()
        .replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
        .replace("-", " ").replace("(", " ").replace(")", " ")
        .replace(Regex("\\s+"), " ").trim()
}
