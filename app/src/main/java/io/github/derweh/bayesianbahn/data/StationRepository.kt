package io.github.derweh.bayesianbahn.data

import android.content.Context

data class Station(val eva: String, val name: String, val weight: Int)

/**
 * Offline station search over the bundled `stations.csv` asset
 * (derived from DB's CC BY 4.0 station dataset, sorted by importance).
 */
class StationRepository(private val context: Context) {

    private val stations: List<Station> by lazy {
        context.assets.open("stations.csv").bufferedReader().readLines().mapNotNull { line ->
            val parts = line.split(';')
            if (parts.size < 3) return@mapNotNull null
            Station(parts[0], parts[1], parts[2].toIntOrNull() ?: 0)
        }
    }

    /** Exact (normalized) name lookup, e.g. for stations named in an IRIS route. */
    fun byName(name: String): Station? {
        val n = normalize(name)
        return stations.firstOrNull { normalize(it.name) == n }
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
