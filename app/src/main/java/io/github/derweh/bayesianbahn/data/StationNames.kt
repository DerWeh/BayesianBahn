package io.github.derweh.bayesianbahn.data

/**
 * Compares station names across the three sources this app joins on them, which
 * spell the same station three different ways:
 *
 *     bundled stations.csv   "Türkheim (Bay) Bahnhof"
 *     IRIS route entries     "Türkheim(Bay)Bf"
 *     delay-history shards   "Türkheim (Bay)"
 *
 * `equals`/`contains` on those raw strings all return false, so a train that
 * genuinely called at the destination looked like it did not go there — the
 * journey search reported "No plannable trains found" for a route that runs
 * every hour.
 *
 * Two names refer to the same station when they agree after normalisation and
 * after dropping a trailing station designation. Dropping only the designation,
 * rather than matching on substrings, is what keeps "Memmingen" apart from
 * "Memmingen Ost" and "München Hbf" apart from "München-Pasing".
 */
object StationNames {

    /** Abbreviations DB uses interchangeably with their long forms. */
    private val ABBREVIATIONS = mapOf(
        "hbf" to "hauptbahnhof",
        "bf" to "bahnhof",
        "bhf" to "bahnhof",
        "pbf" to "personenbahnhof",
        "hp" to "haltepunkt",
        "hst" to "haltestelle",
    )

    /**
     * Words that say "this is a station" rather than *which* station. Note
     * "ostbahnhof" and the like are deliberately absent: they name a specific
     * station and must not be dropped.
     */
    private val DESIGNATIONS = setOf(
        "bahnhof", "hauptbahnhof", "personenbahnhof", "haltepunkt", "haltestelle",
    )

    /**
     * Lower-cased, umlaut-folded words. Punctuation separates words, so the
     * run-together IRIS style ("Türkheim(Bay)Bf") splits the same way as the
     * spaced style ("Türkheim (Bay) Bahnhof").
     */
    fun tokens(name: String): List<String> = name.lowercase()
        .replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
        .replace(Regex("[^a-z0-9]+"), " ")
        .split(' ')
        .filter { it.isNotEmpty() }
        .map { ABBREVIATIONS[it] ?: it }

    /** The name without its station designation: "Ulm Hbf" and "Ulm" both -> [ulm]. */
    fun core(name: String): List<String> = tokens(name).dropLastWhile { it in DESIGNATIONS }

    /** True when both names denote the same station. */
    fun matches(a: String, b: String): Boolean {
        val core = core(a)
        return core.isNotEmpty() && core == core(b)
    }
}
