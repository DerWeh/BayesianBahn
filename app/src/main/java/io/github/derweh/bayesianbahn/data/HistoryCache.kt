package io.github.derweh.bayesianbahn.data

import kotlinx.coroutines.CompletableDeferred

/**
 * Memo for train-history lookups, in front of the disk cache.
 *
 * A single journey search asks for the same train several times over — once
 * while building each board it opens, again when the itinerary is scored, and
 * again for every board that train also appears on. Each of those went to disk,
 * gunzipped the shard and re-parsed its JSON, and on a miss went to the network.
 *
 * Two things happen here. A completed lookup is remembered, so the repeats are
 * free. And a lookup already *in flight* is joined rather than started again,
 * which matters now that boards load their shards concurrently: without it, six
 * parallel stops naming the same train would open six identical requests.
 *
 * The map is bounded. An unbounded one would be fine for a single search and
 * would grow all day across many, holding every train the user ever looked at.
 */
class HistoryCache(private val capacity: Int = CAPACITY) {

    private val lock = Any()

    /** Access-ordered [LinkedHashMap] is the JDK's LRU. */
    private val done = object : LinkedHashMap<String, TrainHistory?>(16, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, TrainHistory?>) =
            size > capacity
    }
    private val inFlight = mutableMapOf<String, CompletableDeferred<TrainHistory?>>()

    /** Lookups that actually reached [get]'s loader, so a test can see the saving. */
    var loads = 0
        private set

    /**
     * The history for [key], loading it at most once even if asked for
     * concurrently. `null` is a real answer — a train with no shard anywhere —
     * and is remembered as such, so a history-less train is not re-fetched on
     * every stop it appears at.
     */
    suspend fun get(key: String, loader: suspend () -> TrainHistory?): TrainHistory? {
        var join: CompletableDeferred<TrainHistory?>? = null
        var own: CompletableDeferred<TrainHistory?>? = null
        synchronized(lock) {
            if (done.containsKey(key)) return done[key]
            val pending = inFlight[key]
            if (pending != null) {
                join = pending
            } else {
                own = CompletableDeferred<TrainHistory?>().also { inFlight[key] = it }
                loads++
            }
        }
        join?.let { return it.await() }

        val mine = own!!
        try {
            val value = loader()
            synchronized(lock) {
                done[key] = value
                inFlight.remove(key)
            }
            mine.complete(value)
            return value
        } catch (t: Throwable) {
            // Nothing is remembered on failure — the next search should retry,
            // not inherit a transient network error for the rest of the session.
            synchronized(lock) { inFlight.remove(key) }
            mine.completeExceptionally(t)
            throw t
        }
    }

    /**
     * Forgets everything. Must be called when the downloaded history changes,
     * or a refresh would leave the app still answering from the data it
     * replaced. In-flight loads are left alone: they are already reading the
     * files as they were, and their callers are waiting on that answer.
     */
    fun invalidate() {
        synchronized(lock) { done.clear() }
    }

    companion object {
        /**
         * A search opens at most a handful of boards of forty stops each, so
         * this holds a whole search comfortably while staying a bounded cost
         * for a session that runs all day.
         */
        const val CAPACITY = 256
    }
}
