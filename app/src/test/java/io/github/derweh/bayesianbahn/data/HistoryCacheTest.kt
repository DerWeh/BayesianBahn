package io.github.derweh.bayesianbahn.data

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.yield
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicInteger

/**
 * A future-date search spent minutes loading the same handful of trains over
 * and over: once per board that named them, again when the itinerary was
 * scored, and — after boards started loading concurrently — several times at
 * once. These pin the two savings, and the bound that keeps the memo from
 * becoming a session-long leak.
 */
class HistoryCacheTest {

    private fun history(name: String) = TrainHistory(name, "RE", emptyMap())

    @Test
    fun `a repeated lookup loads once`() = runTest {
        val cache = HistoryCache()
        val calls = AtomicInteger()
        repeat(5) {
            cache.get("RE_1") { calls.incrementAndGet(); history("RE 1") }
        }
        assertEquals(1, calls.get())
        assertEquals(1, cache.loads)
    }

    @Test
    fun `the remembered value is the one that was loaded`() = runTest {
        val cache = HistoryCache()
        val first = cache.get("RE_1") { history("RE 1") }
        val second = cache.get("RE_1") { history("something else") }
        assertSame(first, second)
        assertEquals("RE 1", second?.trainName)
    }

    @Test
    fun `a train with no history is remembered as having none`() = runTest {
        // The expensive case: a miss costs up to four round trips, and boards
        // are full of trains with no shard at all.
        val cache = HistoryCache()
        val calls = AtomicInteger()
        repeat(4) {
            assertNull(cache.get("RE_404") { calls.incrementAndGet(); null })
        }
        assertEquals(1, calls.get())
    }

    @Test
    fun `concurrent lookups of one train share a single load`() = runTest {
        val cache = HistoryCache()
        val calls = AtomicInteger()
        val release = CompletableDeferred<Unit>()
        // Six coroutines ask at once, as six parallel board stops would.
        val waiters = (1..6).map {
            async {
                cache.get("RE_1") {
                    calls.incrementAndGet()
                    release.await()
                    history("RE 1")
                }
            }
        }
        // Let all six reach the cache before the load is allowed to finish;
        // otherwise the first would complete and the rest would simply hit a
        // warm memo, which is a different property than the one under test.
        yield()
        release.complete(Unit)
        val results = waiters.awaitAll()
        assertEquals("all six should join one load", 1, calls.get())
        assertTrue(results.all { it?.trainName == "RE 1" })
    }

    @Test
    fun `the memo does not grow past its capacity`() = runTest {
        val cache = HistoryCache(capacity = 2)
        val calls = AtomicInteger()
        val load: suspend () -> TrainHistory? = { calls.incrementAndGet(); history("x") }
        cache.get("a", load); cache.get("b", load); cache.get("c", load)
        assertEquals(3, calls.get())
        // "a" was evicted when "c" arrived, so asking again must reload it.
        cache.get("a", load)
        assertEquals(4, calls.get())
        // "c" is still resident.
        cache.get("c", load)
        assertEquals(4, calls.get())
    }

    @Test
    fun `invalidate forgets everything`() = runTest {
        val cache = HistoryCache()
        val calls = AtomicInteger()
        val load: suspend () -> TrainHistory? = { calls.incrementAndGet(); history("x") }
        cache.get("a", load)
        cache.invalidate()
        cache.get("a", load)
        assertEquals("a refreshed download must not be served from memory", 2, calls.get())
    }

    @Test
    fun `a failed load is not remembered`() = runTest {
        val cache = HistoryCache()
        val calls = AtomicInteger()
        repeat(2) {
            runCatching {
                cache.get("a") {
                    calls.incrementAndGet()
                    throw IllegalStateException("network down")
                }
            }
        }
        assertEquals("a transient failure must not stick for the session", 2, calls.get())
    }

    @Test
    fun `a failure reaches the coroutines that joined the load`() = runTest {
        val cache = HistoryCache()
        val release = CompletableDeferred<Unit>()
        val started = CompletableDeferred<Unit>()
        val first = async {
            runCatching {
                cache.get("a") {
                    started.complete(Unit)
                    release.await()
                    throw IllegalStateException("boom")
                }
            }
        }
        started.await()
        val joined = async { runCatching { cache.get("a") { history("never") } } }
        yield()
        release.complete(Unit)
        assertTrue(first.await().isFailure)
        assertTrue("a joiner must not silently get null", joined.await().isFailure)
    }
}
