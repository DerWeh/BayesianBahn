package io.github.derweh.bayesianbahn.data

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.mockito.Mockito.mock
import org.mockito.Mockito.`when`
import java.io.ByteArrayOutputStream
import java.util.zip.GZIPOutputStream

/**
 * The shard cache, against a real server rather than against our beliefs.
 *
 * The hand-written cache this replaced was correct about freshness and wrong
 * about two things nothing tested: it never evicted, and it wrote responses
 * straight to their final path, so a kill mid-write left a truncated file that
 * read as fresh for the whole TTL. Both are OkHttp's problem now — but the way
 * it is *aimed* is ours, and that is what these pin: the origin's own five
 * minutes must not become our refresh interval, an unchanged shard must cost a
 * 304 and not a download, a 404 must be remembered, and a dead network must
 * still return what we already had.
 */
class CachedFetcherTest {

    @get:Rule
    val temp = TemporaryFolder()

    private lateinit var server: MockWebServer
    private lateinit var context: Context

    @Before
    fun setUp() {
        server = MockWebServer().also { it.start() }
        context = mock(Context::class.java)
        `when`(context.cacheDir).thenReturn(temp.newFolder())
    }

    @After
    fun tearDown() = server.shutdown()

    private fun gzip(text: String): ByteArray {
        val out = ByteArrayOutputStream()
        GZIPOutputStream(out).use { it.write(text.toByteArray()) }
        return out.toByteArray()
    }

    private fun fetcher() = CachedFetcher(context, OkHttpClient())

    private fun url(key: String) = server.url("/$key.jgz").toString()

    /** What the shard branches really send: a CDN's five minutes, and an ETag. */
    private fun shard(text: String, etag: String = "\"v1\"") = MockResponse()
        .setHeader("Cache-Control", "max-age=300")
        .setHeader("ETag", etag)
        .setBody(okio.Buffer().write(gzip(text)))

    /**
     * The same response once its five minutes are up.
     *
     * A unit test has no clock to wind forward, so staleness is arranged by
     * serving `max-age=0` rather than by waiting. This is the state a real
     * shard reaches after five minutes, and both TTLs the app asks for — 18
     * hours for the recent overlay, a week for the base — are far past it, so
     * this is the regime every refresh actually happens in.
     */
    private fun staleShard(text: String, etag: String = "\"v1\"") = MockResponse()
        .setHeader("Cache-Control", "max-age=0")
        .setHeader("ETag", etag)
        .setBody(okio.Buffer().write(gzip(text)))

    private val day = 24 * 60 * 60 * 1000L

    @Test
    fun `a shard is served from the cache well past the origin's own five minutes`() {
        server.enqueue(shard("RE 4711"))
        val f = fetcher()
        assertArrayEquals("RE 4711".toByteArray(), f.bytes("d", "k", url("k"), day))
        // No second response is enqueued: a request here would fail the test.
        assertArrayEquals("RE 4711".toByteArray(), f.bytes("d", "k", url("k"), day))
        assertEquals("the second read should not have left the cache", 1, server.requestCount)
    }

    @Test
    fun `past the ttl an unchanged shard costs a 304, not a download`() {
        server.enqueue(staleShard("RE 4711"))
        val f = fetcher()
        assertArrayEquals("RE 4711".toByteArray(), f.bytes("d", "k", url("k"), day))

        server.enqueue(MockResponse().setResponseCode(304))
        // ttl 0: anything stored is stale, so this must revalidate.
        assertArrayEquals("RE 4711".toByteArray(), f.bytes("d", "k", url("k"), 0))
        assertEquals(2, server.requestCount)
        server.takeRequest()
        assertEquals("\"v1\"", server.takeRequest().getHeader("If-None-Match"))
    }

    @Test
    fun `a republished shard replaces the stored one`() {
        server.enqueue(staleShard("old"))
        val f = fetcher()
        assertArrayEquals("old".toByteArray(), f.bytes("d", "k", url("k"), day))
        server.enqueue(staleShard("new", etag = "\"v2\""))
        assertArrayEquals("new".toByteArray(), f.bytes("d", "k", url("k"), 0))
    }

    @Test
    fun `a train with no shard is remembered as having none`() {
        // The `.miss` marker's job. raw.githubusercontent sends Expires on a
        // 404, so it is storable and max-stale covers it like any answer.
        server.enqueue(
            MockResponse().setResponseCode(404)
                .setHeader("Expires", "Thu, 01 Jan 2037 00:00:00 GMT")
                .setBody("404: Not Found"),
        )
        val f = fetcher()
        assertNull(f.bytes("d", "gone", url("gone"), day))
        assertNull(f.bytes("d", "gone", url("gone"), day))
        assertEquals("a missing shard should not be re-asked for", 1, server.requestCount)
    }

    @Test
    fun `a dead network still returns what was already fetched`() {
        server.enqueue(staleShard("RE 4711"))
        val f = fetcher()
        val address = url("k")
        assertArrayEquals("RE 4711".toByteArray(), f.bytes("d", "k", address, day))
        server.shutdown()
        // Stale by ttl *and* unreachable: the stored copy is all there is, and
        // it beats telling the planner the train has no history.
        assertArrayEquals("RE 4711".toByteArray(), f.bytes("d", "k", address, 0))
    }

    @Test
    fun `nothing cached and no network is simply nothing`() {
        server.shutdown()
        assertNull(fetcher().bytes("d", "k", url("k"), day))
    }
}
