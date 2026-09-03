package io.github.derweh.bayesianbahn.data

import android.content.Context
import okhttp3.Cache
import okhttp3.CacheControl
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.zip.GZIPInputStream

/**
 * The one thing callers need from [CachedFetcher]. Narrowing it to this lets
 * [SyntheticTimetable] be tested without an Android `Context`.
 */
fun interface ByteSource {
    fun bytes(dirName: String, key: String, url: String, ttlMillis: Long): ByteArray?
}

/**
 * Fetches the per-file data on the repo's data branches, through OkHttp's own
 * HTTP cache.
 *
 * This used to be a hand-written disk cache: a file per key, a `.miss` marker
 * per 404, freshness by file timestamp. Everything it did, RFC 9111 already
 * specifies and OkHttp already implements — and the hand-written version was
 * missing two things the built-in gets right. It never evicted, so
 * `filesDir/ondemand` grew for the life of the install; and it wrote the
 * response straight to its final path, so a process killed mid-write left a
 * truncated file that looked fresh for the whole TTL and gunzipped to nothing,
 * turning a train into one with no history for as long as it stood.
 * `tools/fetch_shards.py` guards against exactly that with a temp-then-rename
 * and says so; this did not.
 *
 * Three things are worth knowing about how the standard machinery is aimed
 * here:
 *
 *  * **The origin's freshness is not ours.** raw.githubusercontent.com sends
 *    `Cache-Control: max-age=300`, a CDN's number: obeying it would revalidate
 *    every shard every five minutes. [ttlMillis] is applied as `max-stale`
 *    instead, which is the request-side way of saying "a stored answer this
 *    old is fine by me" whatever the origin's own lifetime.
 *  * **Revalidation is now free.** Past that age OkHttp re-asks with the
 *    stored `ETag`, and the branches serve `304` with no body — so the daily
 *    refresh of an unchanged shard costs a round trip rather than the file.
 *  * **A 404 is an answer too.** It arrives with `Expires` and OkHttp stores
 *    it, so the same `max-stale` stops a train with no shard being re-asked
 *    for on every stop it appears at. That was the `.miss` file's job.
 *
 * What is *not* standard, and stays hand-written, is serving a stale copy when
 * the network fails: HTTP has `stale-if-error` but OkHttp does not apply it, so
 * a failure retries against the cache alone.
 */
class CachedFetcher(
    context: Context,
    client: OkHttpClient,
) : ByteSource {

    private val http = client.newBuilder()
        .cache(Cache(File(context.cacheDir, CACHE_DIR), MAX_BYTES))
        .build()

    override fun bytes(dirName: String, key: String, url: String, ttlMillis: Long): ByteArray? {
        val request = Request.Builder()
            .url(url)
            .cacheControl(
                CacheControl.Builder()
                    .maxStale(ttlMillis.toInt().coerceAtLeast(0), TimeUnit.MILLISECONDS)
                    .build(),
            )
            .build()
        return body(request) ?: body(
            // Offline or flaky: a stale cached copy beats nothing, and only
            // the cache is consulted so this cannot wait on the network again.
            request.newBuilder().cacheControl(CacheControl.FORCE_CACHE).build(),
        )
    }

    private fun body(request: Request): ByteArray? = try {
        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) null
            else response.body?.byteStream()?.let { GZIPInputStream(it).readBytes() }
        }
    } catch (_: IOException) {
        null
    }

    companion object {
        const val CACHE_DIR = "http"

        /**
         * Bounded, which the hand-written cache was not. Generous next to what
         * it holds — a train's shard is about a kilobyte, a line's one and a
         * half, the largest measured 13.5 KB — so a heavy user keeps thousands
         * and the least recently used fall out instead of the directory
         * growing until the app is uninstalled.
         */
        const val MAX_BYTES = 32L * 1024 * 1024
    }
}
