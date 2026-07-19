package io.github.derweh.bayesianbahn.data

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.zip.GZIPInputStream

/**
 * Small disk-backed fetch cache for the per-file data hosted on the repo's
 * data branches: returns gunzipped bytes, remembers misses (404) so unknown
 * keys are not re-asked constantly, and falls back to a stale cached copy
 * when the network is unavailable.
 */
class CachedFetcher(
    private val context: Context,
    private val client: OkHttpClient,
) {

    fun bytes(dirName: String, key: String, url: String, ttlMillis: Long): ByteArray? {
        val dir = File(context.filesDir, dirName).apply { mkdirs() }
        val cached = File(dir, "$key.jgz")
        val miss = File(dir, "$key.miss")
        val now = System.currentTimeMillis()
        fun fresh(f: File) = f.isFile && now - f.lastModified() < ttlMillis

        if (fresh(cached)) return gunzip(cached)
        if (fresh(miss)) return null
        try {
            val request = Request.Builder().url(url).build()
            client.newCall(request).execute().use { response ->
                when {
                    response.isSuccessful -> {
                        cached.writeBytes(response.body!!.bytes())
                        miss.delete()
                    }
                    response.code == 404 -> {
                        miss.writeBytes(ByteArray(0))
                        cached.delete()
                        return null
                    }
                    else -> throw IOException("HTTP ${response.code}")
                }
            }
        } catch (_: IOException) {
            // Offline or flaky: a stale cached copy beats nothing.
        }
        return gunzip(cached)
    }

    private fun gunzip(file: File): ByteArray? {
        if (!file.isFile) return null
        return try {
            file.inputStream().use { GZIPInputStream(it).readBytes() }
        } catch (_: IOException) {
            null
        }
    }
}
