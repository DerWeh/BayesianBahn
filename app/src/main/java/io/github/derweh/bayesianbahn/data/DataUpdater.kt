package io.github.derweh.bayesianbahn.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.zip.ZipInputStream

/** Provenance of the delay-history data currently in use. */
data class DataMeta(
    /** Build date of the monthly base data. */
    val baseGenerated: String?,
    /** Newest day covered by the daily "recent" overlay, null if none. */
    val recentThrough: String?,
    val trains: Int?,
    /** True when downloaded data (rather than only the bundled snapshot) is in use. */
    val updated: Boolean,
)

/**
 * Keeps the delay history fresh with minimal downloads. The `data` release
 * hosts three assets: a tiny `meta.json` describing what is current, the
 * monthly `history.zip` base (~16 MB), and the small daily `recent.zip`
 * overlay (~1–3 MB) built from the archive's raw day files. An update
 * fetches `meta.json` first and downloads only the tier that changed.
 * [HistoryRepository] overlays both onto the bundled assets.
 */
class DataUpdater(
    private val context: Context,
    private val client: OkHttpClient = OkHttpClient(),
    private val baseUrl: String = RELEASE_URL,
) {

    suspend fun update(): Result<DataMeta> = withContext(Dispatchers.IO) {
        runCatching {
            val remote = fetchRemoteMeta()
            val localBase = readJson(File(baseDir(context), META_FILE))
                ?: readAssetMeta(context)
            val localRecent = readJson(File(recentDir(context), META_FILE))

            val baseStale = remote?.get("generated") != null &&
                remote["generated"] != localBase?.get("generated")
            val recentStale = remote?.get("recent_through") != null &&
                remote["recent_through"] != localRecent?.get("recent_through")

            if (baseStale) downloadZip("history.zip", baseDir(context))
            if (recentStale) downloadZip("recent.zip", recentDir(context))
            readMeta(context) ?: throw IOException("no data metadata found")
        }
    }

    private fun fetchRemoteMeta(): Map<String, String>? {
        val request = Request.Builder().url(baseUrl + META_FILE).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("meta fetch failed: HTTP ${response.code}")
            return parseMeta(response.body!!.string())
        }
    }

    private fun downloadZip(asset: String, target: File) {
        val tmp = File(context.filesDir, "$asset.tmp")
        tmp.deleteRecursively()
        tmp.mkdirs()
        try {
            val request = Request.Builder().url(baseUrl + asset).build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException("download failed: HTTP ${response.code}")
                }
                ZipInputStream(response.body!!.byteStream()).use { zip ->
                    var entry = zip.nextEntry
                    while (entry != null) {
                        // Flat archive: reject directories and any path component.
                        val name = entry.name
                        if (!entry.isDirectory && !name.contains('/') && !name.contains('\\')) {
                            File(tmp, name).outputStream().use { zip.copyTo(it) }
                        }
                        entry = zip.nextEntry
                    }
                }
            }
            if (!File(tmp, META_FILE).isFile) throw IOException("$asset is missing $META_FILE")
            target.deleteRecursively()
            if (!tmp.renameTo(target)) throw IOException("could not activate downloaded data")
        } finally {
            tmp.deleteRecursively()
        }
    }

    companion object {
        /** The `update-data` workflow keeps these three assets current. */
        const val RELEASE_URL =
            "https://github.com/DerWeh/BayesianBahn/releases/download/data/"
        const val HISTORY_DIR = "history"
        const val RECENT_DIR = "recent"
        const val META_FILE = "meta.json"

        fun baseDir(context: Context) = File(context.filesDir, HISTORY_DIR)
        fun recentDir(context: Context) = File(context.filesDir, RECENT_DIR)

        /** Meta of the data in use: downloaded tiers if present, else bundled. */
        fun readMeta(context: Context): DataMeta? {
            val base = readJson(File(baseDir(context), META_FILE))
            val recent = readJson(File(recentDir(context), META_FILE))
            val bundled = if (base == null) readAssetMeta(context) else null
            val effective = base ?: bundled ?: return null
            return DataMeta(
                baseGenerated = effective["generated"],
                recentThrough = recent?.get("recent_through"),
                trains = effective["trains"]?.toIntOrNull(),
                updated = base != null || recent != null,
            )
        }

        private fun readJson(file: File): Map<String, String>? =
            if (file.isFile) {
                runCatching { file.readText() }.getOrNull()?.let(::parseMeta)
            } else null

        private fun readAssetMeta(context: Context): Map<String, String>? =
            runCatching {
                context.assets.open("$HISTORY_DIR/$META_FILE").use {
                    it.readBytes().decodeToString()
                }
            }.getOrNull()?.let(::parseMeta)

        private fun parseMeta(text: String): Map<String, String>? {
            val obj = runCatching { Json.parseToJsonElement(text).jsonObject }.getOrNull()
                ?: return null
            return obj.entries.mapNotNull { (k, v) ->
                val prim = runCatching { v.jsonPrimitive }.getOrNull() ?: return@mapNotNull null
                (prim.intOrNull?.toString() ?: prim.content).let { k to it }
            }.toMap()
        }
    }
}
