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
data class DataMeta(val generated: String?, val trains: Int?, val updated: Boolean)

/**
 * Downloads the latest history shards so predictions stay fresh without an
 * app update. The pipeline publishes `history.zip` (shards + index + meta);
 * the archive is unpacked into `filesDir/history`, which
 * [HistoryRepository] prefers over the bundled assets.
 */
class DataUpdater(
    private val context: Context,
    private val client: OkHttpClient = OkHttpClient(),
    private val url: String = DATA_URL,
) {

    suspend fun update(): Result<DataMeta> = withContext(Dispatchers.IO) {
        runCatching {
            val tmp = File(context.filesDir, "history.tmp")
            tmp.deleteRecursively()
            tmp.mkdirs()
            try {
                download(tmp)
                if (!File(tmp, META_FILE).isFile) {
                    throw IOException("archive is missing $META_FILE")
                }
                val target = File(context.filesDir, HISTORY_DIR)
                target.deleteRecursively()
                if (!tmp.renameTo(target)) throw IOException("could not activate downloaded data")
            } finally {
                tmp.deleteRecursively()
            }
            readMeta(context) ?: DataMeta(null, null, updated = true)
        }
    }

    private fun download(into: File) {
        val request = Request.Builder().url(url).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("download failed: HTTP ${response.code}")
            ZipInputStream(response.body!!.byteStream()).use { zip ->
                var entry = zip.nextEntry
                while (entry != null) {
                    // Flat archive: reject directories and any path component.
                    val name = entry.name
                    if (!entry.isDirectory && !name.contains('/') && !name.contains('\\')) {
                        File(into, name).outputStream().use { zip.copyTo(it) }
                    }
                    entry = zip.nextEntry
                }
            }
        }
    }

    companion object {
        /** The pipeline uploads each new build here (release with tag `data`). */
        const val DATA_URL =
            "https://github.com/DerWeh/BayesianBahn/releases/download/data/history.zip"
        const val HISTORY_DIR = "history"
        const val META_FILE = "meta.json"

        /** Meta of the data in use: downloaded if present, else bundled. */
        fun readMeta(context: Context): DataMeta? {
            val local = File(File(context.filesDir, HISTORY_DIR), META_FILE)
            val (text, updated) = when {
                local.isFile -> runCatching { local.readText() }.getOrNull() to true
                else -> runCatching {
                    context.assets.open("$HISTORY_DIR/$META_FILE").use {
                        it.readBytes().decodeToString()
                    }
                }.getOrNull() to false
            }
            if (text == null) return null
            val obj = runCatching { Json.parseToJsonElement(text).jsonObject }.getOrNull()
                ?: return null
            return DataMeta(
                generated = obj["generated"]?.jsonPrimitive?.content,
                trains = obj["trains"]?.jsonPrimitive?.intOrNull,
                updated = updated,
            )
        }
    }
}
