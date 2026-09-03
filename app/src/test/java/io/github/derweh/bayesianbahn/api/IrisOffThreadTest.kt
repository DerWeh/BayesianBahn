package io.github.derweh.bayesianbahn.api

import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertTrue
import org.junit.Test
import org.kxml2.io.KXmlParser
import java.util.concurrent.Executors

/**
 * The board must not be parsed on the thread that asked for it.
 *
 * `IrisClient` used to leave the caller's dispatcher only for the requests, so
 * the XML parse — the larger half, since a station's `fchg` runs to 168 KB
 * against 10.6 KB for an hour of `plan` — ran wherever the caller was. Every
 * `viewModelScope.launch` reaching it named no dispatcher, and that is
 * `Dispatchers.Main`: a journey search froze the spinner it had just started
 * and unfroze only once the work was done.
 *
 * A unit test cannot see Android's main thread, so this pins the property that
 * matters and is observable here: whichever thread calls in, the parse happens
 * on a different one.
 */
class IrisOffThreadTest {

    private val plan = """
        <timetable station="Ulm Hbf" eva="8000170"><s id="1-2609030810-7">
        <tl f="N" t="p" o="80" c="RE" n="4711"/>
        <ar pt="2609031000" pp="2" l="9" ppth="Augsburg Hbf"/>
        <dp pt="2609031002" pp="2" l="9" ppth="Ehingen(Donau)"/>
        </s></timetable>
    """.trimIndent()

    private fun stubbing(body: String) = OkHttpClient.Builder()
        .addInterceptor { chain ->
            Response.Builder()
                .request(chain.request())
                .protocol(Protocol.HTTP_1_1)
                .code(200)
                .message("OK")
                .body(body.toResponseBody())
                .build()
        }
        .build()

    /**
     * Runs [call] on a thread of its own and reports which threads parsed.
     *
     * Thread *identity*, not name: coroutines append "@coroutine#n" to the name
     * of whatever thread they run on, so a comparison by name passes whatever
     * the code actually does.
     */
    private fun parsedOn(
        body: String,
        call: suspend (IrisClient) -> Unit,
    ): Pair<Set<Thread>, Thread> {
        val parsers = mutableSetOf<Thread>()
        val parser = IrisParser {
            synchronized(parsers) { parsers += Thread.currentThread() }
            KXmlParser()
        }
        val client = IrisClient(parser, client = stubbing(body))
        val caller = Executors.newSingleThreadExecutor { r -> Thread(r, "the-caller") }
        try {
            lateinit var callerThread: Thread
            runBlocking {
                withContext(caller.asCoroutineDispatcher()) {
                    callerThread = Thread.currentThread()
                    call(client)
                }
            }
            return synchronized(parsers) { parsers.toSet() } to callerThread
        } finally {
            caller.shutdown()
        }
    }

    @Test
    fun `the board is parsed off the thread that asked for it`() {
        var stops = 0
        val (parsers, caller) = parsedOn(plan) { stops = it.board("8000170", hours = 1).size }
        assertTrue("the stub board should have parsed", stops > 0)
        assertTrue("nothing was parsed at all", parsers.isNotEmpty())
        assertTrue(
            "the XML was parsed on the caller's thread; from a launch that names " +
                "no dispatcher, that thread is the UI thread",
            caller !in parsers,
        )
    }

    @Test
    fun `a station lookup is parsed off the thread that asked for it`() {
        val stations = """<stations><station eva="8000170" name="Ulm Hbf"/></stations>"""
        val (parsers, caller) = parsedOn(stations) { it.stations("Ulm") }
        assertTrue("nothing was parsed at all", parsers.isNotEmpty())
        assertTrue("the station XML was parsed on the caller's thread", caller !in parsers)
    }
}
