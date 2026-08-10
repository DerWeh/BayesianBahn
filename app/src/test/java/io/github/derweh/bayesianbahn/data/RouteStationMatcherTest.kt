package io.github.derweh.bayesianbahn.data

import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.IrisParser
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.kxml2.io.KXmlParser
import java.io.IOException

/**
 * Route lists name stations; the app knows EVA numbers. Rather than guessing
 * whether two spellings are the same station, IRIS is asked to name it.
 *
 * Driving the real [IrisClient] through a canned HTTP layer keeps the parser,
 * the URL and the caching under test — a hand-written fake client would have
 * tested none of them.
 */
class RouteStationMatcherTest {

    // The live response for /iris-tts/timetable/station/8000144.
    private val turkheimXml = """
        <stations>
        <station p="1|3|4" meta="510921|872268" name="Türkheim(Bay)Bf" eva="8000144"
                 ds100="MTHB" db="true" creationts="26-08-06 10:33:57.008"/>
        </stations>
    """.trimIndent()

    private val turkheim = Station("8000144", "Türkheim (Bay) Bahnhof", 69, 48.04569, 10.61744)

    private var requests = mutableListOf<String>()

    /** Only the destination matters here; the list is needed for construction. */
    private val stations = StationRepository.of(listOf(turkheim))

    private fun client(body: String?, fail: Boolean = false): IrisClient {
        val http = OkHttpClient.Builder().addInterceptor { chain: Interceptor.Chain ->
            requests += chain.request().url.encodedPath
            if (fail) throw IOException("no network")
            Response.Builder()
                .request(chain.request())
                .protocol(Protocol.HTTP_1_1)
                .code(if (body == null) 404 else 200)
                .message("ok")
                .body((body ?: "").toResponseBody("text/xml".toMediaType()))
                .build()
        }.build()
        return IrisClient(IrisParser { KXmlParser() }, client = http)
    }

    @Test
    fun `the IRIS spelling of the destination matches its route entries`() = runBlocking {
        val isDestination = RouteStationMatcher(client(turkheimXml), stations).matcherFor(turkheim)
        assertTrue(isDestination("Türkheim(Bay)Bf"))
        assertEquals(listOf("/iris-tts/timetable/station/8000144"), requests)
    }

    @Test
    fun `other stations on the route are not the destination`() = runBlocking {
        val isDestination = RouteStationMatcher(client(turkheimXml), stations).matcherFor(turkheim)
        assertFalse(isDestination("Buchloe"))
        assertFalse(isDestination("Mindelheim"))
        assertFalse(isDestination("Rammingen(Bay)"))
    }

    @Test
    fun `the name is resolved once and reused`() = runBlocking {
        val matcher = RouteStationMatcher(client(turkheimXml), stations)
        matcher.matcherFor(turkheim)
        matcher.matcherFor(turkheim)
        matcher.matcherFor(turkheim.copy(name = "spelled differently"))
        assertEquals(1, requests.size)
    }

    @Test
    fun `without a network it falls back to comparing names`() = runBlocking {
        val isDestination = RouteStationMatcher(client(null, fail = true), stations).matcherFor(turkheim)
        assertTrue("the fallback must still find the station", isDestination("Türkheim(Bay)Bf"))
        assertFalse(isDestination("Buchloe"))
    }

    @Test
    fun `a station IRIS does not know falls back to comparing names`() = runBlocking {
        val isDestination = RouteStationMatcher(client(null), stations).matcherFor(turkheim)  // 404
        assertTrue(isDestination("Türkheim (Bay)"))
        assertFalse(isDestination("Buchloe"))
    }

    @Test
    fun `a failed lookup is not retried for every route entry`() = runBlocking {
        val matcher = RouteStationMatcher(client(null, fail = true), stations)
        matcher.matcherFor(turkheim)
        matcher.matcherFor(turkheim)
        assertEquals(1, requests.size)
    }

    /** A name query can return several stations; the eva decides which is meant. */
    @Test
    fun `the right station is picked out of a multi-station response`() = runBlocking {
        val many = """
            <stations>
            <station name="Ulm Hbf" eva="8000170" ds100="TU"/>
            <station name="Ulm-Söflingen" eva="8006724" ds100="TUS"/>
            </stations>
        """.trimIndent()
        val soflingen = Station("8006724", "Ulm-Söflingen", 50)
        val isDestination = RouteStationMatcher(client(many), stations).matcherFor(soflingen)
        assertTrue(isDestination("Ulm-Söflingen"))
        assertFalse("must not take the first entry in the document", isDestination("Ulm Hbf"))
    }

    @Test
    fun `the destination is resolved for the IRIS spelling of a live route`() = runBlocking {
        // The real ppth of a Memmingen departure, as IRIS serves it.
        val route = "Augsburg Hbf|Buchloe|Türkheim(Bay)Bf|Rammingen(Bay)|Mindelheim"
            .split("|")
        val isDestination = RouteStationMatcher(client(turkheimXml), stations).matcherFor(turkheim)
        assertEquals(listOf("Türkheim(Bay)Bf"), route.filter(isDestination))
    }

    @Test
    fun `zero-padded evas resolve to the same station`() = runBlocking {
        val padded = """<stations><station name="Türkheim(Bay)Bf" eva="08000144"/></stations>"""
        val isDestination = RouteStationMatcher(client(padded), stations).matcherFor(turkheim)
        assertTrue(isDestination("Türkheim(Bay)Bf"))
    }
}
