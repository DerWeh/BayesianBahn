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
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.kxml2.io.KXmlParser

/**
 * From the F-Droid review of 0.1.1: "Plan a connection from this train"
 * dead-ended with `Transfer station "Frankfurt(M) Flughafen Regionalbf" not
 * found.` The transfer chips are IRIS route names, but the transfer was looked
 * up by substring against the station list, which spells that stop
 * "Frankfurt (Main) Flughafen Regionalbahnhof".
 *
 * No string rule bridges that pair — "M" vs "Main", "Regionalbf" vs
 * "Regionalbahnhof" — so it is resolved by asking IRIS for the EVA number.
 */
class TransferLookupTest {

    private val stations = StationRepository.of(listOf(
        Station("8000105", "Frankfurt (Main) Hbf", 1064, 50.10715, 8.66379),
        Station("8070004", "Frankfurt (Main) Flughafen Regionalbahnhof", 539, 50.05122, 8.57125),
        Station("8003200", "Langen (Hess)", 168, 49.99, 8.66),
    ))

    /** The live response for /timetable/station/Frankfurt(M)%20Flughafen%20Regionalbf. */
    private val regionalbfXml = """
        <stations>
        <station p="1|2|Nord" meta="8070003|8089361" name="Frankfurt(M) Flughafen Regionalbf"
                 eva="8070004" ds100="FFLU" db="true"/>
        </stations>
    """.trimIndent()

    private val requests = mutableListOf<String>()

    private fun iris(body: String?): IrisClient {
        val http = OkHttpClient.Builder().addInterceptor { chain: Interceptor.Chain ->
            requests += chain.request().url.encodedPath
            Response.Builder()
                .request(chain.request()).protocol(Protocol.HTTP_1_1)
                .code(if (body == null) 404 else 200).message("ok")
                .body((body ?: "").toResponseBody("text/xml".toMediaType()))
                .build()
        }.build()
        return IrisClient(IrisParser { KXmlParser() }, client = http)
    }

    @Test
    fun `the reported transfer station now resolves`() = runBlocking {
        val matcher = RouteStationMatcher(iris(regionalbfXml), stations)
        val found = matcher.station("Frankfurt(M) Flughafen Regionalbf")
        assertEquals("Frankfurt (Main) Flughafen Regionalbahnhof", found?.name)
        assertEquals("8070004", found?.eva)
    }

    @Test
    fun `no string comparison could have resolved it`() {
        // Guards the claim above: if these ever match, the IRIS lookup is no
        // longer what makes this work and the test below is measuring nothing.
        assertTrue(
            !StationNames.matches(
                "Frankfurt(M) Flughafen Regionalbf",
                "Frankfurt (Main) Flughafen Regionalbahnhof",
            ),
        )
    }

    @Test
    fun `a station the list already spells the same way costs no request`() = runBlocking {
        val matcher = RouteStationMatcher(iris(regionalbfXml), stations)
        // The review noted this screen worked for RE60 via Langen(Hess).
        assertEquals("Langen (Hess)", matcher.station("Langen(Hess)")?.name)
        assertTrue("resolved locally, so IRIS must not be asked", requests.isEmpty())
    }

    @Test
    fun `an unknown station stays unknown rather than resolving to something else`() = runBlocking {
        val matcher = RouteStationMatcher(iris(null), stations)  // 404
        assertNull(matcher.station("Somewhere That Does Not Exist"))
    }

    @Test
    fun `a failed lookup is asked for only once`() = runBlocking {
        val matcher = RouteStationMatcher(iris(null), stations)
        matcher.station("Somewhere That Does Not Exist")
        matcher.station("Somewhere That Does Not Exist")
        assertEquals(1, requests.size)
    }
}
