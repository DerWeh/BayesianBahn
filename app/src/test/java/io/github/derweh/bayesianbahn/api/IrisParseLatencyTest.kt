package io.github.derweh.bayesianbahn.api

import org.junit.Assert.assertTrue
import org.junit.Test
import org.kxml2.io.KXmlParser

/**
 * What parsing one station's IRIS documents costs.
 *
 * It is measured because of where it happens rather than because it is slow.
 * `IrisClient.get` leaves the IO dispatcher as soon as the bytes are in hand,
 * so every `parsePlan`, `parseChanges` and `merge` runs on whatever dispatcher
 * the caller is on — and every `viewModelScope.launch` in `AppViewModel` names
 * none, which means `Dispatchers.Main`. A journey search opens several boards,
 * and each board is three or four plan documents plus one `fchg`.
 *
 * The sizes here are real. Measured against Ulm Hbf on 2026-09-03, one hour of
 * `plan` is 10.6 KB and 35 stops, while `fchg` for the same station is 168 KB,
 * 323 stops and 1,368 disruption messages — the messages are most of it. The
 * documents below are generated to that shape rather than captured, because
 * captured IRIS responses are deliberately not committed (see .gitignore).
 */
class IrisParseLatencyTest {

    private val parser = IrisParser { KXmlParser() }

    private fun message(i: Int) =
        """<m id="r$i" t="h" from="2609030630" to="2609031956" cat="Information" """ +
            """ts="2609030157" ts-tts="26-09-03 01:57:57.988" pr="2"/>"""

    private fun changes(stops: Int, messagesPerStop: Int) = buildString {
        append("""<timetable station="Ulm Hbf" eva="8000170">""")
        for (s in 0 until stops) {
            append("""<s id="90423952609427960$s-2609030810-7" eva="8000170">""")
            repeat(messagesPerStop) { append(message(s * messagesPerStop + it)) }
            append("""<ar ct="26090310${(s % 60).toString().padStart(2, '0')}"/>""")
            append("""<dp ct="26090311${(s % 60).toString().padStart(2, '0')}"/>""")
            append("</s>")
        }
        append("</timetable>")
    }

    private fun plan(stops: Int) = buildString {
        append("""<timetable station="Ulm Hbf" eva="8000170">""")
        for (s in 0 until stops) {
            append("""<s id="90423952609427960$s-2609030810-7">""")
            append("""<tl f="N" t="p" o="80" c="RE" n="${4000 + s}"/>""")
            append(
                """<ar pt="26090310${(s % 60).toString().padStart(2, '0')}" pp="2" l="9" """ +
                    """ppth="Augsburg Hbf|Günzburg|Neu-Ulm"/>""",
            )
            append(
                """<dp pt="26090311${(s % 60).toString().padStart(2, '0')}" pp="2" l="9" """ +
                    """ppth="Ehingen(Donau)|Herbertingen|Sigmaringen"/>""",
            )
            append("</s>")
        }
        append("</timetable>")
    }

    private fun time(rounds: Int, body: () -> Unit): Double {
        repeat(rounds / 4) { body() }
        val start = System.nanoTime()
        repeat(rounds) { body() }
        return (System.nanoTime() - start) / 1e6 / rounds
    }

    @Test
    fun `one station's documents, at the sizes IRIS really serves`() {
        val fchg = changes(stops = 323, messagesPerStop = 4)
        val hour = plan(stops = 35)
        println("fchg ${fchg.length / 1024} KB, plan ${hour.length / 1024} KB")

        val changesMs = time(40) { parser.parseChanges(fchg) }
        val planMs = time(200) { parser.parsePlan(hour) }
        // What one board costs: IrisClient.board fetches `hours` plan slices
        // and one fchg, then merges.
        val board = changesMs + 3 * planMs
        println(
            "parseChanges %.1f ms, parsePlan %.2f ms, one 3-hour board %.1f ms"
                .format(changesMs, planMs, board),
        )
        // A journey search opens up to eight boards.
        println("eight boards: %.0f ms".format(8 * board))
        assertTrue("a board's parse took $board ms", board < 500.0)
    }
}
