package io.github.derweh.bayesianbahn.api

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.kxml2.io.KXmlParser

class IrisParserTest {

    private val parser = IrisParser { KXmlParser() }

    private val planXml = """
        <?xml version='1.0' encoding='UTF-8'?>
        <timetable station='Augsburg Hbf'>
          <s id="123-2607161833-8">
            <tl t="p" o="GYRB" c="RB" n="57177"/>
            <ar pt="2607161904" pp="8" l="RB86" ppth="Dinkelscherben|Neus&#228;&#223;|Augsburg-Oberhausen"/>
            <dp pt="2607161908" pp="8" l="RB86" ppth="Augsburg Haunstetterstra&#223;e|M&#252;nchen Hbf"/>
          </s>
          <s id="456-2607161815-19">
            <tl f="D" t="p" o="80" c="ICE" n="512"/>
            <ar pt="2607161938" pp="5" ppth="M&#252;nchen Hbf|M&#252;nchen-Pasing"/>
          </s>
        </timetable>
    """.trimIndent()

    private val changesXml = """
        <timetable station="Augsburg Hbf" eva="8000013">
          <s id="123-2607161833-8" eva="8000013">
            <m id="r1" t="h" cat="Störung" pr="2" from="2607161800" to="2607162359"/>
            <ar ct="2607161920" cp="9"><m id="r2" t="d" c="43"/></ar>
            <dp ct="2607161923"/>
          </s>
          <s id="999-unknown-1" eva="8000013">
            <ar ct="2607162011"/>
          </s>
          <s id="456-2607161815-19" eva="8000013">
            <ar cs="c"/>
          </s>
        </timetable>
    """.trimIndent()

    @Test
    fun `parses planned stops`() {
        val stops = parser.parsePlan(planXml)
        assertEquals(2, stops.size)

        val rb = stops[0]
        assertEquals("RB", rb.label.category)
        assertEquals("57177", rb.label.number)
        assertEquals("RB86", rb.label.line)
        assertEquals("RB86", rb.label.display)
        assertEquals("8", rb.arrival?.plannedPlatform)
        assertEquals("München Hbf", rb.destination)
        assertEquals("Dinkelscherben", rb.origin)

        val ice = stops[1]
        assertEquals("ICE 512", ice.label.display)
        assertNull(ice.departure)
    }

    @Test
    fun `parses times as europe berlin`() {
        // 2026-07-16 19:04 CEST == 17:04 UTC
        assertEquals(1784221440000L, IrisParser.parseTime("2607161904"))
        assertNull(IrisParser.parseTime(null))
        assertNull(IrisParser.parseTime("garbage"))
    }

    @Test
    fun `merges changes onto plan`() {
        val stops = parser.parsePlan(planXml)
        val changes = parser.parseChanges(changesXml)
        val merged = IrisParser.merge(stops, changes)

        val rb = merged[0]
        assertEquals(IrisParser.parseTime("2607161920"), rb.arrival?.changedTime)
        assertEquals("9", rb.arrival?.changedPlatform)
        assertEquals("9", rb.arrival?.platform)
        assertEquals(16.0, rb.arrival?.liveDelayMinutes)
        assertEquals(IrisParser.parseTime("2607161923"), rb.departure?.changedTime)

        val ice = merged[1]
        assertTrue(ice.arrival!!.cancelled)
    }

    @Test
    fun `reads what db says besides the times`() {
        val change = parser.parseChanges(changesXml).getValue("123-2607161833-8")
        assertEquals(
            listOf(
                StopMessage(StopMessage.Kind.HIM, category = "Störung", priority = 2),
                StopMessage(StopMessage.Kind.DELAY_CAUSE, code = "43"),
            ),
            change.messages,
        )
    }

    @Test
    fun `a blocked route is a disruption even though nothing is cancelled`() {
        // The failure this exists for: DB reports the trip impossible through a
        // notice while the stops keep their times, so a reader of `ct` and `cs`
        // alone sees an ordinary train and predicts an arrival for it.
        val merged = IrisParser.merge(
            parser.parsePlan(planXml), parser.parseChanges(changesXml),
        )
        assertTrue(merged[0].disrupted)
        assertFalse(merged[0].arrival!!.cancelled)
    }

    @Test
    fun `a train db says nothing about is not disrupted`() {
        val merged = IrisParser.merge(
            parser.parsePlan(planXml), parser.parseChanges(changesXml),
        )
        assertFalse(merged[1].disrupted)
    }

    @Test
    fun `roadworks are not a disruption`() {
        // Every second stop in Germany carries a construction notice; treating
        // those as trouble would warn on everything and mean nothing.
        val xml = """
            <timetable station="Augsburg Hbf" eva="8000013">
              <s id="123-2607161833-8" eva="8000013">
                <m id="r1" t="h" cat="Bauarbeiten" pr="1"/>
                <ar ct="2607161920"/>
              </s>
            </timetable>
        """.trimIndent()
        val merged = IrisParser.merge(parser.parsePlan(planXml), parser.parseChanges(xml))
        assertFalse(merged[0].disrupted)
    }

    @Test
    fun `one notice repeated across a stop is kept once`() {
        val xml = """
            <timetable station="Augsburg Hbf" eva="8000013">
              <s id="123-2607161833-8" eva="8000013">
                <m id="a" t="h" cat="Störung" pr="2"/>
                <ar ct="2607161920"><m id="b" t="h" cat="Störung" pr="2"/></ar>
                <dp ct="2607161923"><m id="c" t="h" cat="Störung" pr="2"/></dp>
              </s>
            </timetable>
        """.trimIndent()
        assertEquals(1, parser.parseChanges(xml).getValue("123-2607161833-8").messages.size)
    }
}
