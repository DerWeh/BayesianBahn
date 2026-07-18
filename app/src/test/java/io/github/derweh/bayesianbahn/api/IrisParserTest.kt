package io.github.derweh.bayesianbahn.api

import org.junit.Assert.assertEquals
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
            <ar ct="2607161920" cp="9"/>
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
}
