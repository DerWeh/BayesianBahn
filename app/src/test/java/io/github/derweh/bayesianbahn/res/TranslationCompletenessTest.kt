package io.github.derweh.bayesianbahn.res

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.w3c.dom.Element

/**
 * A translation that silently loses a key falls back to English at runtime, and
 * one that loses a format argument crashes the screen it appears on — neither
 * shows up in a JVM build, and `MissingTranslation` only fires on a release
 * build. This checks both on every test run.
 */
class TranslationCompletenessTest {

    private val translations = listOf("de")

    @Test
    fun `every translation defines every string`() {
        val base = strings("values")
        for (locale in translations) {
            val translated = strings("values-$locale")
            assertEquals(
                "values-$locale is missing strings",
                emptySet<String>(),
                base.keys - translated.keys,
            )
            assertEquals(
                "values-$locale defines strings that do not exist in values",
                emptySet<String>(),
                translated.keys - base.keys,
            )
        }
    }

    @Test
    fun `every translation takes the same format arguments`() {
        val base = strings("values")
        for (locale in translations) {
            for ((key, value) in strings("values-$locale")) {
                assertEquals(
                    "format arguments of $key differ in values-$locale",
                    formatArguments(base.getValue(key)),
                    formatArguments(value),
                )
            }
        }
    }

    @Test
    fun `every translation covers every plural`() {
        val base = plurals("values")
        for (locale in translations) {
            val translated = plurals("values-$locale")
            assertEquals(
                "values-$locale is missing plurals",
                emptySet<String>(),
                base.keys - translated.keys,
            )
            for ((key, quantities) in translated) {
                assertTrue(
                    "plural $key in values-$locale must define at least one and other",
                    quantities.containsAll(setOf("one", "other")),
                )
            }
        }
    }

    /**
     * Android trims a resource value's leading and trailing whitespace and
     * collapses runs of spaces inside it, unless the whole value is wrapped in
     * double quotes. The separators here are "  ·  " with two spaces a side;
     * unquoted they reached the screen as "Memmingen· P(erster Anschluss)",
     * which no test could see and the build had no opinion about.
     */
    @Test
    fun `values whose whitespace matters are quoted`() {
        for (dir in listOf("values") + translations.map { "values-$it" }) {
            for ((key, value) in raw(dir)) {
                val quoted = value.startsWith("\"") && value.endsWith("\"")
                val significant = value != value.trim() || value.contains("  ")
                assertTrue(
                    "$dir/$key has whitespace Android will eat: it must be quoted",
                    !significant || quoted,
                )
            }
        }
    }

    /** Positional specifiers only — `%1$s` and friends, not a literal `%%`. */
    private fun formatArguments(value: String): Set<String> =
        Regex("""%\d+\$[a-zA-Z]""").findAll(value).map { it.value }.toSet()

    /** Values as written in the XML, quotes and all. */
    private fun raw(dir: String): Map<String, String> =
        elements(dir, "string").associate { it.getAttribute("name") to it.textContent }

    /** `translatable="false"` marks a brand name that stays as it is. */
    private fun strings(dir: String): Map<String, String> =
        elements(dir, "string")
            .filterNot { it.getAttribute("translatable") == "false" }
            .associate { it.getAttribute("name") to it.textContent }

    private fun plurals(dir: String): Map<String, Set<String>> =
        elements(dir, "plurals").associate { plural ->
            val quantities = plural.getElementsByTagName("item")
            plural.getAttribute("name") to (0 until quantities.length)
                .map { (quantities.item(it) as Element).getAttribute("quantity") }
                .toSet()
        }

    private fun elements(dir: String, tag: String): List<Element> {
        val file = File("src/main/res/$dir/strings.xml")
        assertTrue("${file.path} does not exist", file.isFile)
        val root = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file)
        val nodes = root.getElementsByTagName(tag)
        return (0 until nodes.length).map { nodes.item(it) as Element }
    }
}
