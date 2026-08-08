package io.github.derweh.bayesianbahn.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.IOException

/**
 * From the F-Droid review of 0.1.1: the delay-history refresh failed on the
 * first tap with "stream was reset: REFUSED_STREAM" and succeeded on the
 * second. OkHttp does not retry a stream reset that happens while a body is
 * being read, and 15 MB of history.zip gives it plenty of opportunity, so one
 * unlucky tap looked like a broken updater.
 */
class RetryingIoTest {

    @Test
    fun `a reset stream is retried rather than shown to the user`() {
        var calls = 0
        val result = retryingIo(3, sleep = {}) {
            calls++
            if (calls < 2) throw IOException("stream was reset: REFUSED_STREAM")
            "downloaded"
        }
        assertEquals("downloaded", result)
        assertEquals(2, calls)
    }

    @Test
    fun `a working download is not retried`() {
        var calls = 0
        retryingIo(3, sleep = {}) { calls++ }
        assertEquals(1, calls)
    }

    @Test
    fun `giving up reports the last failure`() {
        var calls = 0
        val thrown = assertThrows(IOException::class.java) {
            retryingIo(3, sleep = {}) {
                calls++
                throw IOException("attempt $calls")
            }
        }
        assertEquals(3, calls)
        assertEquals("attempt 3", thrown.message)
    }

    @Test
    fun `it backs off between tries but not after the last`() {
        val waits = mutableListOf<Long>()
        assertThrows(IOException::class.java) {
            retryingIo(3, sleepMillis = 100, sleep = { waits += it }) {
                throw IOException("nope")
            }
        }
        assertEquals(listOf(100L, 200L), waits)
    }

    @Test
    fun `failures that are not IO problems are not retried`() {
        var calls = 0
        assertThrows(IllegalStateException::class.java) {
            retryingIo(3, sleep = {}) {
                calls++
                throw IllegalStateException("a bug, not a bad connection")
            }
        }
        assertEquals(1, calls)
    }
}
