package io.github.derweh.bayesianbahn

import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.IrisParser
import io.github.derweh.bayesianbahn.data.HistoryRepository
import io.github.derweh.bayesianbahn.data.Predictor
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.kxml2.io.KXmlParser
import java.io.File
import java.util.zip.GZIPInputStream

/**
 * End-to-end smoke test: live IRIS + real bundled shards + predictor.
 * Needs network; opt in with `E2E=1 pixi run ./gradlew :app:testDebugUnitTest --tests '*SmokeE2E*'`.
 */
class SmokeE2E {

    @Test
    fun `live board to forecast`() = runBlocking {
        org.junit.Assume.assumeNotNull(System.getenv("E2E"))
        val client = IrisClient(IrisParser { KXmlParser() })
        val stops = client.board("8000013", hours = 2)
        println("== live board Augsburg Hbf: ${stops.size} stops")
        check(stops.isNotEmpty()) { "empty board" }

        val assets = File("src/main/assets/history")
        var forecasts = 0
        for (stop in stops.take(40)) {
            val key = HistoryRepository.shardKey("${stop.label.category} ${stop.label.number}")
            val f = File(assets, "$key.jgz")
            val history = if (f.exists()) {
                HistoryRepository.parseShard(
                    GZIPInputStream(f.inputStream()).readBytes().decodeToString(),
                )
            } else null
            val event = stop.arrival ?: stop.departure ?: continue
            val forecast = Predictor().forecast(
                history = history,
                stationEva = "8000013",
                stationName = "Augsburg Hbf",
                trainCategory = stop.label.category,
                plannedTimeMillis = event.plannedTime ?: continue,
                liveDelayMinutes = event.liveDelayMinutes,
            )
            val d = forecast.distribution
            println(
                "${stop.label.display.padEnd(10)} planned=${event.plannedTime} " +
                    "live=${event.liveDelayMinutes} src=${forecast.source} " +
                    "runs=${forecast.runCount} median=%+.1f 80%%=[%+.1f, %+.1f] P(<=5)=%.2f".format(
                        d.quantile(0.5), d.quantile(0.1), d.quantile(0.9), d.cdf(5.0),
                    ),
            )
            forecasts++
        }
        println("== $forecasts forecasts produced")
        check(forecasts > 0)
    }
}
