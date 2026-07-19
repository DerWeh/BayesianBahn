package io.github.derweh.bayesianbahn.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.derweh.bayesianbahn.R
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.data.Forecast
import io.github.derweh.bayesianbahn.data.ForecastSource
import io.github.derweh.bayesianbahn.data.Station
import io.github.derweh.bayesianbahn.model.DelayDistribution
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PredictionScreen(
    viewModel: AppViewModel,
    station: Station,
    stop: TimetableStop,
    onBack: () -> Unit,
) {
    val state = viewModel.predictionState
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "${stop.label.display} → ${stop.destination ?: "?"}",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.back),
                        )
                    }
                },
            )
        },
    ) { padding ->
        when (state) {
            PredictionState.Loading -> Box(
                Modifier.padding(padding).fillMaxSize(),
                contentAlignment = Alignment.Center,
            ) { CircularProgressIndicator() }
            is PredictionState.Loaded -> PredictionContent(
                Modifier.padding(padding),
                station,
                stop,
                state.forecast,
                onPlanConnection = if (stop.departure?.plannedPath?.isNotEmpty() == true) {
                    { viewModel.openConnection(station, stop) }
                } else null,
            )
        }
    }
}

@Composable
private fun PredictionContent(
    modifier: Modifier,
    station: Station,
    stop: TimetableStop,
    forecast: Forecast,
    onPlanConnection: (() -> Unit)? = null,
) {
    val event = stop.arrival ?: stop.departure
    val planned = event?.plannedTime
    val dist = forecast.distribution
    val median = dist.quantile(0.5)
    val q10 = dist.quantile(0.1)
    val q90 = dist.quantile(0.9)

    fun delayToClock(delayMin: Double): String =
        planned?.let { formatTime(it + (delayMin * 60_000).toLong()) } ?: "+${delayMin.roundToInt()}"

    Column(
        modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            buildString {
                append(station.name)
                planned?.let { append("  ·  plan ${formatTime(it)}") }
                event?.platform?.let { append("  ·  Gl. $it") }
            },
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // Hero: the predicted arrival time
        Surface(
            shape = MaterialTheme.shapes.large,
            color = MaterialTheme.colorScheme.primaryContainer,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(Modifier.padding(20.dp)) {
                Text(
                    if (stop.arrival != null) "Predicted arrival" else "Predicted departure",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(
                        delayToClock(median),
                        style = MaterialTheme.typography.displayMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                    Spacer(Modifier.weight(1f))
                    Text(
                        if (median >= 0.5) "+${median.roundToInt()} min" else "on time",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
                Text(
                    "80% between ${delayToClock(q10)} and ${delayToClock(q90)}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            }
        }

        DelayDistributionChart(
            dist = dist,
            modifier = Modifier.fillMaxWidth().height(180.dp),
            referenceMillis = planned,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatTile("P(≤ 5 min late)", "${(dist.cdf(5.0) * 100).roundToInt()} %", Modifier.weight(1f))
            StatTile(
                "P(cancelled)",
                forecast.cancelProbability?.let { "${(it * 100).roundToInt()} %" } ?: "n/a",
                Modifier.weight(1f),
            )
        }

        onPlanConnection?.let {
            OutlinedButton(onClick = it, modifier = Modifier.fillMaxWidth()) {
                Text("Plan a connection from this train")
            }
        }

        Text(
            when (forecast.source) {
                ForecastSource.EMPIRICAL_LIVE ->
                    "Empirical: ${forecast.runCount} past runs of this train here, " +
                        "reweighted to match its current live delay " +
                        "(effective ${forecast.effectiveRuns.roundToInt()} runs)."
                ForecastSource.EMPIRICAL ->
                    "Empirical: ${forecast.runCount} past runs of this train at this station."
                ForecastSource.PRIOR ->
                    "No delay history available for this train — showing a prior " +
                        "estimate for its category and time of day."
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun StatTile(label: String, value: String, modifier: Modifier = Modifier) {
    Surface(
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = modifier,
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        }
    }
}

/**
 * Histogram of the forecast distribution, computed from CDF differences so
 * empirical and parametric distributions render identically. Single series:
 * primary hue for bars, ink for text, recessive baseline; quantile markers
 * at q10/median/q90. With [referenceMillis] set, the x-axis is labelled
 * with clock times (ticks on round minutes) instead of bare +X offsets.
 */
@Composable
fun DelayDistributionChart(
    dist: DelayDistribution,
    modifier: Modifier = Modifier,
    referenceMillis: Long? = null,
) {
    val barColor = MaterialTheme.colorScheme.primary
    val inkMuted = MaterialTheme.colorScheme.onSurfaceVariant
    val ink = MaterialTheme.colorScheme.onSurface
    val baseline = MaterialTheme.colorScheme.outlineVariant
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(fontSize = 10.sp, color = inkMuted)
    val density = LocalDensity.current

    // Bin the distribution over its central 99%.
    val lo = floor(dist.quantile(0.005)).coerceAtLeast(-20.0)
    val hi = ceil(dist.quantile(0.995)).coerceAtMost(180.0).coerceAtLeast(lo + 6)
    val binWidth = when {
        hi - lo <= 30 -> 1
        hi - lo <= 75 -> 2
        else -> 5
    }
    val start = floor(lo / binWidth).toInt() * binWidth
    val bins = generateSequence(start) { it + binWidth }
        .takeWhile { it < hi }
        .map { b -> b to (dist.cdf((b + binWidth).toDouble()) - dist.cdf(b.toDouble())).coerceAtLeast(0.0) }
        .toList()
    val maxP = bins.maxOfOrNull { it.second } ?: return
    val q10 = dist.quantile(0.1)
    val median = dist.quantile(0.5)
    val q90 = dist.quantile(0.9)

    Canvas(modifier) {
        val labelSpace = with(density) { 16.dp.toPx() }
        val plotH = size.height - labelSpace
        val range = (hi - lo).toFloat()
        fun x(delay: Double): Float = ((delay - lo) / range).toFloat() * size.width
        val gap = with(density) { 1.dp.toPx() }

        // bars: anchored to baseline, rounded data-end, 1dp gap
        for ((binStart, p) in bins) {
            val left = x(binStart.toDouble()) + gap / 2
            val right = x((binStart + binWidth).toDouble()) - gap / 2
            val h = (p / maxP).toFloat() * (plotH * 0.92f)
            if (h <= 0f || right <= left) continue
            drawRoundRect(
                color = barColor,
                topLeft = Offset(left, plotH - h),
                size = Size(right - left, h),
                cornerRadius = CornerRadius(with(density) { 2.dp.toPx() }),
            )
        }
        // baseline
        drawLine(baseline, Offset(0f, plotH), Offset(size.width, plotH), strokeWidth = gap)

        // quantile markers
        val dash = PathEffect.dashPathEffect(floatArrayOf(8f, 6f))
        for ((q, emphasized) in listOf(q10 to false, median to true, q90 to false)) {
            val qx = x(q)
            drawLine(
                color = if (emphasized) ink else inkMuted,
                start = Offset(qx, plotH * 0.05f),
                end = Offset(qx, plotH),
                strokeWidth = with(density) { if (emphasized) 2.dp.toPx() else 1.dp.toPx() },
                pathEffect = if (emphasized) null else dash,
            )
        }

        // x-axis labels: ~4 ticks; with a reference, on round clock minutes.
        val tickStep = listOf(1, 2, 5, 10, 15, 30, 60).first { it * 4 >= range }
        val refOffsetMin = referenceMillis?.let {
            java.time.Instant.ofEpochMilli(it).atZone(java.time.ZoneId.of("Europe/Berlin"))
                .let { z -> z.hour * 60 + z.minute }
        } ?: 0
        var tick = ceil((lo + refOffsetMin) / tickStep).toInt() * tickStep - refOffsetMin
        while (tick <= hi) {
            val label = if (referenceMillis != null) {
                formatTime(referenceMillis + tick * 60_000L)
            } else if (tick > 0) "+$tick" else "$tick"
            val measured = textMeasurer.measure(label, labelStyle)
            val tx = (x(tick.toDouble()) - measured.size.width / 2)
                .coerceIn(0f, size.width - measured.size.width)
            drawText(measured, topLeft = Offset(tx, plotH + 2))
            tick += tickStep
        }
    }
}
