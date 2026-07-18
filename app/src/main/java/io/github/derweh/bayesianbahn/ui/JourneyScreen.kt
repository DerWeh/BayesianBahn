package io.github.derweh.bayesianbahn.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.github.derweh.bayesianbahn.R
import io.github.derweh.bayesianbahn.data.JourneyPlanner
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.roundToInt

private val ZONE = ZoneId.of("Europe/Berlin")
private val HHMM = DateTimeFormatter.ofPattern("HH:mm")

/**
 * Home screen in the familiar from/to style: origin, destination, departure
 * time and the Deutschland-Ticket filter; the app figures out direct trains
 * and transfers and predicts the arrival distribution for each option.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JourneyScreen(viewModel: AppViewModel) {
    var from by rememberSaveable { mutableStateOf("") }
    var to by rememberSaveable { mutableStateOf("") }
    var timeText by rememberSaveable { mutableStateOf("") }
    var tomorrow by rememberSaveable { mutableStateOf(false) }
    var deutschlandTicket by rememberSaveable { mutableStateOf(true) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name)) },
                actions = {
                    IconButton(onClick = viewModel::openStationSearch) {
                        Icon(
                            Icons.AutoMirrored.Filled.List,
                            contentDescription = "Station boards",
                        )
                    }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier.padding(padding).fillMaxSize()
                .verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedTextField(
                value = from,
                onValueChange = { from = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("From") },
                singleLine = true,
            )
            OutlinedTextField(
                value = to,
                onValueChange = { to = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("To") },
                singleLine = true,
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedTextField(
                    value = timeText,
                    onValueChange = { timeText = it },
                    modifier = Modifier.width(120.dp),
                    label = { Text("Depart") },
                    placeholder = { Text("now") },
                    singleLine = true,
                )
                FilterChip(
                    selected = !tomorrow,
                    onClick = { tomorrow = false },
                    label = { Text("Today") },
                )
                FilterChip(
                    selected = tomorrow,
                    onClick = { tomorrow = true },
                    label = { Text("Tomorrow") },
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Deutschland-Ticket only", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        "Only regional trains (RE, RB, S, …)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(checked = deutschlandTicket, onCheckedChange = { deutschlandTicket = it })
            }
            Button(
                onClick = {
                    viewModel.planJourney(
                        from, to, departMillis(timeText, tomorrow), deutschlandTicket,
                    )
                },
                enabled = from.isNotBlank() && to.isNotBlank() &&
                    viewModel.journeyState != JourneyState.Loading,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Search connections") }

            when (val state = viewModel.journeyState) {
                JourneyState.Idle -> {}
                JourneyState.Loading -> Box(
                    Modifier.fillMaxWidth().padding(24.dp),
                    contentAlignment = Alignment.Center,
                ) { CircularProgressIndicator() }
                is JourneyState.Error -> Text(
                    state.message,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
                is JourneyState.Loaded -> {
                    Text(
                        "${state.outcome.from.name} → ${state.outcome.to.name}",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    state.outcome.itineraries.forEach { ItineraryCard(it) }
                    Text(
                        "Predicted from each train's real delay history; " +
                            "transfers use the first catchable connection.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

/** "HH:mm" + today/tomorrow → epoch millis; blank means now. */
internal fun departMillis(timeText: String, tomorrow: Boolean): Long {
    val date = LocalDate.now(ZONE).plusDays(if (tomorrow) 1 else 0)
    val time = runCatching { LocalTime.parse(timeText.trim(), HHMM) }.getOrNull()
        ?: if (tomorrow) LocalTime.of(6, 0) else LocalTime.now(ZONE)
    return ZonedDateTime.of(date, time, ZONE).toInstant().toEpochMilli()
}

@Composable
private fun ItineraryCard(itinerary: JourneyPlanner.Itinerary) {
    var expanded by rememberSaveable(itinerary.feeder.id) { mutableStateOf(false) }
    val dist = itinerary.distribution
    val median = itinerary.medianArrivalMillis
    val q10 = itinerary.referenceArrivalMillis + (dist.quantile(0.1) * 60_000).toLong()
    val q90 = itinerary.referenceArrivalMillis + (dist.quantile(0.9) * 60_000).toLong()

    Surface(
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier.fillMaxWidth().clickable { expanded = !expanded },
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "${formatTime(itinerary.departureMillis)} ${itinerary.feeder.label.display}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(Modifier.weight(1f))
                Text(
                    "→ ~${formatTime(median)}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
            Text(
                buildString {
                    if (itinerary.transferStation != null) {
                        append("Change at ${itinerary.transferStation}")
                        itinerary.catchProbability?.let {
                            append("  ·  P(first connection) ${(it * 100).roundToInt()} %")
                        }
                    } else {
                        append("Direct")
                        itinerary.feeder.departure?.liveDelayMinutes?.let {
                            if (it >= 1) append("  ·  now +${it.roundToInt()} min")
                        }
                    }
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                "80% between ${formatTime(q10)} and ${formatTime(q90)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (expanded) {
                DelayDistributionChart(
                    dist = dist,
                    modifier = Modifier.fillMaxWidth().height(140.dp),
                )
                itinerary.connection?.result?.candidates?.forEach { cand ->
                    Text(
                        "${formatTime(cand.candidate.plannedDepartureMillis)} " +
                            "${cand.candidate.label}: " +
                            if (cand.candidate.cancelledLive) "cancelled"
                            else "${(cand.boardProbability * 100).roundToInt()} %",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                itinerary.missProbability?.takeIf { it > 0.005 }?.let {
                    Text(
                        "P(miss all listed connections): ${(it * 100).roundToInt()} %",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }
    }
}
