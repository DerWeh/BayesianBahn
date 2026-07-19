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
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.TextButton
import androidx.compose.material3.TimePicker
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.material3.rememberTimePickerState
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
import io.github.derweh.bayesianbahn.model.GermanCalendar
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
    // null = depart now.
    var pickedHour by rememberSaveable { mutableStateOf<Int?>(null) }
    var pickedMinute by rememberSaveable { mutableStateOf<Int?>(null) }
    var showTimePicker by rememberSaveable { mutableStateOf(false) }
    var epochDay by rememberSaveable { mutableStateOf<Long?>(null) } // null = today
    var showDatePicker by rememberSaveable { mutableStateOf(false) }
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
            StationSuggestField(
                value = from,
                onValueChange = { from = it },
                label = "From",
                suggest = viewModel::suggestStations,
            )
            StationSuggestField(
                value = to,
                onValueChange = { to = it },
                label = "To",
                suggest = viewModel::suggestStations,
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(onClick = { showTimePicker = true }) {
                    Icon(Icons.Default.Schedule, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text(
                        pickedHour?.let { h ->
                            "%02d:%02d".format(h, pickedMinute ?: 0)
                        } ?: "now",
                    )
                }
                Spacer(Modifier.weight(1f))
                OutlinedButton(onClick = { showDatePicker = true }) {
                    Icon(Icons.Default.CalendarMonth, contentDescription = null)
                    Spacer(Modifier.width(6.dp))
                    Text(dateLabel(epochDay))
                }
            }
            if (showDatePicker) {
                val state = rememberDatePickerState(
                    initialSelectedDateMillis = (epochDay ?: LocalDate.now(ZONE).toEpochDay()) * 86_400_000L,
                )
                DatePickerDialog(
                    onDismissRequest = { showDatePicker = false },
                    confirmButton = {
                        TextButton(onClick = {
                            epochDay = state.selectedDateMillis?.let { it / 86_400_000L }
                            showDatePicker = false
                        }) { Text("OK") }
                    },
                    dismissButton = {
                        TextButton(onClick = { showDatePicker = false }) { Text("Cancel") }
                    },
                ) { DatePicker(state = state) }
            }
            if (showTimePicker) {
                DepartureTimeDialog(
                    initialHour = pickedHour ?: LocalTime.now(ZONE).hour,
                    initialMinute = pickedMinute ?: LocalTime.now(ZONE).minute,
                    onDismiss = { showTimePicker = false },
                    onNow = {
                        pickedHour = null
                        pickedMinute = null
                        showTimePicker = false
                    },
                    onConfirm = { h, m ->
                        pickedHour = h
                        pickedMinute = m
                        showTimePicker = false
                    },
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
                        from, to, departMillis(pickedHour, pickedMinute, epochDay),
                        deutschlandTicket,
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
                    if (state.outcome.synthetic) {
                        val change = GermanCalendar.nextTimetableChange(LocalDate.now(ZONE))
                        val crossesChange = state.outcome.itineraries.firstOrNull()?.let {
                            java.time.Instant.ofEpochMilli(it.departureMillis)
                                .atZone(ZONE).toLocalDate() >= change
                        } == true
                        Text(
                            "Planned from the historical timetable — DB publishes " +
                                "live plans only about a day ahead. Times may shift; " +
                                "check again closer to departure." +
                                if (crossesChange) {
                                    " Your date lies beyond the timetable change on " +
                                        change.format(DateTimeFormatter.ofPattern("dd.MM.")) +
                                        " — expect larger differences."
                                } else "",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.tertiary,
                        )
                    }
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

/** Picked time + date → epoch millis; null hour means now, null day today. */
internal fun departMillis(hour: Int?, minute: Int?, epochDay: Long?): Long {
    val today = LocalDate.now(ZONE)
    val date = epochDay?.let { LocalDate.ofEpochDay(it) } ?: today
    val time = hour?.let { LocalTime.of(it, minute ?: 0) }
        ?: if (date != today) LocalTime.of(6, 0) else LocalTime.now(ZONE)
    return ZonedDateTime.of(date, time, ZONE).toInstant().toEpochMilli()
}

private fun dateLabel(epochDay: Long?): String {
    val today = LocalDate.now(ZONE)
    val date = epochDay?.let { LocalDate.ofEpochDay(it) } ?: today
    return when (date) {
        today -> "Today"
        today.plusDays(1) -> "Tomorrow"
        else -> date.format(DateTimeFormatter.ofPattern("EE dd.MM."))
    }
}

/** Material time picker in a dialog, with a shortcut back to "depart now". */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DepartureTimeDialog(
    initialHour: Int,
    initialMinute: Int,
    onDismiss: () -> Unit,
    onNow: () -> Unit,
    onConfirm: (Int, Int) -> Unit,
) {
    val state = rememberTimePickerState(
        initialHour = initialHour,
        initialMinute = initialMinute,
        is24Hour = true,
    )
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Departure time") },
        text = { TimePicker(state = state) },
        confirmButton = {
            TextButton(onClick = { onConfirm(state.hour, state.minute) }) { Text("OK") }
        },
        dismissButton = {
            Row {
                TextButton(onClick = onNow) { Text("Now") }
                TextButton(onClick = onDismiss) { Text("Cancel") }
            }
        },
    )
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
