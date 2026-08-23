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
                            contentDescription = stringResource(R.string.station_boards),
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
            Surface(
                shape = MaterialTheme.shapes.medium,
                color = MaterialTheme.colorScheme.tertiaryContainer,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    stringResource(R.string.early_release_warning),
                    Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                )
            }
            StationSuggestField(
                value = from,
                onValueChange = { from = it },
                label = stringResource(R.string.label_from),
                suggest = viewModel::suggestStations,
            )
            StationSuggestField(
                value = to,
                onValueChange = { to = it },
                label = stringResource(R.string.label_to),
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
                        } ?: stringResource(R.string.depart_now),
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
                        }) { Text(stringResource(R.string.action_ok)) }
                    },
                    dismissButton = {
                        TextButton(onClick = { showDatePicker = false }) { Text(stringResource(R.string.action_cancel)) }
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
                    Text(
                        stringResource(R.string.deutschland_ticket_only),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Text(
                        stringResource(R.string.deutschland_ticket_hint),
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
            ) { Text(stringResource(R.string.search_connections)) }

            when (val state = viewModel.journeyState) {
                JourneyState.Idle -> {}
                // A future date is planned from the downloaded history, which
                // means fetching a shard per train — minutes on a slow device.
                // A bare spinner for that long is indistinguishable from a hang,
                // so say what is happening and roughly how long it takes.
                JourneyState.Loading -> Column(
                    Modifier.fillMaxWidth().padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    CircularProgressIndicator()
                    Text(
                        if (beyondLivePlan(epochDay)) {
                            stringResource(R.string.searching_historical_timetable)
                        } else {
                            stringResource(R.string.searching_connections)
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                is JourneyState.Error -> Text(
                    state.message.text(),
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
                            stringResource(
                                if (state.outcome.offline) {
                                    R.string.synthetic_notice_offline
                                } else {
                                    R.string.synthetic_notice_beyond_horizon
                                },
                            ) +
                                if (crossesChange) {
                                    stringResource(
                                        R.string.synthetic_notice_timetable_change,
                                        change.format(DateTimeFormatter.ofPattern("dd.MM.")),
                                    )
                                } else "",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.tertiary,
                        )
                    }
                    state.outcome.itineraries.forEach { ItineraryCard(it) }
                    Text(
                        stringResource(R.string.journey_footnote),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

/** Picked time + date → epoch millis; null hour means now, null day today. */
/**
 * True when the requested date is past IRIS's ~1-day plan horizon, so the
 * search falls back to the historical timetable and takes much longer.
 */
internal fun beyondLivePlan(epochDay: Long?, today: LocalDate = LocalDate.now(ZONE)): Boolean =
    epochDay != null && epochDay > today.toEpochDay() + 1

internal fun departMillis(hour: Int?, minute: Int?, epochDay: Long?): Long {
    val today = LocalDate.now(ZONE)
    val date = epochDay?.let { LocalDate.ofEpochDay(it) } ?: today
    val time = hour?.let { LocalTime.of(it, minute ?: 0) }
        ?: if (date != today) LocalTime.of(6, 0) else LocalTime.now(ZONE)
    return ZonedDateTime.of(date, time, ZONE).toInstant().toEpochMilli()
}

@Composable
internal fun formatDuration(millis: Long): String {
    val totalMin = (millis / 60_000L).coerceAtLeast(0)
    val h = totalMin / 60
    val m = totalMin % 60
    return if (h > 0) {
        stringResource(R.string.duration_hours_minutes, h, m)
    } else {
        stringResource(R.string.duration_minutes, m)
    }
}

@Composable
private fun dateLabel(epochDay: Long?): String {
    val today = LocalDate.now(ZONE)
    val date = epochDay?.let { LocalDate.ofEpochDay(it) } ?: today
    return when (date) {
        today -> stringResource(R.string.date_today)
        today.plusDays(1) -> stringResource(R.string.date_tomorrow)
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
        title = { Text(stringResource(R.string.departure_time)) },
        text = { TimePicker(state = state) },
        confirmButton = {
            TextButton(onClick = { onConfirm(state.hour, state.minute) }) { Text(stringResource(R.string.action_ok)) }
        },
        dismissButton = {
            Row {
                TextButton(onClick = onNow) { Text(stringResource(R.string.action_now)) }
                TextButton(onClick = onDismiss) { Text(stringResource(R.string.action_cancel)) }
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
                        append(
                            stringResource(
                                R.string.itinerary_change_at,
                                itinerary.transferStation,
                            ),
                        )
                        itinerary.catchProbability?.let {
                            append(
                                stringResource(
                                    R.string.itinerary_catch_probability,
                                    (it * 100).roundToInt(),
                                ),
                            )
                        }
                    } else {
                        append(stringResource(R.string.itinerary_direct))
                        itinerary.feeder.departure?.liveDelayMinutes?.let {
                            if (it >= 1) {
                                append(
                                    stringResource(
                                        R.string.itinerary_live_delay,
                                        it.roundToInt(),
                                    ),
                                )
                            }
                        }
                    }
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                stringResource(
                    R.string.itinerary_duration_interval,
                    formatDuration(median - itinerary.departureMillis),
                    formatTime(q10),
                    formatTime(q90),
                ),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (expanded) {
                DelayDistributionChart(
                    dist = dist,
                    modifier = Modifier.fillMaxWidth().height(140.dp),
                    referenceMillis = itinerary.referenceArrivalMillis,
                )
                itinerary.connection?.result?.candidates?.forEach { cand ->
                    Text(
                        stringResource(
                            R.string.itinerary_candidate,
                            formatTime(cand.candidate.plannedDepartureMillis),
                            cand.candidate.label,
                            if (cand.candidate.cancelledLive) {
                                stringResource(R.string.cancelled_inline)
                            } else {
                                stringResource(
                                    R.string.percent,
                                    (cand.boardProbability * 100).roundToInt(),
                                )
                            },
                        ),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                itinerary.missProbability?.takeIf { it > 0.005 }?.let {
                    Text(
                        stringResource(R.string.miss_all_listed, (it * 100).roundToInt()),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }
    }
}
