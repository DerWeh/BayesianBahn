package io.github.derweh.bayesianbahn.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.derweh.bayesianbahn.R
import io.github.derweh.bayesianbahn.api.EventInfo
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.data.Station
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val ZONE = ZoneId.of("Europe/Berlin")
private val HHMM = DateTimeFormatter.ofPattern("HH:mm")

fun formatTime(epochMillis: Long): String =
    Instant.ofEpochMilli(epochMillis).atZone(ZONE).format(HHMM)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BoardScreen(
    viewModel: AppViewModel,
    station: Station,
    onStopSelected: (TimetableStop) -> Unit,
    onBack: () -> Unit,
) {
    var tab by remember { mutableIntStateOf(0) }
    val state = viewModel.boardState

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(station.name, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.back),
                        )
                    }
                },
                actions = {
                    IconButton(onClick = viewModel::refreshBoard) {
                        Icon(Icons.Default.Refresh, contentDescription = stringResource(R.string.refresh))
                    }
                },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            TabRow(selectedTabIndex = tab) {
                Tab(tab == 0, onClick = { tab = 0 }, text = { Text(stringResource(R.string.departures)) })
                Tab(tab == 1, onClick = { tab = 1 }, text = { Text(stringResource(R.string.arrivals)) })
            }
            when (state) {
                BoardState.Loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
                is BoardState.Error -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text(
                        stringResource(R.string.error_network),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(24.dp),
                    )
                }
                is BoardState.Loaded -> BoardList(
                    stops = state.stops,
                    departures = tab == 0,
                    onStopSelected = onStopSelected,
                )
            }
        }
    }
}

@Composable
private fun BoardList(
    stops: List<TimetableStop>,
    departures: Boolean,
    onStopSelected: (TimetableStop) -> Unit,
) {
    val now = System.currentTimeMillis()
    val rows = stops
        .mapNotNull { stop ->
            val event = if (departures) stop.departure else stop.arrival
            val time = event?.bestTime ?: return@mapNotNull null
            if (time < now - 5 * 60_000) return@mapNotNull null
            Triple(stop, event, time)
        }
        .sortedBy { it.third }

    if (rows.isEmpty()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(stringResource(R.string.no_trains))
        }
        return
    }
    LazyColumn {
        items(rows, key = { "${it.first.id}/$departures" }) { (stop, event, _) ->
            BoardRow(stop, event, departures, onClick = { onStopSelected(stop) })
            HorizontalDivider()
        }
    }
}

@Composable
private fun BoardRow(
    stop: TimetableStop,
    event: EventInfo,
    departures: Boolean,
    onClick: () -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.width(64.dp)) {
            val planned = event.plannedTime
            Text(
                planned?.let(::formatTime) ?: "—",
                style = MaterialTheme.typography.titleMedium,
                textDecoration = if (event.cancelled) TextDecoration.LineThrough else null,
            )
            val delay = event.liveDelayMinutes
            if (event.cancelled) {
                Text(
                    stringResource(R.string.cancelled),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error,
                )
            } else if (delay != null && delay != 0.0) {
                Text(
                    "+${delay.toInt()}",
                    style = MaterialTheme.typography.labelMedium,
                    color = if (delay >= 5) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.secondary,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
        Column(Modifier.weight(1f)) {
            Text(stop.label.display, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
            val towards = if (departures) stop.destination else stop.origin
            Text(
                (if (departures) "→ " else "← ") + (towards ?: "?"),
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        Spacer(Modifier.width(8.dp))
        event.platform?.let { platform ->
            Text(
                "Gl. $platform",
                style = MaterialTheme.typography.labelLarge,
                color = if (event.changedPlatform != null) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
        }
    }
}
