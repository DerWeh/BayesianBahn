package io.github.derweh.bayesianbahn.ui

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
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.derweh.bayesianbahn.R
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.data.ConnectionPlanner
import io.github.derweh.bayesianbahn.data.Station
import io.github.derweh.bayesianbahn.model.DeutschlandTicket
import kotlin.math.roundToInt

/**
 * Connection planner: feeder train → transfer → destination. The user picks a
 * transfer stop on the feeder's onward route plus a destination; the screen
 * shows the Bayes-propagated final arrival distribution and per-train catch
 * probabilities.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConnectionScreen(
    viewModel: AppViewModel,
    station: Station,
    stop: TimetableStop,
    onBack: () -> Unit,
) {
    val routeStations = stop.departure?.plannedPath ?: emptyList()
    var transfer by rememberSaveable(stop.id) {
        mutableStateOf(routeStations.firstOrNull() ?: "")
    }
    var destination by rememberSaveable(stop.id) { mutableStateOf("") }
    var transferMinutes by rememberSaveable(stop.id) { mutableStateOf(5) }
    // Riding a regional train usually means holding a Deutschland-Ticket.
    var deutschlandTicket by rememberSaveable(stop.id) {
        mutableStateOf(DeutschlandTicket.covers(stop.label.category))
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Connection from ${stop.label.display}",
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
        Column(
            Modifier.padding(padding).fillMaxSize()
                .verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                "Riding ${stop.label.display} from ${station.name} — change at:",
                style = MaterialTheme.typography.bodyMedium,
            )
            TransferPicker(routeStations, transfer) { transfer = it }
            OutlinedTextField(
                value = destination,
                onValueChange = { destination = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Destination station") },
                singleLine = true,
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Transfer time", style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.weight(1f))
                OutlinedButton(
                    onClick = { if (transferMinutes > 1) transferMinutes -= 2 },
                ) { Text("−") }
                Text(
                    "$transferMinutes min",
                    Modifier.padding(horizontal = 12.dp),
                    fontWeight = FontWeight.Bold,
                )
                OutlinedButton(onClick = { transferMinutes += 2 }) { Text("+") }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Deutschland-Ticket only", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        "Only regional trains (RE, RB, S, …) as connections",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(checked = deutschlandTicket, onCheckedChange = { deutschlandTicket = it })
            }
            if (deutschlandTicket && !DeutschlandTicket.covers(stop.label.category)) {
                Text(
                    "Note: ${stop.label.display} itself is not covered by the " +
                        "Deutschland-Ticket.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            Button(
                onClick = {
                    viewModel.evaluateConnection(
                        stop, transfer, destination, transferMinutes, deutschlandTicket,
                    )
                },
                enabled = transfer.isNotBlank() && destination.isNotBlank() &&
                    viewModel.connectionState != ConnectionState.Loading,
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Evaluate connection") }

            when (val state = viewModel.connectionState) {
                ConnectionState.Idle -> {}
                ConnectionState.Loading -> Box(
                    Modifier.fillMaxWidth().padding(24.dp),
                    contentAlignment = Alignment.Center,
                ) { CircularProgressIndicator() }
                is ConnectionState.Error -> Text(
                    state.message,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
                is ConnectionState.Loaded -> ConnectionResult(state.outcome)
            }
        }
    }
}

@Composable
private fun TransferPicker(
    stations: List<String>,
    selected: String,
    onSelect: (String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Box {
        OutlinedButton(onClick = { expanded = true }, modifier = Modifier.fillMaxWidth()) {
            Text(selected.ifBlank { "Pick transfer station" }, maxLines = 1)
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            stations.forEach { name ->
                DropdownMenuItem(
                    text = { Text(name) },
                    onClick = {
                        onSelect(name)
                        expanded = false
                    },
                )
            }
        }
    }
}

@Composable
private fun ConnectionResult(outcome: ConnectionPlanner.Outcome.Success) {
    val result = outcome.result
    val dist = result.distribution
    val median = dist.quantile(0.5)
    val q10 = dist.quantile(0.1)
    val q90 = dist.quantile(0.9)
    fun clock(delayMin: Double): String =
        formatTime(result.referenceArrivalMillis + (delayMin * 60_000).toLong())

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Surface(
            shape = MaterialTheme.shapes.large,
            color = MaterialTheme.colorScheme.primaryContainer,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(Modifier.padding(20.dp)) {
                Text(
                    "Predicted arrival at ${outcome.destinationName}",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
                Text(
                    clock(median),
                    style = MaterialTheme.typography.displayMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
                Text(
                    "80% between ${clock(q10)} and ${clock(q90)}, " +
                        "changing at ${outcome.transferStation.name}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            }
        }

        DelayDistributionChart(
            dist = dist,
            modifier = Modifier.fillMaxWidth().height(160.dp),
        )

        Text("Which train will you catch?", style = MaterialTheme.typography.titleSmall)
        result.candidates.forEach { cand ->
            Row(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Column(Modifier.weight(1f)) {
                    Text(cand.candidate.label, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        "dep ${formatTime(cand.candidate.plannedDepartureMillis)} → " +
                            "arr ${formatTime(cand.candidate.plannedArrivalMillis)}" +
                            (cand.candidate.liveDepartureDelay?.let {
                                if (it >= 1) "  (now +${it.roundToInt()})" else ""
                            } ?: ""),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    if (cand.candidate.cancelledLive) {
                        stringResource(R.string.cancelled)
                    } else {
                        "${(cand.boardProbability * 100).roundToInt()} %"
                    },
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = if (cand.candidate.cancelledLive) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.onSurface
                    },
                )
            }
            HorizontalDivider()
        }
        if (result.missProbability > 0.005) {
            Text(
                "P(miss all of the above): ${(result.missProbability * 100).roundToInt()} % — " +
                    "later trains are not listed.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            "Distributions are empirical; the feeder and connecting trains are " +
                "assumed independent, so shared disruptions make this optimistic.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
