package io.github.derweh.bayesianbahn.ui

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import io.github.derweh.bayesianbahn.data.Station

/**
 * Station input with live suggestions from the bundled offline station list,
 * the same source the station-board search uses.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StationSuggestField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    suggest: (String) -> List<Station>,
    modifier: Modifier = Modifier,
) {
    var expanded by remember { mutableStateOf(false) }
    val suggestions = remember(value) { if (value.length >= 2) suggest(value) else emptyList() }
    val open = expanded && suggestions.isNotEmpty() && suggestions.singleOrNull()?.name != value

    ExposedDropdownMenuBox(
        expanded = open,
        onExpandedChange = { expanded = it },
        modifier = modifier,
    ) {
        OutlinedTextField(
            value = value,
            onValueChange = {
                onValueChange(it)
                expanded = true
            },
            label = { Text(label) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth().menuAnchor(MenuAnchorType.PrimaryEditable),
        )
        ExposedDropdownMenu(
            expanded = open,
            onDismissRequest = { expanded = false },
        ) {
            suggestions.forEach { station ->
                DropdownMenuItem(
                    text = {
                        Text(station.name, style = MaterialTheme.typography.bodyMedium)
                    },
                    onClick = {
                        onValueChange(station.name)
                        expanded = false
                    },
                )
            }
        }
    }
}
