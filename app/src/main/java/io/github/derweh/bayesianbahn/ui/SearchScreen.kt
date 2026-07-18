package io.github.derweh.bayesianbahn.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import io.github.derweh.bayesianbahn.R
import io.github.derweh.bayesianbahn.data.Station

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchScreen(viewModel: AppViewModel, onStationSelected: (Station) -> Unit) {
    LaunchedEffect(Unit) {
        if (viewModel.searchResults.isEmpty()) viewModel.onQueryChange(viewModel.query)
    }
    Scaffold(
        topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            OutlinedTextField(
                value = viewModel.query,
                onValueChange = viewModel::onQueryChange,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                placeholder = { Text(stringResource(R.string.search_hint)) },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
                singleLine = true,
            )
            LazyColumn(Modifier.weight(1f)) {
                items(viewModel.searchResults, key = { it.eva }) { station ->
                    ListItem(
                        headlineContent = { Text(station.name) },
                        supportingContent = {
                            Text("EVA ${station.eva}", style = MaterialTheme.typography.bodySmall)
                        },
                        modifier = Modifier.fillMaxWidth()
                            .clickable { onStationSelected(station) },
                    )
                    HorizontalDivider()
                }
            }
            DataStatusRow(viewModel)
        }
    }
}

/** Age of the bundled/downloaded delay history, with a manual update action. */
@Composable
private fun DataStatusRow(viewModel: AppViewModel) {
    val meta = viewModel.dataMeta
    ListItem(
        headlineContent = {
            Text(
                when {
                    meta?.recentThrough != null -> "Delay history through ${meta.recentThrough}"
                    meta?.baseGenerated != null -> "Delay history: ${meta.baseGenerated}"
                    else -> "Delay history: bundled"
                },
                style = MaterialTheme.typography.bodySmall,
            )
        },
        supportingContent = {
            Text(
                viewModel.dataUpdateError
                    ?: buildString {
                        meta?.trains?.let { append("$it trains") }
                        if (meta?.updated == true) append("  ·  downloaded")
                    },
                style = MaterialTheme.typography.bodySmall,
                color = if (viewModel.dataUpdateError != null) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
        },
        trailingContent = {
            if (viewModel.dataUpdating) {
                CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
            } else {
                IconButton(onClick = viewModel::updateData) {
                    Icon(
                        Icons.Default.Refresh,
                        contentDescription = stringResource(R.string.update_data),
                    )
                }
            }
        },
    )
}
