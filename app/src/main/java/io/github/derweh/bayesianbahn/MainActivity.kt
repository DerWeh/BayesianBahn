package io.github.derweh.bayesianbahn

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.derweh.bayesianbahn.ui.AppViewModel
import io.github.derweh.bayesianbahn.ui.BayesianBahnTheme
import io.github.derweh.bayesianbahn.ui.BoardScreen
import io.github.derweh.bayesianbahn.ui.ConnectionScreen
import io.github.derweh.bayesianbahn.ui.PredictionScreen
import io.github.derweh.bayesianbahn.ui.Route
import io.github.derweh.bayesianbahn.ui.SearchScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            BayesianBahnTheme {
                App()
            }
        }
    }
}

@Composable
private fun App(viewModel: AppViewModel = viewModel()) {
    BackHandler(enabled = viewModel.routes.size > 1) { viewModel.pop() }
    when (val route = viewModel.current) {
        Route.Search -> SearchScreen(viewModel, onStationSelected = viewModel::openBoard)
        is Route.Board -> BoardScreen(
            viewModel = viewModel,
            station = route.station,
            onStopSelected = { stop -> viewModel.openPrediction(route.station, stop) },
            onBack = { viewModel.pop() },
        )
        is Route.Prediction -> PredictionScreen(
            viewModel = viewModel,
            station = route.station,
            stop = route.stop,
            onBack = { viewModel.pop() },
        )
        is Route.Connection -> ConnectionScreen(
            viewModel = viewModel,
            station = route.station,
            stop = route.stop,
            onBack = { viewModel.pop() },
        )
    }
}
