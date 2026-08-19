package io.github.derweh.bayesianbahn

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.derweh.bayesianbahn.ui.AppViewModel
import io.github.derweh.bayesianbahn.ui.BayesianBahnTheme
import io.github.derweh.bayesianbahn.ui.BoardScreen
import io.github.derweh.bayesianbahn.ui.ConnectionScreen
import io.github.derweh.bayesianbahn.ui.JourneyScreen
import io.github.derweh.bayesianbahn.ui.PredictionScreen
import io.github.derweh.bayesianbahn.ui.Route
import io.github.derweh.bayesianbahn.ui.SearchScreen
import io.github.derweh.bayesianbahn.ui.staleKeys

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
    // Only the current route is composed, so every other screen leaves the
    // composition and loses what it held in rememberSaveable. That is what
    // emptied From and To whenever the user opened a station board and came
    // back — with the results still on screen, and the button disabled because
    // the fields behind it were now blank.
    val stateHolder = rememberSaveableStateHolder()
    val known = remember { mutableSetOf<String>() }
    LaunchedEffect(viewModel.routes) {
        staleKeys(known, viewModel.routes).forEach { stateHolder.removeState(it) }
        known.clear()
        viewModel.routes.mapTo(known) { it.key }
    }
    stateHolder.SaveableStateProvider(viewModel.current.key) {
        Screen(viewModel)
    }
}

@Composable
private fun Screen(viewModel: AppViewModel) {
    when (val route = viewModel.current) {
        Route.Journey -> JourneyScreen(viewModel)
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
