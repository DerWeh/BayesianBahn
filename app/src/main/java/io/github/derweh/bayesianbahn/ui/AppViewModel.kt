package io.github.derweh.bayesianbahn.ui

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.derweh.bayesianbahn.BayesianBahnApp
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.data.Forecast
import io.github.derweh.bayesianbahn.data.Predictor
import io.github.derweh.bayesianbahn.data.Station
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

sealed interface Route {
    data object Search : Route
    data class Board(val station: Station) : Route
    data class Prediction(val station: Station, val stop: TimetableStop) : Route
}

sealed interface BoardState {
    data object Loading : BoardState
    data class Error(val message: String) : BoardState
    data class Loaded(val stops: List<TimetableStop>) : BoardState
}

sealed interface PredictionState {
    data object Loading : PredictionState
    data class Loaded(val forecast: Forecast) : PredictionState
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    private val container get() = getApplication<BayesianBahnApp>()
    private val predictor = Predictor()

    var routes by mutableStateOf(listOf<Route>(Route.Search))
        private set

    val current: Route get() = routes.last()

    fun pop(): Boolean {
        if (routes.size <= 1) return false
        routes = routes.dropLast(1)
        return true
    }

    // ---- station search ----
    var query by mutableStateOf("")
        private set
    var searchResults by mutableStateOf(listOf<Station>())
        private set
    private var searchJob: Job? = null

    fun onQueryChange(value: String) {
        query = value
        searchJob?.cancel()
        searchJob = viewModelScope.launch {
            searchResults = container.stationRepository.search(value)
        }
    }

    // ---- board ----
    var boardState by mutableStateOf<BoardState>(BoardState.Loading)
        private set

    fun openBoard(station: Station) {
        routes = routes + Route.Board(station)
        refreshBoard()
    }

    fun refreshBoard() {
        val station = (current as? Route.Board)?.station
            ?: (routes.filterIsInstance<Route.Board>().lastOrNull())?.station
            ?: return
        boardState = BoardState.Loading
        viewModelScope.launch {
            boardState = try {
                BoardState.Loaded(container.irisClient.board(station.eva, hours = 3))
            } catch (e: Exception) {
                BoardState.Error(e.message ?: "network error")
            }
        }
    }

    // ---- prediction ----
    var predictionState by mutableStateOf<PredictionState>(PredictionState.Loading)
        private set

    fun openPrediction(station: Station, stop: TimetableStop) {
        routes = routes + Route.Prediction(station, stop)
        predictionState = PredictionState.Loading
        viewModelScope.launch {
            val history = container.historyRepository.load(
                category = stop.label.category,
                number = stop.label.number,
                line = stop.label.line,
            )
            val event = stop.arrival ?: stop.departure
            val forecast = predictor.forecast(
                history = history,
                stationEva = station.eva,
                stationName = station.name,
                trainCategory = stop.label.category,
                plannedTimeMillis = event?.plannedTime ?: System.currentTimeMillis(),
                liveDelayMinutes = event?.liveDelayMinutes,
            )
            predictionState = PredictionState.Loaded(forecast)
        }
    }
}
