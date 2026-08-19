package io.github.derweh.bayesianbahn.ui

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.derweh.bayesianbahn.BayesianBahnApp
import io.github.derweh.bayesianbahn.api.TimetableStop
import io.github.derweh.bayesianbahn.data.ConnectionPlanner
import io.github.derweh.bayesianbahn.data.DataMeta
import io.github.derweh.bayesianbahn.data.DataUpdater
import io.github.derweh.bayesianbahn.data.Forecast
import io.github.derweh.bayesianbahn.data.JourneyPlanner
import io.github.derweh.bayesianbahn.data.Predictor
import io.github.derweh.bayesianbahn.data.Station
import io.github.derweh.bayesianbahn.data.UserMessages
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

sealed interface Route {
    /**
     * Identity of this screen's retained UI state.
     *
     * Only the current route is composed, so a screen navigated away from is
     * removed from the composition and everything it held in `rememberSaveable`
     * — typed stations, the chosen date, scroll positions — goes with it. The
     * state holder in `MainActivity` keys off this to hand that state back when
     * the screen returns.
     *
     * Two visits to the same station are the same screen and share their state;
     * two different stations must not.
     */
    val key: String

    /** Home: from/to journey search. */
    data object Journey : Route {
        override val key get() = "journey"
    }

    /** Station search list for browsing live boards. */
    data object Search : Route {
        override val key get() = "search"
    }

    data class Board(val station: Station) : Route {
        override val key get() = "board/${station.eva}"
    }

    data class Prediction(val station: Station, val stop: TimetableStop) : Route {
        override val key get() = "prediction/${station.eva}/${stop.id}"
    }

    data class Connection(val station: Station, val stop: TimetableStop) : Route {
        override val key get() = "connection/${station.eva}/${stop.id}"
    }
}

/**
 * Keys whose screen has left the stack, so the state held for them can go.
 *
 * Without this the holder would keep the state of every screen the session ever
 * opened — every station board, every prediction — for as long as the app runs.
 */
fun staleKeys(known: Set<String>, stack: List<Route>): Set<String> =
    known - stack.mapTo(mutableSetOf()) { it.key }

sealed interface BoardState {
    data object Loading : BoardState
    data class Error(val message: String) : BoardState
    data class Loaded(val stops: List<TimetableStop>) : BoardState
}

sealed interface PredictionState {
    data object Loading : PredictionState
    data class Loaded(val forecast: Forecast) : PredictionState
}

sealed interface ConnectionState {
    /** Waiting for the user to pick transfer and destination. */
    data object Idle : ConnectionState
    data object Loading : ConnectionState
    data class Error(val message: String) : ConnectionState
    data class Loaded(val outcome: ConnectionPlanner.Outcome.Success) : ConnectionState
}

sealed interface JourneyState {
    data object Idle : JourneyState
    data object Loading : JourneyState
    data class Error(val message: String) : JourneyState
    data class Loaded(val outcome: JourneyPlanner.Outcome.Success) : JourneyState
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    private val container get() = getApplication<BayesianBahnApp>()
    private val predictor = Predictor()

    var routes by mutableStateOf(listOf<Route>(Route.Journey))
        private set

    // ---- journey search ----
    var journeyState by mutableStateOf<JourneyState>(JourneyState.Idle)
        private set

    fun openStationSearch() {
        routes = routes + Route.Search
    }

    /** Synchronous suggestions for station input fields (in-memory list). */
    fun suggestStations(query: String): List<Station> =
        container.stationRepository.search(query, limit = 6)

    fun planJourney(
        fromQuery: String,
        toQuery: String,
        departMillis: Long,
        deutschlandTicketOnly: Boolean,
    ) {
        journeyState = JourneyState.Loading
        viewModelScope.launch {
            journeyState = when (
                val outcome = container.journeyPlanner.plan(
                    fromQuery, toQuery, departMillis, deutschlandTicketOnly,
                )
            ) {
                is JourneyPlanner.Outcome.Error -> JourneyState.Error(outcome.message)
                is JourneyPlanner.Outcome.Success -> JourneyState.Loaded(outcome)
            }
        }
    }

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
                BoardState.Error(UserMessages.TIMETABLE_UNREACHABLE)
            }
        }
    }

    // ---- prediction ----
    var predictionState by mutableStateOf<PredictionState>(PredictionState.Loading)
        private set

    // ---- connection ----
    var connectionState by mutableStateOf<ConnectionState>(ConnectionState.Idle)
        private set

    fun openConnection(station: Station, stop: TimetableStop) {
        routes = routes + Route.Connection(station, stop)
        connectionState = ConnectionState.Idle
    }

    fun evaluateConnection(
        stop: TimetableStop,
        transferName: String,
        destinationQuery: String,
        transferMinutes: Int,
        deutschlandTicketOnly: Boolean,
    ) {
        connectionState = ConnectionState.Loading
        viewModelScope.launch {
            connectionState = when (
                val outcome = container.connectionPlanner.plan(
                    feeder = stop,
                    transferQuery = transferName,
                    destinationQuery = destinationQuery,
                    transferMinutes = transferMinutes,
                    deutschlandTicketOnly = deutschlandTicketOnly,
                )
            ) {
                is ConnectionPlanner.Outcome.Error -> ConnectionState.Error(outcome.message)
                is ConnectionPlanner.Outcome.Success -> ConnectionState.Loaded(outcome)
            }
        }
    }

    // ---- history data updates ----
    var dataMeta by mutableStateOf<DataMeta?>(null)
        private set
    var dataUpdating by mutableStateOf(false)
        private set
    var dataUpdateError by mutableStateOf<String?>(null)
        private set

    init {
        viewModelScope.launch(Dispatchers.IO) {
            dataMeta = DataUpdater.readMeta(getApplication())
            // Prewarm the station list so field suggestions never block the UI.
            container.stationRepository.search("")
        }
    }

    fun updateData() {
        if (dataUpdating) return
        dataUpdating = true
        dataUpdateError = null
        viewModelScope.launch {
            container.dataUpdater.update()
                .onSuccess {
                    dataMeta = it
                    // The shards on disk have changed; anything already
                    // remembered in memory is now the previous release.
                    container.historyRepository.invalidate()
                }
                .onFailure { dataUpdateError = it.message ?: "update failed" }
            dataUpdating = false
        }
    }

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
