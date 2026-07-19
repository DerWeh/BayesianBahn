package io.github.derweh.bayesianbahn

import android.app.Application
import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.IrisParser
import io.github.derweh.bayesianbahn.data.CachedFetcher
import io.github.derweh.bayesianbahn.data.ConnectionPlanner
import io.github.derweh.bayesianbahn.data.DataUpdater
import io.github.derweh.bayesianbahn.data.HistoryRepository
import io.github.derweh.bayesianbahn.data.JourneyPlanner
import io.github.derweh.bayesianbahn.data.StationRepository
import io.github.derweh.bayesianbahn.data.SyntheticTimetable

/** Manual dependency container — the app is small enough to not need a DI framework. */
class BayesianBahnApp : Application() {
    val httpClient by lazy { okhttp3.OkHttpClient() }
    val stationRepository by lazy { StationRepository(this) }
    val historyRepository by lazy { HistoryRepository(this, httpClient) }
    val irisClient by lazy { IrisClient(IrisParser { android.util.Xml.newPullParser() }) }
    val dataUpdater by lazy { DataUpdater(this, httpClient) }
    val syntheticTimetable by lazy {
        SyntheticTimetable(CachedFetcher(this, httpClient), historyRepository)
    }
    val connectionPlanner by lazy {
        ConnectionPlanner(
            stationRepository, historyRepository, irisClient,
            syntheticTimetable = syntheticTimetable,
        )
    }
    val journeyPlanner by lazy {
        JourneyPlanner(
            stationRepository, historyRepository, irisClient, connectionPlanner,
            syntheticTimetable = syntheticTimetable,
        )
    }
}
