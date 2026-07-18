package io.github.derweh.bayesianbahn

import android.app.Application
import io.github.derweh.bayesianbahn.api.IrisClient
import io.github.derweh.bayesianbahn.api.IrisParser
import io.github.derweh.bayesianbahn.data.HistoryRepository
import io.github.derweh.bayesianbahn.data.StationRepository

/** Manual dependency container — the app is small enough to not need a DI framework. */
class BayesianBahnApp : Application() {
    val stationRepository by lazy { StationRepository(this) }
    val historyRepository by lazy { HistoryRepository(this) }
    val irisClient by lazy { IrisClient(IrisParser { android.util.Xml.newPullParser() }) }
}
