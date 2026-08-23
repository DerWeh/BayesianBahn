package io.github.derweh.bayesianbahn.ui

import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import io.github.derweh.bayesianbahn.R
import io.github.derweh.bayesianbahn.data.UserMessage

/** The user's language applied to a [UserMessage] raised without one. */
@Composable
fun UserMessage.text(): String = when (this) {
    UserMessage.TimetableUnreachable ->
        stringResource(R.string.error_timetable_unreachable)
    UserMessage.TimetableUnreachableNoHistory ->
        stringResource(R.string.error_timetable_unreachable_no_history)
    UserMessage.SameOriginAndDestination ->
        stringResource(R.string.error_same_origin_and_destination)
    is UserMessage.UpdateFailed ->
        detail ?: stringResource(R.string.error_update_failed)
    is UserMessage.StationNotFound ->
        stringResource(R.string.error_station_not_found, query)
    is UserMessage.TransferStationNotFound ->
        stringResource(R.string.error_transfer_station_not_found, query)
    is UserMessage.NoTimetableData ->
        stringResource(R.string.error_no_timetable_data, station)
    is UserMessage.NoConnection -> stringResource(
        R.string.error_no_connection,
        from,
        to,
        stringResource(R.string.error_one_change_only),
    )
    is UserMessage.FeederDoesNotReach ->
        stringResource(R.string.error_feeder_does_not_reach, train, station)
    is UserMessage.NoPlannedArrival ->
        stringResource(R.string.error_no_planned_arrival, station)
    is UserMessage.NoTrainsTowards -> stringResource(
        if (deutschlandTicketOnly) {
            R.string.error_no_ticket_trains_towards
        } else {
            R.string.error_no_trains_towards
        },
        destination,
        station,
    )
    is UserMessage.NotEnoughHistory ->
        stringResource(R.string.error_not_enough_history, destination)
}
