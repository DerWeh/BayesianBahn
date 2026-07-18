package io.github.derweh.bayesianbahn.model

/**
 * Deutschland-Ticket validity: the ticket covers all local and regional
 * public transport (RE, RB, IRE, S-Bahn and the private regional operators)
 * but no long-distance trains (ICE/IC/EC, night trains, FlixTrain, …) —
 * which is exactly the [TrainClass.LONG_DISTANCE] boundary.
 *
 * Approximations: the handful of IC segments released for regional tickets
 * (e.g. Bremen–Norddeich) count as not covered, and validity ends at the
 * border on cross-border regional trains.
 */
object DeutschlandTicket {
    fun covers(category: String): Boolean =
        TrainClass.fromCategory(category) != TrainClass.LONG_DISTANCE
}
