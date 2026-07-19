package io.github.derweh.bayesianbahn.model

import java.time.DayOfWeek
import java.time.LocalDate
import java.time.Month
import java.time.temporal.TemporalAdjusters

/**
 * German calendar facts the synthetic timetable needs: nationwide public
 * holidays (trains run the Sunday timetable) and the biannual timetable
 * change dates. Regional holidays (Fronleichnam, Allerheiligen, …) are not
 * modelled — station positions don't carry the Bundesland.
 */
object GermanCalendar {

    /** Nationwide public holidays; trains run the Sunday timetable. */
    fun isNationwideHoliday(date: LocalDate): Boolean {
        if (date.month == Month.JANUARY && date.dayOfMonth == 1) return true
        if (date.month == Month.MAY && date.dayOfMonth == 1) return true
        if (date.month == Month.OCTOBER && date.dayOfMonth == 3) return true
        if (date.month == Month.DECEMBER && date.dayOfMonth in 25..26) return true
        val easter = easterSunday(date.year)
        return date == easter.minusDays(2) || // Good Friday
            date == easter.plusDays(1) ||     // Easter Monday
            date == easter.plusDays(39) ||    // Ascension
            date == easter.plusDays(50)       // Whit Monday
    }

    /**
     * Next timetable change strictly after [from]: the major change on the
     * second Sunday of December and the minor one mid-June.
     */
    fun nextTimetableChange(from: LocalDate): LocalDate {
        val candidates = (from.year..from.year + 1).flatMap { year ->
            listOf(secondSunday(year, Month.JUNE), secondSunday(year, Month.DECEMBER))
        }
        return candidates.first { it.isAfter(from) }
    }

    private fun secondSunday(year: Int, month: Month): LocalDate =
        LocalDate.of(year, month, 1)
            .with(TemporalAdjusters.dayOfWeekInMonth(2, DayOfWeek.SUNDAY))

    /** Anonymous Gregorian computus. */
    fun easterSunday(year: Int): LocalDate {
        val a = year % 19
        val b = year / 100
        val c = year % 100
        val d = b / 4
        val e = b % 4
        val f = (b + 8) / 25
        val g = (b - f + 1) / 3
        val h = (19 * a + b - d - g + 15) % 30
        val i = c / 4
        val k = c % 4
        val l = (32 + 2 * e + 2 * i - h - k) % 7
        val m = (a + 11 * h + 22 * l) / 451
        val month = (h + l - 7 * m + 114) / 31
        val day = (h + l - 7 * m + 114) % 31 + 1
        return LocalDate.of(year, month, day)
    }
}
