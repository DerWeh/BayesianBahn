"""Tests for the second-tier station set.

The set decides which one-change journeys can ever be scored, so the errors
that matter are the silent ones: a station that resolves to the wrong eva (we
would then poll a different place for the rest of the study and never know), a
name the lookup cannot express, and a rule that could be nudged towards a
result. None of them raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import build_destinations as bd  # noqa: E402

PLAN = """<timetable station="Ulm Hbf" eva="8000170">
  <s id="trip-1"><tl c="RE" n="4230"/>
    <dp pt="2606101212" ppth="Neu-Ulm|Senden|Memmingen"/>
  </s>
  <s id="trip-2"><tl c="ICE" n="599"/>
    <ar pt="2606101230"/>
  </s>
  <s id="trip-3"><tl c="RB" n="17"/>
    <dp pt="2606101240" ppth="Neu-Ulm|Senden|Memmingen"/>
  </s>
  <s id="trip-4"><tl c="RB" n="19"/>
    <dp pt="2606101250" ppth="Langenau"/>
  </s>
</timetable>"""


def plan_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    out = tmp_path / "plan"
    out.mkdir()
    for name, body in files.items():
        (out / name).write_text(body, encoding="utf-8")
    return out


def test_the_terminus_is_the_last_stop_of_the_onward_path(tmp_path: Path) -> None:
    """Where the train ends up, not where it goes next: that is the far end of
    the second leg for a passenger changing onto it here."""
    counts, days = bd.terminus_counts(plan_dir(tmp_path, {"8000170-260610-12.xml": PLAN}))
    assert counts == {"Memmingen": 2, "Langenau": 1}
    assert days == {"260610"}


def test_a_stop_with_no_departure_is_not_a_terminus(tmp_path: Path) -> None:
    counts, _ = bd.terminus_counts(plan_dir(tmp_path, {"8000170-260610-12.xml": PLAN}))
    assert "trip-2" not in counts and sum(counts.values()) == 3


def test_a_half_written_cache_entry_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The collector writes plan documents through a temp file, but a cache
    directory copied mid-write can still hold one."""
    counts, days = bd.terminus_counts(plan_dir(tmp_path, {
        "8000170-260610-12.xml": PLAN,
        "8000170-260610-13.xml": "<timetable><s id=",
    }))
    assert counts == {"Memmingen": 2, "Langenau": 1}


def test_files_that_are_not_plan_documents_are_ignored(tmp_path: Path) -> None:
    counts, days = bd.terminus_counts(plan_dir(tmp_path, {
        "8000170-260610-12.xml": PLAN, "notes.xml": PLAN}))
    assert days == {"260610"} and sum(counts.values()) == 3


def test_the_floor_is_per_day_not_per_run() -> None:
    """Otherwise the set grows every time the collector runs longer, which
    would make the rule a function of how long we happened to collect."""
    counts = {"often": 40, "rare": 20}
    assert bd.frequent(counts, days=8, minimum=5) == ["often"]
    assert bd.frequent(counts, days=4, minimum=5) == ["often", "rare"]


def test_the_set_is_ordered_by_name_so_it_does_not_reshuffle() -> None:
    assert bd.frequent({"b": 10, "a": 10, "c": 10}, days=1, minimum=1) == ["a", "b", "c"]


def test_no_plan_documents_gives_no_stations() -> None:
    assert bd.frequent({"a": 10}, days=0) == []


def test_a_slash_in_the_name_is_asked_for_by_prefix() -> None:
    """IRIS routes on the raw path and answers 404 even for `%2F`, which is how
    `Köln Messe/Deutz` went missing the first time this ran."""
    assert bd.query_for("Köln Messe/Deutz") == "Köln Messe"
    assert bd.query_for("Hamburg-Neugraben") == "Hamburg-Neugraben"


STATIONS = """<stations>
  <station name="Neu-Ulm" eva="8000260" db="true"/>
  <station name="Neu-Ulm Nord" eva="8000261" db="true"/>
</stations>"""


def test_the_lookup_takes_the_exact_name_only() -> None:
    """The lookup is a prefix search. Accepting the first hit would have
    resolved `Neu-Ulm` to its neighbour and quietly polled the wrong station."""
    assert bd.lookup("Neu-Ulm", lambda url: STATIONS) == ("8000260", "Neu-Ulm")
    assert bd.lookup("Neu-Ul", lambda url: STATIONS) is None


def test_a_stop_with_no_timetable_is_dropped() -> None:
    """Replacement bus stops answer the lookup but have no forecast to read."""
    body = '<stations><station name="Bus X" eva="733523" db="false"/></stations>'
    assert bd.lookup("Bus X", lambda url: body) is None


def test_a_lookup_that_fails_drops_the_station_rather_than_the_run() -> None:
    def boom(url: str) -> str:
        raise OSError("no route to host")
    assert bd.lookup("Neu-Ulm", boom) is None


def test_the_committed_file_records_the_rule_it_came_from() -> None:
    """It is committed precisely so a later timetable cannot re-derive a
    different set and silently re-interpret data already collected."""
    text = (TOOLS / "forecast_destinations.csv").read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert any("departures a day" in line for line in header)
    assert any("never used as the origin" in line for line in header)
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert all(len(line.split(";")) == 3 for line in rows)


def test_the_rendered_file_reads_back_as_stations(tmp_path: Path) -> None:
    import collect_forecasts as cf
    path = tmp_path / "d.csv"
    path.write_text(bd.render([("8000260", "Neu-Ulm", 42)], ["260610"], 5),
                    encoding="utf-8")
    got = cf.load_stations(path, 2)
    assert [(s.eva, s.name, s.tier) for s in got] == [("8000260", "Neu-Ulm", 2)]
