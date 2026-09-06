"""The app and the study must carry the same numbers.

`tools/anchor-model.json` is what `sensitivity_live.py freeze` fitted and what
the report quotes; `AnchoredDelay.kt` is what the app ships. Nothing enforces
that they agree except this file, and the failure it guards against is silent:
a refit that nobody ports, or a hand-tweaked constant, leaves a study measuring
a model the app does not implement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "tools/anchor-model.json"
KOTLIN = ROOT / "app/src/main/java/io/github/derweh/bayesianbahn/model/AnchoredDelay.kt"

# The Kotlin name for each residual in the frozen file.
SHAPES = {"arrival": "ARRIVAL",
          "departure_reported": "DEPARTURE_REPORTED",
          "departure_silent": "DEPARTURE_SILENT"}
FIELDS = ("width_intercept", "width_per_minute", "centre_intercept",
          "centre_per_sqrt_minute", "left", "right")


def frozen() -> dict:
    return json.loads(MODEL.read_text())["residuals"]


def shipped() -> dict:
    """The six numbers of each `ResidualShape` constant, in declared order."""
    text = KOTLIN.read_text()
    out = {}
    for name in SHAPES.values():
        m = re.search(rf"val {name} = ResidualShape\(([^)]*)\)", text, re.S)
        assert m, f"no ResidualShape constant named {name}"
        numbers = [float(v) for v in re.findall(r"-?\d+\.\d+(?:[eE]-?\d+)?", m.group(1))]
        assert len(numbers) == len(FIELDS), \
            f"{name} has {len(numbers)} numbers, expected {len(FIELDS)}"
        out[name] = dict(zip(FIELDS, numbers))
    return out


@pytest.mark.parametrize("residual", sorted(SHAPES))
def test_the_app_ships_the_numbers_that_were_fitted(residual):
    want = frozen()[residual]
    got = shipped()[SHAPES[residual]]
    for field in FIELDS:
        assert got[field] == pytest.approx(want[field], abs=5e-4), \
            (f"{residual}.{field}: the app says {got[field]}, "
             f"anchor-model.json says {want[field]}")


def test_every_frozen_residual_has_somewhere_to_live_in_the_app():
    assert set(frozen()) == set(SHAPES), \
        "a residual was added to the frozen model with no Kotlin constant"


def test_the_silent_departure_really_has_no_lead_term():
    """Not a rounding artefact: it is the finding, so it is pinned.

    With no report there is no information to decay, and the width is flat from
    a quarter of an hour out to beyond ninety minutes. If a refit gives this a
    slope, the fit changed and the comment in the app is no longer true.
    """
    silent = frozen()["departure_silent"]
    assert silent["width_per_minute"] == 0.0
    assert silent["centre_per_sqrt_minute"] == 0.0
    assert silent["centre_intercept"] > 0.0, "silence is not on time"


def test_the_right_tail_is_the_long_one_everywhere():
    """Trains lose more time than they win back, arriving and departing."""
    for name, r in frozen().items():
        assert r["right"] > r["left"], f"{name} leans the wrong way"


# --- the harnesses must ask at the time the question was asked ---------------

HARNESSES = {
    "ForecastHarness": ROOT / "app/src/test/java/io/github/derweh/bayesianbahn/ForecastHarness.kt",
    "JourneyHarness": ROOT / "app/src/test/java/io/github/derweh/bayesianbahn/JourneyHarness.kt",
}


@pytest.mark.parametrize("name", sorted(HARNESSES))
def test_the_harness_pins_now_to_when_the_event_was_read(name):
    """Otherwise every scored event is at a lead of zero.

    `Predictor.forecast` and `ConnectionModel.propagate` both default
    `nowMillis` to the wall clock, which is right in the app and silently wrong
    in a harness replaying last week. The residual is a function of the lead, so
    the default would score the sharpest model the parameters can describe and
    report it as the app's. Nothing else would look broken.
    """
    text = HARNESSES[name].read_text()
    assert "nowMillis" in text, f"{name} never pins nowMillis"
    for call in re.finditer(r"nowMillis = ([^\n,]+)", text):
        assert "read_at" in call.group(1), \
            f"{name} pins nowMillis to {call.group(1)!r}, not to the read time"


def test_every_call_that_takes_a_lead_gets_one_from_the_harness():
    """Both entry points, not just whichever was remembered."""
    text = HARNESSES["JourneyHarness"].read_text()
    for entry in ("predictor.forecast(", "ConnectionModel.propagate("):
        start = text.index(entry)
        call = text[start:text.index("\n                    )", start)]
        assert "nowMillis" in call, f"{entry} in JourneyHarness has no nowMillis"
