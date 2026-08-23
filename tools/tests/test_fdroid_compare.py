"""Tests for the relation between our metadata mirror and fdroiddata's copy.

The check this replaces was a byte-diff against our fork branch. That was right
while the merge request was open and wrong the moment it merged: AutoUpdateMode
put F-Droid's bot in charge of fdroiddata master, and the fork stopped moving.
The check failed for ever afterwards, which is the failure mode that trains
people to ignore a red pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import fdroid_compare as fc  # noqa: E402

WHAT = "fdroiddata master"


def meta(*entries, current=None):
    builds = [{"versionName": n, "versionCode": c, "commit": h} for n, c, h in entries]
    out = {"Builds": builds}
    if current is not None:
        out["CurrentVersionCode"] = current
    return out


A = ("0.1.2", 3, "a" * 40)
B = ("0.1.3", 4, "b" * 40)
C = ("0.1.4", 5, "c" * 40)


def test_identical_files_agree():
    problems, notes = fc.compare(meta(A, B, current=4), meta(A, B, current=4), WHAT)
    assert problems == [] and notes == []


def test_being_ahead_is_a_note_not_a_failure():
    """A release is prepared here before its tag is pushed; the bot cannot know
    about it yet, and saying so must not fail the build."""
    problems, notes = fc.compare(meta(A, B, C, current=5), meta(A, B, current=4), WHAT)
    assert problems == []
    assert notes == ["prepared here but not yet published: 0.1.4"]


def test_being_behind_is_a_failure():
    """If fdroiddata builds something this file does not describe, every drift
    guard here is looking at the wrong thing."""
    problems, _ = fc.compare(meta(A, current=3), meta(A, B, current=4), WHAT)
    assert any("builds 0.1.3" in p for p in problems)


def test_a_shared_entry_that_differs_is_a_failure():
    theirs = meta(A, ("0.1.3", 4, "d" * 40), current=4)
    problems, _ = fc.compare(meta(A, B, current=4), theirs, WHAT)
    assert len(problems) == 1 and "0.1.3" in problems[0]


def test_a_differing_version_code_for_the_same_name_is_caught():
    problems, _ = fc.compare(meta(("0.1.3", 4, "b" * 40), current=4),
                             meta(("0.1.3", 9, "b" * 40), current=4), WHAT)
    assert problems and "0.1.3" in problems[0]


def test_a_published_current_version_ahead_of_ours_is_a_failure():
    problems, _ = fc.compare(meta(A, B, current=3), meta(A, B, current=4), WHAT)
    assert any("ahead of this file" in p for p in problems)


def test_our_current_version_ahead_of_published_is_fine():
    problems, _ = fc.compare(meta(A, B, C, current=5), meta(A, B, current=4), WHAT)
    assert problems == []


def test_a_missing_builds_key_does_not_crash():
    problems, notes = fc.compare({}, {}, WHAT)
    assert problems == [] and notes == []


def test_the_message_names_which_side_is_which():
    problems, _ = fc.compare(meta(A, current=3), meta(A, B, current=4), "the fork branch")
    assert all("the fork branch" in p for p in problems)
