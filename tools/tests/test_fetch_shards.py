"""Tests for the shard cache's freshness rule and its concurrency.

Two defects motivated these. Fetching was strictly sequential — one request at a
time against a 275 ms round trip, which for a day's ~4,000 trains is 43 minutes
and was most of what an evaluation cost. And the cache kept a *recent* overlay
for ever, so an overlay downloaded on the 19th was still being used to score the
21st, starving the model of the two days of history a real user would have had.
Nothing caught the second one: the harness asserts there is no leak, and a stale
overlay is short of data rather than ahead of it, so it passes.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import fetch_shards as fs  # noqa: E402


def shard(tmp_path: Path, tier: str, name: str, *, fetched: dt.date) -> Path:
    target = tmp_path / tier / f"{name}.jgz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"gz")
    stamp = dt.datetime.combine(fetched, dt.time(12, 0)).timestamp()
    os.utime(target, (stamp, stamp))
    return target


DAY = dt.date(2026, 8, 21)


def test_a_missing_shard_is_never_fresh(tmp_path):
    assert not fs.is_fresh(tmp_path / "recent" / "RE_1.jgz", "recent", DAY)


def test_a_cached_base_shard_stays_fresh(tmp_path):
    old = shard(tmp_path, "base", "RE_1", fetched=dt.date(2026, 1, 1))
    assert fs.is_fresh(old, "base", DAY)


def test_a_recent_overlay_older_than_the_evaluated_day_is_refetched(tmp_path):
    stale = shard(tmp_path, "recent", "RE_1", fetched=dt.date(2026, 8, 19))
    assert not fs.is_fresh(stale, "recent", DAY)


def test_a_recent_overlay_from_the_evaluated_day_is_kept(tmp_path):
    same = shard(tmp_path, "recent", "RE_1", fetched=DAY)
    assert fs.is_fresh(same, "recent", DAY)


def test_a_recent_overlay_fetched_later_is_kept(tmp_path):
    """A later overlay carries runs from after the evaluated day, which the
    harness trims away; being ahead is safe, being behind is not."""
    later = shard(tmp_path, "recent", "RE_1", fetched=dt.date(2026, 8, 30))
    assert fs.is_fresh(later, "recent", DAY)


def test_a_cached_key_costs_no_request(tmp_path, monkeypatch):
    for tier in fs.BRANCHES:
        shard(tmp_path, tier, "RE_1", fetched=DAY)
    monkeypatch.setattr(fs, "fetch", lambda url: pytest.fail(f"refetched {url}"))
    assert set(fs.fetch_key("RE_1", tmp_path, DAY)) == set(fs.BRANCHES)


def test_a_stale_overlay_costs_exactly_one_request(tmp_path, monkeypatch):
    shard(tmp_path, "base", "RE_1", fetched=dt.date(2026, 1, 1))
    shard(tmp_path, "recent", "RE_1", fetched=dt.date(2026, 8, 19))
    asked = []
    monkeypatch.setattr(fs, "fetch", lambda url: asked.append(url) or b"fresh")
    monkeypatch.setattr(fs, "PAUSE", 0)
    fs.fetch_key("RE_1", tmp_path, DAY)
    assert len(asked) == 1 and "shards-recent" in asked[0]
    assert (tmp_path / "recent" / "RE_1.jgz").read_bytes() == b"fresh"


def test_a_key_with_no_shard_anywhere_reports_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "fetch", lambda url: None)
    monkeypatch.setattr(fs, "PAUSE", 0)
    assert fs.fetch_key("RE_404", tmp_path, DAY) == ()


def test_an_interrupted_write_leaves_no_shard_behind(tmp_path, monkeypatch):
    """Temp-then-rename: a truncated file must never look like a complete one."""
    def explode(url):
        raise KeyboardInterrupt
    monkeypatch.setattr(fs, "fetch", explode)
    with pytest.raises(KeyboardInterrupt):
        fs.fetch_key("RE_1", tmp_path, DAY)
    assert not list(tmp_path.rglob("*.jgz"))
    assert not list(tmp_path.rglob("*.part"))


def test_the_temp_name_is_unique_per_thread(tmp_path, monkeypatch):
    """Two threads fetching different keys must not collide on a temp path; the
    name carries the pid and thread id for that reason."""
    source = (TOOLS / "fetch_shards.py").read_text(encoding="utf-8")
    assert "os.getpid()" in source and "threading.get_ident()" in source


def test_fetching_is_concurrent_by_default():
    assert fs.WORKERS > 1
