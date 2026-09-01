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
import http.client
import os
import sys
import urllib.error
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


class _FakeResponse:
    """Enough of an HTTPResponse for `fetch`: a body, headers, a context."""

    def __init__(self, body: bytes, declared: int | None = None) -> None:
        self._body = body
        self.headers = {"Content-Length":
                        str(len(body) if declared is None else declared)}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_fetch_retries_a_truncated_shard(monkeypatch):
    """A transport failure returned as None is indistinguishable from a train
    with no history, so the train would be scored as one the model never saw."""
    attempts = []

    def flaky(request, timeout=0):
        attempts.append(1)
        if len(attempts) < 3:
            raise http.client.IncompleteRead(b"half", 99)
        return _FakeResponse(b"gzipped")

    monkeypatch.setattr(fs.urllib.request, "urlopen", flaky)
    assert fs.fetch("https://x/RE_1.jgz") == b"gzipped"
    assert len(attempts) == 3


def test_fetch_rejects_a_short_body(monkeypatch):
    """A body shorter than Content-Length is a truncation, not a shard."""
    monkeypatch.setattr(fs.urllib.request, "urlopen",
                        lambda request, timeout=0: _FakeResponse(b"half", declared=99))
    assert fs.fetch("https://x/RE_1.jgz") is None


def test_fetch_treats_404_as_no_history_without_retrying(monkeypatch):
    """404 is the one answer that means the train genuinely has no shard."""
    calls = []

    def missing(request, timeout=0):
        calls.append(1)
        raise urllib.error.HTTPError("https://x/RE_1.jgz", 404, "Not Found", {}, None)

    monkeypatch.setattr(fs.urllib.request, "urlopen", missing)
    assert fs.fetch("https://x/RE_1.jgz") is None
    assert len(calls) == 1


# --- which keys get fetched -------------------------------------------------
#
# The evaluation is only honest if it fetches the shards the app would hold.
# A key spelled differently here is not an error anywhere — the model simply
# scores those trains as ones it has never seen, which is exactly the state the
# line fallback exists to get out of.

APP = (
    TOOLS.parent
    / "app/src/main/java/io/github/derweh/bayesianbahn/data/HistoryRepository.kt"
)


def test_the_keys_are_spelled_as_the_app_spells_them():
    assert fs.train_key("ICE", "512") == "ICE_512"
    assert fs.line_key("S", "S7", "8089105") == "S7_8089105"
    assert fs.line_key("RE", "RE9", "8000013") == "RE9_8000013"
    # A replacement bus on the S7 is not an S7 train.
    assert fs.line_key("Bus", "S7", "8089105") == "BUS_S7_8089105"
    source = APP.read_text(encoding="utf-8")
    assert 'if (line.startsWith(category)) line else "$category $line"' in source
    assert '+ " " + stationEva' in source


def test_events_carry_the_station_a_line_shard_would_be_keyed_by(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"cat":"RE","num":"4711","line":"RE9","eva":"8000013",'
        '"candidates":[{"cat":"RB","num":"5000","line":"RB7"}]}\n',
        encoding="utf-8",
    )
    trains = fs.trains_scored([events])
    assert ("RE", "4711", "RE9", "8000013") in trains
    # The candidate gets its own shard and no line: CandidateBuilder pairs the
    # two legs of a run by date, which a line's pooled runs would scramble.
    assert ("RB", "5000", None, None) in trains


def test_the_line_is_read_off_the_train_s_own_shard(tmp_path):
    import gzip
    import json

    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "RE_4711.jgz").write_bytes(
        gzip.compress(json.dumps({"train": "RE 4711", "line": "RE9"}).encode()),
    )
    assert fs.shard_line(tmp_path, "RE_4711") == "RE9"
    assert fs.shard_line(tmp_path, "RE_9999") is None


def test_a_corrupt_shard_is_not_a_line(tmp_path):
    """A truncated cache entry must read as "no line", not crash the fetch."""
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "RE_4711.jgz").write_bytes(b"not gzip")
    assert fs.shard_line(tmp_path, "RE_4711") is None
