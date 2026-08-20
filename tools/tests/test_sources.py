"""Guards on the source files themselves, independent of what they say.

A stray NUL byte reached HistoryRepository.kt as a string literal — a cache
separator that should have been a visible character. The code was correct and
the tests passed, but git classifies a file containing a NUL as binary: it stops
producing diffs for it, review sees "Bin 9830 -> 10742 bytes" instead of the
change, and a three-way merge fails outright. The defect was invisible in every
view except a hex dump, which is exactly the kind a test should carry.
"""

from __future__ import annotations

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Extensions git should always be able to diff. Extensionless tracked files
# (LICENSE, .gitignore) are text too and are included by the empty string.
TEXT_SUFFIXES = {
    "", ".kt", ".kts", ".py", ".md", ".yml", ".yaml", ".toml", ".sh", ".txt",
    ".xml", ".json", ".csv", ".pro", ".properties", ".gradle", ".cfg",
}

# Tab, newline, carriage return are the control characters that belong in text.
ALLOWED_CONTROL = {0x09, 0x0A, 0x0D}


def tracked_text_files() -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout
    files = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        path = ROOT / raw.decode()
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files


def test_the_scan_actually_reaches_the_source():
    """A guard that silently matched nothing would pass forever."""
    names = {p.name for p in tracked_text_files()}
    assert "HistoryRepository.kt" in names
    assert "report.py" in names


def test_no_tracked_source_file_contains_a_control_character():
    offenders = []
    for path in tracked_text_files():
        data = path.read_bytes()
        for offset, byte in enumerate(data):
            if byte < 0x20 and byte not in ALLOWED_CONTROL:
                line = data[:offset].count(b"\n") + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line} has byte 0x{byte:02x}")
                break
    assert not offenders, "git treats these as binary and stops diffing them:\n" + \
        "\n".join(offenders)
