"""Length limits on the F-Droid / fastlane store metadata.

The 0.2.0 changelog was 519 characters and F-Droid displayed a truncated
version of it: the release's headline claim was cut off mid-sentence on the
website and in the client. Nothing in the build noticed, because the file is
data that no code reads — it is handed to F-Droid as-is, and only a reader
sees the damage.

The limits are F-Droid's own, from `All About Descriptions, Graphics and
Screenshots`; they are counted in *characters*, so a German umlaut costs one
here and two on disk. Measuring bytes would reject a legal German file.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
METADATA = ROOT / "fastlane" / "metadata" / "android"

# https://f-droid.org/docs/All_About_Descriptions_Graphics_and_Screenshots/
LIMITS = {
    "title.txt": 50,
    "short_description.txt": 80,
    "full_description.txt": 4000,
}
CHANGELOG_LIMIT = 500


def locales() -> list[pathlib.Path]:
    return sorted(p for p in METADATA.iterdir() if p.is_dir())


def metadata_files() -> list[tuple[pathlib.Path, int]]:
    found = []
    for locale in locales():
        for name, limit in LIMITS.items():
            path = locale / name
            if path.exists():
                found.append((path, limit))
        for path in sorted((locale / "changelogs").glob("*.txt")):
            found.append((path, CHANGELOG_LIMIT))
    return found


def test_locales_exist() -> None:
    """A typo in the metadata path would make every other test vacuous."""
    assert [p.name for p in locales()] == ["de-DE", "en-US"]


def test_changelogs_are_found() -> None:
    """Likewise: globbing nothing would pass the length test silently."""
    changelogs = [p for p, limit in metadata_files() if limit == CHANGELOG_LIMIT]
    assert len(changelogs) >= 7


@pytest.mark.parametrize(
    ("path", "limit"),
    metadata_files(),
    ids=lambda v: str(v.relative_to(METADATA)) if isinstance(v, pathlib.Path) else "",
)
def test_within_limit(path: pathlib.Path, limit: int) -> None:
    text = path.read_text(encoding="utf-8").rstrip()
    assert len(text) <= limit, (
        f"{path.relative_to(ROOT)} is {len(text)} characters, over the {limit} "
        f"F-Droid allows — it will be shown truncated"
    )


@pytest.mark.parametrize("locale", locales(), ids=lambda p: p.name)
def test_changelog_matches_a_version_code(locale: pathlib.Path) -> None:
    """`<versionCode>.txt` with no padding, or F-Droid shows nothing at all."""
    gradle = (ROOT / "app" / "build.gradle.kts").read_text(encoding="utf-8")
    current = int(
        next(l for l in gradle.splitlines() if "versionCode" in l).split("=")[1]
    )
    for path in (locale / "changelogs").glob("*.txt"):
        assert path.stem.isdigit(), f"{path.name} is not a version code"
        assert path.stem == str(int(path.stem)), f"{path.name} is zero-padded"
        # current + 1 is the release being prepared: the changelog is written
        # before the bump, and fdroid-check is what refuses to let the two
        # halves ship apart. Anything beyond that matches no version F-Droid
        # will ever look for, so it would silently display nothing.
        assert int(path.stem) <= current + 1, (
            f"{path.name} is ahead of versionCode {current} in build.gradle.kts"
        )


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(path: pathlib.Path) -> tuple[int, int]:
    """Width and height out of the IHDR header — no rasteriser needed."""
    header = path.read_bytes()[:24]
    assert header[:8] == PNG_MAGIC, f"{path.name} is not a PNG"
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def shots(locale: pathlib.Path) -> list[pathlib.Path]:
    return sorted((locale / "images" / "phoneScreenshots").glob("*.png"))


@pytest.mark.parametrize("locale", locales(), ids=lambda p: p.name)
def test_locale_has_its_own_screenshots(locale: pathlib.Path) -> None:
    """de-DE shipped 0.2.0 with none, so German users were shown English ones.

    F-Droid falls back to the default locale rather than showing nothing, which
    is why it went unnoticed: the listing looked complete in both languages.
    """
    assert shots(locale), f"{locale.name} has no phone screenshots of its own"


def test_the_languages_show_the_same_screens() -> None:
    """One language quietly lagging the other is the failure worth catching."""
    counts = {p.name: [s.name for s in shots(p)] for p in locales()}
    assert len(set(map(tuple, counts.values()))) == 1, counts


@pytest.mark.parametrize("locale", locales(), ids=lambda p: p.name)
def test_screenshots_are_named_for_their_order(locale: pathlib.Path) -> None:
    """F-Droid orders them by filename; a padded 01.png sorts before 1.png."""
    for path in shots(locale):
        assert path.stem.isdigit(), f"{path.name} is not a number"
        assert path.stem == str(int(path.stem)), f"{path.name} is zero-padded"


@pytest.mark.parametrize("locale", locales(), ids=lambda p: p.name)
def test_screenshots_share_one_size(locale: pathlib.Path) -> None:
    """A set of mixed sizes renders as a ragged strip on the listing page."""
    sizes = {png_size(p) for p in shots(locale)}
    assert len(sizes) == 1, f"{locale.name} mixes screenshot sizes: {sizes}"
