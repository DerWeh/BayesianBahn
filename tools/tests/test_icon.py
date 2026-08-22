"""Guards on the icon F-Droid publishes.

0.1.2 was listed on F-Droid with no icon at all. The app ships an adaptive icon
only — an XML in `mipmap-anydpi-v26` plus a vector foreground, which is all a
minSdk-26 app needs — and F-Droid, which extracts the icon by looking for a
raster in the density buckets, found nothing it could rasterise. The fix is a
PNG in the fastlane tree, rendered from the same vector so the two cannot drift.

These tests need no rasteriser: `render_icon.build_svg()` is pure stdlib, and
the PNG's dimensions come out of its IHDR header. What they cannot check is
whether the committed PNG is the current render — that is
`render_icon.py --check`, which needs cairosvg.
"""

from __future__ import annotations

import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import render_icon as R  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(path: Path) -> tuple[int, int]:
    """Width and height straight out of the IHDR chunk."""
    data = path.read_bytes()
    assert data[:8] == PNG_MAGIC, f"{path} is not a PNG"
    assert data[12:16] == b"IHDR", f"{path} does not start with an IHDR chunk"
    return struct.unpack(">II", data[16:24])


def test_the_icon_exists_where_fdroid_looks():
    """F-Droid reads fastlane/metadata/android/<locale>/images/icon.png. The
    whole bug was that this file did not exist."""
    assert R.OUT.is_file(), f"{R.OUT} is missing — F-Droid will list no icon"
    assert R.OUT.parent == R.ROOT / "fastlane/metadata/android/en-US/images"


def test_the_icon_is_a_square_png_of_the_expected_size():
    assert png_size(R.OUT) == (R.SIZE, R.SIZE)


def test_the_svg_paints_the_background_the_adaptive_icon_declares():
    """A brand-colour change in colors.xml must not leave the icon behind."""
    colours = ET.parse(R.COLORS).getroot()
    declared = next(c.text.strip() for c in colours.iter("color")
                    if c.get("name") == "ic_launcher_background")
    assert f'fill="{declared}"' in R.build_svg()


def test_every_path_in_the_vector_reaches_the_svg():
    vector = ET.parse(R.drawable_path("@drawable/ic_launcher_foreground")).getroot()
    in_vector = len([n for n in vector.iter("path") if n.get(R.ANDROID + "pathData")])
    assert in_vector > 0
    assert R.build_svg().count("<path ") == in_vector


def test_the_render_crops_to_what_a_launcher_shows():
    """The outer sixth of an adaptive icon is bleed the launcher masks away;
    rendering the full canvas publishes an icon ringed by dead background."""
    view_box = re.search(r'viewBox="([^"]+)"', R.build_svg()).group(1)
    left, top, width, height = (float(v) for v in view_box.split())
    canvas = float(ET.parse(
        R.drawable_path("@drawable/ic_launcher_foreground")
    ).getroot().get(R.ANDROID + "viewportWidth"))
    assert width == height == canvas * R.VISIBLE_FRACTION
    assert left == top == (canvas - width) / 2


def test_a_group_transform_is_carried_over():
    """The foreground's artwork sits inside a translated <group>; dropping the
    transform would render it off-centre and still produce a plausible PNG."""
    assert 'transform="translate(' in R.build_svg()
