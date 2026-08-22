"""Render the launcher icon to the PNG F-Droid needs.

The app ships an adaptive icon only: `mipmap-anydpi-v26/ic_launcher.xml` plus a
vector foreground. That is everything a device with API 26+ wants, and minSdk is
26, so there is no reason for the APK to carry raster copies. F-Droid, however,
extracts the app icon from the APK by looking for a raster in the density
buckets, finds an XML it cannot rasterise, and lists the app with no icon —
which is what happened to 0.1.2.

The documented fix is a PNG in the fastlane tree, which F-Droid prefers over
anything it can dig out of the APK. Rendering it from the vector rather than
drawing it by hand is the point of this script: the published icon and the
launcher icon cannot drift apart, because there is only one source.

Android's `android:pathData` is SVG path syntax, so the translation is
mechanical: read the viewport, the group transform and the paths, emit an SVG,
and rasterise it over the adaptive icon's background colour.

Needs cairosvg, which is not a project dependency — this runs when the icon
changes, which is roughly never:

    python3 -m venv /tmp/iconvenv && /tmp/iconvenv/bin/pip install cairosvg
    /tmp/iconvenv/bin/python tools/render_icon.py

Usage:
    python tools/render_icon.py [--check]

    --check  fail instead of writing, for CI: asserts the committed PNG is what
             the current vector renders to.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "app/src/main/res"
ADAPTIVE = RES / "mipmap-anydpi-v26/ic_launcher.xml"
COLORS = RES / "values/colors.xml"
OUT = ROOT / "fastlane/metadata/android/en-US/images/icon.png"

ANDROID = "{http://schemas.android.com/apk/res/android}"

# F-Droid's store icon size. Also what the Play listing wants, so one file does
# for both if the app is ever published elsewhere.
SIZE = 512

# An adaptive icon is authored on a 108dp canvas of which the launcher shows
# only the middle 72dp — the outer sixth on each edge is bleed, there so the
# icon can be shifted for parallax without exposing a gap. Rendering the whole
# canvas would therefore publish an icon noticeably smaller than the one on the
# phone, ringed by empty background. This is the fraction the launcher keeps.
VISIBLE_FRACTION = 72 / 108


def attr(node: ET.Element, name: str, default: str | None = None) -> str | None:
    return node.get(ANDROID + name, default)


def resolve_colour(reference: str) -> str:
    """`@color/ic_launcher_background` -> the literal from values/colors.xml."""
    if not reference.startswith("@color/"):
        return reference
    name = reference.removeprefix("@color/")
    for colour in ET.parse(COLORS).getroot().iter("color"):
        if colour.get("name") == name:
            return (colour.text or "").strip()
    raise SystemExit(f"{COLORS}: no colour named {name!r}")


def drawable_path(reference: str) -> Path:
    """`@drawable/ic_launcher_foreground` -> the file that defines it."""
    if not reference.startswith("@drawable/"):
        raise SystemExit(f"not a drawable reference: {reference!r}")
    return RES / "drawable" / f"{reference.removeprefix('@drawable/')}.xml"


def argb_to_svg(colour: str) -> tuple[str, str]:
    """Android colours may carry an alpha nibble pair that SVG puts elsewhere."""
    value = colour.lstrip("#")
    if len(value) == 8:                      # #AARRGGBB
        alpha = int(value[:2], 16) / 255.0
        return f"#{value[2:]}", f"{alpha:g}"
    return f"#{value}", "1"


def path_element(node: ET.Element) -> str:
    data = attr(node, "pathData")
    if not data:
        return ""
    fill, fill_opacity = argb_to_svg(attr(node, "fillColor", "#00000000"))
    bits = [f'd="{data}"', f'fill="{fill}"', f'fill-opacity="{fill_opacity}"']
    stroke = attr(node, "strokeColor")
    if stroke:
        colour, opacity = argb_to_svg(stroke)
        bits += [f'stroke="{colour}"', f'stroke-opacity="{opacity}"',
                 f'stroke-width="{attr(node, "strokeWidth", "1")}"',
                 f'stroke-linecap="{attr(node, "strokeLineCap", "butt")}"',
                 f'stroke-linejoin="{attr(node, "strokeLineJoin", "miter")}"']
    return "  <path " + " ".join(bits) + " />"


def group_transform(node: ET.Element) -> str:
    parts = []
    tx, ty = attr(node, "translateX", "0"), attr(node, "translateY", "0")
    if float(tx) or float(ty):
        parts.append(f"translate({tx},{ty})")
    sx, sy = attr(node, "scaleX", "1"), attr(node, "scaleY", "1")
    if float(sx) != 1 or float(sy) != 1:
        parts.append(f"scale({sx},{sy})")
    rotation = attr(node, "rotation")
    if rotation and float(rotation):
        parts.append(f"rotate({rotation},{attr(node, 'pivotX', '0')},"
                     f"{attr(node, 'pivotY', '0')})")
    return " ".join(parts)


def vector_to_svg(vector_file: Path, background: str) -> str:
    root = ET.parse(vector_file).getroot()
    if root.tag != "vector":
        raise SystemExit(f"{vector_file}: expected a <vector>, got <{root.tag}>")
    width = float(attr(root, "viewportWidth"))
    height = float(attr(root, "viewportHeight"))
    # Crop to what a launcher actually shows, centred on the canvas.
    view_w, view_h = width * VISIBLE_FRACTION, height * VISIBLE_FRACTION
    left, top = (width - view_w) / 2, (height - view_h) / 2
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" '
            f'height="{SIZE}" viewBox="{left:g} {top:g} {view_w:g} {view_h:g}">',
            f'  <rect x="{left:g}" y="{top:g}" width="{view_w:g}" '
            f'height="{view_h:g}" fill="{background}" />']
    for node in root:
        if node.tag == "path":
            body.append(path_element(node))
        elif node.tag == "group":
            transform = group_transform(node)
            body.append(f'  <g transform="{transform}">' if transform else "  <g>")
            for child in node:
                if child.tag == "path":
                    body.append("  " + path_element(child))
                elif child.tag == "group":
                    raise SystemExit(f"{vector_file}: nested <group> not handled")
            body.append("  </g>")
        elif node.tag in ("clip-path",):
            raise SystemExit(f"{vector_file}: <{node.tag}> not handled")
    body.append("</svg>")
    return "\n".join(body)


def build_svg() -> str:
    adaptive = ET.parse(ADAPTIVE).getroot()
    background = adaptive.find("background")
    foreground = adaptive.find("foreground")
    if background is None or foreground is None:
        raise SystemExit(f"{ADAPTIVE}: needs both <background> and <foreground>")
    colour = resolve_colour(attr(background, "drawable", ""))
    return vector_to_svg(drawable_path(attr(foreground, "drawable", "")), colour)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed PNG is not what the vector renders to")
    args = ap.parse_args()

    try:
        import cairosvg
    except ImportError:
        raise SystemExit("needs cairosvg — see the module docstring for the venv recipe")

    png = cairosvg.svg2png(bytestring=build_svg().encode("utf-8"),
                           output_width=SIZE, output_height=SIZE)
    if args.check:
        if not OUT.is_file():
            raise SystemExit(f"{OUT} does not exist; run without --check")
        if OUT.read_bytes() != png:
            raise SystemExit(f"{OUT} is stale — the launcher vector has changed "
                             f"since it was rendered; re-run without --check")
        print(f"{OUT.relative_to(ROOT)} matches the launcher vector")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(png)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(png):,} bytes, {SIZE}x{SIZE})")


if __name__ == "__main__":
    main()
