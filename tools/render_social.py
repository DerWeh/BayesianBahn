"""Render the GitHub social preview card from the launcher vector.

GitHub has no repository icon. What it has is the *social preview*: the image
shown when a repo link is unfurled on a chat or social site, and in GitHub's own
search results. Settings -> General -> Social preview, 1280x640.

Reuses render_icon's translation of the Android vector, so the mark on the card
is the same artwork as the launcher icon and cannot drift from it.

Unlike the F-Droid icon this card has text on it, so its output depends on the
fonts installed on the machine that renders it. That is why it has no --check
mode and lives outside the test suite: the committed PNG is the artefact, and
this script is how it was made rather than something CI reproduces.

Needs cairosvg:

    python3 -m venv /tmp/iconvenv && /tmp/iconvenv/bin/pip install cairosvg
    /tmp/iconvenv/bin/python tools/render_social.py

GitHub silently lost the first upload of this card: it recorded that a custom
preview existed — the page's og:image pointed at a repository-images UUID rather
than the generated default — but fetching that URL returned 404
WebContentNotFound, so the record was there and the blob was not. cairosvg emits
a `bKGD` chunk, which is valid, rare, and precisely the sort of thing an image
pipeline mishandles, so the PNG is now re-encoded through Pillow with only the
chunks it needs. `--format jpg` is the escape hatch if a PNG is refused again.

Usage:
    python tools/render_social.py [--format png|jpg]
"""

from __future__ import annotations

import argparse
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import render_icon as icon  # noqa: E402

OUT = icon.ROOT / "docs/social-preview"   # suffix added per --format

WIDTH, HEIGHT = 1280, 640

# GitHub crops the card to different aspect ratios depending on where it is
# unfurled, so nothing that matters may sit near an edge.
MARGIN = 96

TITLE = "BayesianBahn"
SUBTITLE = "Arrival times as a distribution, not a guess"
FOOTER = "Delay predictions for Deutsche Bahn · MIT · F-Droid"

# Present on essentially every Linux box and on the CI images; the fallback
# below keeps the card legible rather than correct if it is ever absent.
FONT = "DejaVu Sans, Noto Sans, sans-serif"


def foreground_group() -> str:
    """The launcher artwork as an SVG group, without its background."""
    vector = ET.parse(icon.drawable_path("@drawable/ic_launcher_foreground")).getroot()
    parts = []
    for node in vector:
        if node.tag == "path":
            parts.append(icon.path_element(node))
        elif node.tag == "group":
            transform = icon.group_transform(node)
            parts.append(f'<g transform="{transform}">' if transform else "<g>")
            for child in node:
                if child.tag == "path":
                    parts.append(icon.path_element(child))
            parts.append("</g>")
    return "\n".join(parts)


def build_svg() -> str:
    background = icon.resolve_colour("@color/ic_launcher_background")
    canvas = float(ET.parse(
        icon.drawable_path("@drawable/ic_launcher_foreground")
    ).getroot().get(icon.ANDROID + "viewportWidth"))

    # The mark, cropped to the launcher-visible area exactly as the icon is,
    # then scaled into a square on the left of the card.
    visible = canvas * icon.VISIBLE_FRACTION
    inset = (canvas - visible) / 2
    mark = 340
    mark_x, mark_y = MARGIN, (HEIGHT - mark) / 2
    scale = mark / visible

    text_x = mark_x + mark + 72
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{background}" />
  <g transform="translate({mark_x},{mark_y}) scale({scale:g}) translate({-inset:g},{-inset:g})">
{foreground_group()}
  </g>
  <text x="{text_x}" y="{HEIGHT / 2 - 46:g}" font-family="{FONT}" font-size="74"
        font-weight="700" fill="#FFFFFF">{TITLE}</text>
  <text x="{text_x}" y="{HEIGHT / 2 + 18:g}" font-family="{FONT}" font-size="31"
        fill="#BBD4E6">{SUBTITLE}</text>
  <text x="{text_x}" y="{HEIGHT / 2 + 86:g}" font-family="{FONT}" font-size="22"
        fill="#7FA6C4">{FOOTER}</text>
</svg>"""


def check_margins(image) -> None:
    """Refuse to write a card whose content runs into the crop zone.

    Text width depends on the font that was actually used, so the layout cannot
    be reasoned about in advance — it has to be measured after rendering. The
    first version of this card put the footer 35px from the right edge, which no
    reading of the source would have revealed.
    """
    width, height = image.size
    background = image.getpixel((5, 5))
    pixels = image.load()
    left, top, right, bottom = width, height, 0, 0
    for y in range(height):
        for x in range(width):
            if pixels[x, y] != background:
                left, right = min(left, x), max(right, x)
                top, bottom = min(top, y), max(bottom, y)
    margins = {"left": left, "right": width - 1 - right,
               "top": top, "bottom": height - 1 - bottom}
    tight = {side: value for side, value in margins.items() if value < MARGIN}
    if tight:
        raise SystemExit(
            f"content sits inside the {MARGIN}px margin: {tight} — "
            f"shorten the text or reduce the font size")
    print("  margins " + "  ".join(f"{k} {v}" for k, v in margins.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=("png", "jpg"), default="png",
                    help="png by default; jpg if GitHub refuses the PNG again")
    args = ap.parse_args()

    try:
        import cairosvg
    except ImportError:
        raise SystemExit("needs cairosvg — see the module docstring for the venv recipe")
    from PIL import Image

    raw = cairosvg.svg2png(bytestring=build_svg().encode("utf-8"),
                           output_width=WIDTH, output_height=HEIGHT)
    # Re-encode rather than shipping cairosvg's own file: flattened onto the
    # card's background so there is no alpha channel to interpret, and written
    # by Pillow so the result carries IHDR, IDAT and IEND and nothing else.
    rendered = Image.open(io.BytesIO(raw))
    flat = Image.new("RGB", rendered.size, icon.resolve_colour("@color/ic_launcher_background"))
    flat.paste(rendered, mask=rendered.split()[-1] if rendered.mode == "RGBA" else None)
    check_margins(flat)

    out = OUT.with_suffix("." + args.format)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "png":
        flat.save(out, format="PNG", optimize=True)
    else:
        flat.save(out, format="JPEG", quality=92, subsampling=0, optimize=True)
    print(f"wrote {out.relative_to(icon.ROOT)} "
          f"({out.stat().st_size:,} bytes, {WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
