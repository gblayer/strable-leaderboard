"""README banner in the site's palette, with the lettering converted to outlines so GitHub
renders Outfit and IBM Plex Mono without web fonts.

    pip install fonttools
    python scripts/make_banner.py

Writes docs/banner-light.svg and docs/banner-dark.svg. The two fonts (both OFL) are downloaded
into scripts/fonts/ on first use.
"""

import re
import urllib.request
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "scripts" / "fonts"
OUT = ROOT / "docs"
URLS = {
    "Outfit.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/outfit/Outfit%5Bwght%5D.ttf",
    "IBMPlexMono-Medium.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexmono/IBMPlexMono-Medium.ttf",
    "NotoSansMono.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmono/NotoSansMono%5Bwdth%2Cwght%5D.ttf",
}

_cache = {}


def font(name, wght=None, wdth=None):
    key = (name, wght, wdth)
    if key not in _cache:
        FONTS.mkdir(exist_ok=True)
        if not (FONTS / name).exists():
            urllib.request.urlretrieve(URLS[name], FONTS / name)
        f = TTFont(FONTS / name)
        if wght is not None and "fvar" in f:
            axes = {"wght": wght, **({"wdth": wdth} if wdth is not None else {})}
            f = instantiateVariableFont(f, axes)
        _cache[key] = f
    return _cache[key]


def text_path(f, text, size, x, y, tracking=0.0):
    """SVG path data for ``text`` with its baseline at (x, y). Returns (d, end_x)."""
    gs, cmap, upm = f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm
    k = size / upm
    parts, pen_x = [], x
    for ch in text:
        g = cmap.get(ord(ch))
        if g is None:
            pen_x += size * 0.3
            continue
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (k, 0, 0, -k, pen_x, y)))
        parts.append(pen.getCommands())
        pen_x += gs[g].width * k + tracking * size
    d = " ".join(p for p in parts if p)
    d = re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group(0)):.1f}".rstrip("0").rstrip("."), d)  # smaller files
    return d, pen_x


LIGHT = dict(bg="#E9D4C3", surface="#F3E6DA", surface2="#E3CCB9", line="#D4BBA6", ink="#13273F", ink2="#34465E",
             ink3="#4C5767", head="#4E0000", accent="#4E0000", accent_ink="#F3E6DA", accent2="#223631", mark="#8E2A33",
             hdr="#223631", hdr_ink="#F3E6DA")
DARK = dict(bg="#13273F", surface="#1A3050", surface2="#233B5D", line="#2E4768", ink="#E9D4C3", ink2="#CDBBAC",
            ink3="#A2A9B3", head="#E9D4C3", accent="#9E2F3A", accent_ink="#F3E6DA", accent2="#91ACA5", mark="#D0646E",
            hdr="#3A5A50", hdr_ink="#F3E6DA")

W, H = 1200, 340


def sheet(c, ox, oy):
    """The hero's spreadsheet motif: a string column dropped into a table, and a selected cell holding "str"."""
    sw, sh = 420, 268
    g = [f'<g transform="translate({ox},{oy})">',
         f'<rect x="0" y="0" width="{sw}" height="{sh}" rx="10" fill="{c["surface"]}" stroke="{c["line"]}"/>',
         f'<rect x="0" y="0" width="{sw}" height="14" fill="{c["surface2"]}"/>',
         f'<rect x="0" y="0" width="18" height="{sh}" fill="{c["surface2"]}"/>',
         f'<rect x="237" y="0" width="171" height="14" fill="{c["hdr"]}"/>',
         f'<rect x="0" y="15" width="18" height="55" fill="{c["hdr"]}"/>']
    for x in [18.5, 100.5, 168.5, 236.5, 284.5, 346.5, 408.5]:
        g.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{sh}" stroke="{c["line"]}"/>')
    for y in [14.5, 42.5, 70.5, 98.5, 126.5, 154.5, 182.5, 210.5, 238.5]:
        g.append(f'<line x1="0" y1="{y}" x2="{sw}" y2="{y}" stroke="{c["line"]}"/>')
    mono = font("IBMPlexMono-Medium.ttf")
    hdrs = [("A", 59.5, c["ink3"]), ("B", 134.5, c["ink3"]), ("C", 202.5, c["ink3"]),
            ("D", 260.5, c["hdr_ink"]), ("E", 315.5, c["hdr_ink"]), ("F", 377.5, c["hdr_ink"])]
    for t, x, col in hdrs:
        d, _ = text_path(mono, t, 7.5, x - 2.3, 10)
        g.append(f'<path d="{d}" fill="{col}"/>')
    for t, y, col in [("1", 31.5, c["hdr_ink"]), ("2", 59.5, c["hdr_ink"])] + [(str(n), 87.5 + 28 * (n - 3), c["ink3"]) for n in range(3, 10)]:
        d, _ = text_path(mono, t, 7.5, 6.7, y)
        g.append(f'<path d="{d}" fill="{col}"/>')
    bold = font("Outfit.ttf", 700)
    for t, x in [("NAME", 19), ("CODE", 101), ("CITY", 169), ("Y", 237)]:
        w = 81 if t == "NAME" else 67 if t != "Y" else 47
        g.append(f'<rect x="{x}" y="99" width="{w}" height="27" fill="{c["ink"]}"/>')
        d, _ = text_path(bold, t, 9, x + 8, 116, tracking=0.1)
        g.append(f'<path d="{d}" fill="{c["surface"]}"/>')
    rows = [("ACME CO", "ICD-9", "PARIS", "4.2"), ("ZORBEL LTD", "N18.3", "BERLIN", "3.9"),
            ("PLUMERA SA", "90210", "MILAN", "5.1"), ("HRSA", "Z-114", "LYON", "2.7")]
    for i, (a, b, city, yv) in enumerate(rows):
        y = 144 + 28 * i
        g.append(f'<rect x="169" y="{y - 17}" width="67" height="27" fill="{c["accent"]}"/>')
        for t, x, col in [(a, 27, c["ink2"]), (b, 109, c["ink2"]), (city, 177, c["accent_ink"]), (yv, 245, c["ink3"])]:
            d, _ = text_path(mono, t, 9.5, x, y)
            g.append(f'<path d="{d}" fill="{col}"/>')
    g.append(f'<rect x="168.5" y="98.5" width="68" height="140" fill="none" stroke="{c["mark"]}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    g.append(f'<rect x="18.5" y="98.5" width="266" height="140" fill="none" stroke="{c["ink"]}" stroke-width="1.5"/>')
    g.append(f'<rect x="236.5" y="14.5" width="172" height="56" fill="{c["surface"]}" stroke="{c["mark"]}" stroke-width="2"/>')
    d, end = text_path(bold, '"', 40, 252, 56)
    g.append(f'<path d="{d}" fill="{c["accent2"]}"/>')
    d, end = text_path(bold, "str", 40, end, 56, tracking=-0.01)
    g.append(f'<path d="{d}" fill="{c["head"]}"/>')
    d, end2 = text_path(bold, '"', 40, end, 56)
    g.append(f'<path d="{d}" fill="{c["accent2"]}"/>')
    g.append(f'<rect x="{end2 + 6:.1f}" y="24" width="3" height="37" fill="{c["mark"]}"/>')
    g.append(f'<rect x="404.5" y="66.5" width="8" height="8" fill="{c["mark"]}" stroke="{c["surface"]}" stroke-width="1.5"/>')
    g.append("</g>")
    return "\n".join(g)


def banner(c):
    black = font("Outfit.ttf", 800)
    medium = font("Outfit.ttf", 500)
    ipa = font("NotoSansMono.ttf", 500, 100)  # Plex Mono lacks the IPA glyphs
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="STRABLE, STRing-tABLE, pronounced stray-buhl: the living leaderboard of 108 real-world tables with raw strings">',
           f'<rect width="{W}" height="{H}" rx="18" fill="{c["bg"]}"/>']
    d, _ = text_path(black, "STRABLE", 124, 60, 158, tracking=-0.03)
    out.append(f'<path d="{d}" fill="{c["head"]}"/>')
    x = 64
    for t, f, col in [("STR", black, c["head"]), ("ing-t", medium, c["ink2"]), ("ABLE", black, c["head"])]:
        d, x = text_path(f, t, 26, x, 206, tracking=0.01)
        out.append(f'<path d="{d}" fill="{col}"/>')
    d, x = text_path(ipa, "/\u02c8stre\u026a.b\u0259l/", 21, x + 18, 206)
    out.append(f'<path d="{d}" fill="{c["ink2"]}"/>')
    d, x = text_path(medium, "stray-buhl", 24, x + 18, 206)
    out.append(f'<path d="{d}" fill="{c["ink3"]}"/>')
    for line, y in [("The living leaderboard of 108 real-world", 262), ("tables with raw strings.", 294)]:
        d, _ = text_path(medium, line, 23, 64, y)
        out.append(f'<path d="{d}" fill="{c["ink2"]}"/>')
    out.append(sheet(c, W - 420 - 56, 36))
    out.append("</svg>")
    return "\n".join(out)


for name, c in [("light", LIGHT), ("dark", DARK)]:
    path = OUT / f"banner-{name}.svg"
    path.write_text(banner(c))
    print("wrote", path)
