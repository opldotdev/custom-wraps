#!/usr/bin/env python3
"""Generate the 1Sat Ordinals wrap for the Model Y (2025+) templates.

The concept: the car drives nose-first through the 1Sat Ordinals roundel, so
the bullseye's rings transfer onto it along the length of the body -- the
yellow core lands on the front clip, the white ring wraps the doors, and the
black outer ring covers the tail.

The wrap templates are UV unwraps with the nose at the top of the image and the
tail at the bottom, so longitudinal position along the car is very nearly
linear in image y. That makes each ring a horizontal band. The band edges are
pinned to the shutlines measured off each template so a ring never ends
part-way across a panel, and the feather between bands is kept tight so they
read as crisp as the rings in the logo rather than as an airbrushed fade.

Usage:  python3 tools/gen_1sat_wrap.py [--preview]
"""
import os
import sys

from PIL import Image, ImageFilter

# Brand colors, sampled from https://1satordinals.com/images/logo-light.png
YELLOW = (240, 187, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

OUT_NAME = "1Sat_Ordinals.png"

# (yellow -> white edge, white -> black edge) as fractions of vehicle length,
# 0 = nose, 1 = tail. Measured per template: the first edge sits in the front
# fender / front door gap, the second in the front door / rear door gap.
TRIMS = {
    "modely-2025-base": (0.334, 0.545),
    "modely-2025-premium": (0.345, 0.566),
    "modely-2025-performance": (0.345, 0.566),
}
FEATHER = 0.012


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def smoothstep(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def bands(front, rear):
    h = FEATHER / 2
    return [(0.0, YELLOW), (front - h, YELLOW), (front + h, WHITE),
            (rear - h, WHITE), (rear + h, BLACK), (1.0, BLACK)]


def sample(stops, t):
    if t <= stops[0][0]:
        return stops[0][1]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            return lerp(c0, c1, smoothstep(t0, t1, t))
    return stops[-1][1]


def build(template_path, out_path, edges, preview_path=None):
    tpl = Image.open(template_path).convert("RGBA")
    W, H = tpl.size
    stops = bands(*edges)

    art = Image.new("RGB", (W, H))
    pixels = art.load()
    for y in range(H):
        color = sample(stops, y / (H - 1))
        for x in range(W):
            pixels[x, y] = color

    # A template's alpha channel is exactly its paintable area: panel interiors
    # are opaque, everything outside them is transparent. Grow it by a pixel so
    # no unpainted seam shows along the panel outlines.
    mask = tpl.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(3))

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(art, (0, 0), mask)
    out.save(out_path, optimize=True)

    if preview_path:
        # The wrap with the template outlines laid over it, to check that each
        # band edge still lands on a shutline after a tweak.
        pv = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        pv.alpha_composite(out)
        lines = tpl.copy()
        lp = lines.load()
        for y in range(H):
            for x in range(W):
                r, g, b, a = lp[x, y]
                dark = a > 128 and (r + g + b) / 3 < 128
                lp[x, y] = (255, 0, 255, 255) if dark else (0, 0, 0, 0)
        pv.alpha_composite(lines)
        pv.convert("RGB").save(preview_path)


def main():
    preview = "--preview" in sys.argv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for trim, edges in TRIMS.items():
        out = os.path.join(root, trim, "example", OUT_NAME)
        build(os.path.join(root, trim, "template.png"), out, edges,
              os.path.join(root, trim, "example", "preview.png") if preview else None)
        print(f"{trim}: {os.path.getsize(out):,} bytes")


if __name__ == "__main__":
    main()
