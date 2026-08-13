#!/usr/bin/env python3
"""Generate the 1Sat Ordinals wrap for the Model Y (2025+) Premium template.

The concept: the 1Sat Ordinals roundel is stretched over the car from the nose
outward, so the bullseye's rings wrap the body as real concentric rings -- the
yellow core covers the front clip, the black ring bands the middle, and the
white outer ring covers the tail.

The mark ships in two versions, one for light backgrounds with black on the
outside and one for dark backgrounds with white on the outside. This wrap uses
the dark-background version, so the ring order out from the nose is yellow,
black, white.

Two properties of the wrap template make this work:

* It is a UV unwrap with the nose at the top of the image and the tail at the
  bottom, so the rings can be laid out directly in image space.
* It is very close to isotropic -- roughly 0.0050 m/px along the car against
  0.0054 m/px around it, measured off the front door panel. A circle drawn in
  the image is therefore a good stand-in for a circle laid over the body, which
  is what a stretched decal looks like.

ASPECT controls how strongly the rings bow. Taking it straight from those two
measurements (~1.08) over-curves them, because the unwrap is centered on the
roof line while the roundel is centered on the nose: a door pixel's horizontal
distance in the image is its arc length down from the roof, which is much
further than it actually sits from the nose badge. Pulling ASPECT back to 0.70
compensates and leaves the rings bowing the way a stretched decal would.

Because the rings are drawn as circles about the nose rather than as straight
cuts across the image, every ring edge curves: it sits furthest back along the
centerline (hood, roof, tailgate) and sweeps forward as it runs down the
flanks, the way a stretched decal would.

Radii come straight from the logo, so the proportions are the logo's own: the
core out to 0.602 of the radius, the middle ring to 0.734, the outer ring to
the edge. Along the length of the car that reads as ~60% yellow, ~13% black,
~27% white.

Edges are hard, as they are in the logo. They are rendered by supersampling and
downsampling, which anti-aliases the boundary without softening it into a fade.

Usage:  python3 tools/gen_1sat_wrap.py [--preview]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

TRIM = "modely-2025-premium"
OUT_NAME = "1Sat_Ordinals.png"

# Brand colors, sampled from https://1satordinals.com/images/logo-light.png
YELLOW = (240, 187, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# The three bands, nose outward, as the dark-background mark orders them, with
# radii as fractions of the logo's outer radius measured off the logo file.
CORE, R_CORE = YELLOW, 0.602
MIDDLE, R_MIDDLE = BLACK, 0.734
OUTER = WHITE

# Center of the roundel, in template pixels: the middle of the front fascia
# panel, which is the point of the car that goes through the logo first.
NOSE = (511.0, 62.0)

# How strongly the rings bow. See the note at the top of the file.
ASPECT = 0.70

# Supersampling factor used to anti-alias the hard ring edges.
SS = 4


def build(template_path, out_path, preview_path=None):
    tpl = Image.open(template_path).convert("RGBA")
    W, H = tpl.size

    # The template's alpha channel is exactly its paintable area: panel
    # interiors are opaque, everything outside them is transparent. Grow it by
    # a pixel so no unpainted seam shows along the panel outlines.
    mask = tpl.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(3))

    # Scale the roundel so its outer edge reaches the furthest painted pixel.
    # That pins the ring proportions to the car's full extent.
    ys, xs = np.nonzero(np.array(mask) > 128)
    outer = np.sqrt(((xs - NOSE[0]) * ASPECT) ** 2 + (ys - NOSE[1]) ** 2).max()

    # Distance field at supersampled resolution, so the ring edges land on a
    # sub-pixel grid and average down to a clean hard edge.
    yy, xx = np.mgrid[0:H * SS, 0:W * SS]
    dx = ((xx + 0.5) / SS - NOSE[0]) * ASPECT
    dy = (yy + 0.5) / SS - NOSE[1]
    dist = np.sqrt(dx * dx + dy * dy)

    art = np.empty((H * SS, W * SS, 3), dtype=np.uint8)
    art[...] = OUTER
    art[dist <= outer * R_MIDDLE] = MIDDLE
    art[dist <= outer * R_CORE] = CORE

    art = Image.fromarray(art).resize((W, H), Image.LANCZOS)

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(art, (0, 0), mask)
    out.save(out_path, optimize=True)

    if preview_path:
        # The wrap with the template outlines laid over it, to check where each
        # ring crosses the panels after a tweak.
        pv = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        pv.alpha_composite(out)
        t = np.array(tpl)
        dark = (t[..., 3] > 128) & (t[..., :3].mean(axis=2) < 128)
        lines = np.zeros((H, W, 4), dtype=np.uint8)
        lines[dark] = (255, 0, 255, 255)
        pv.alpha_composite(Image.fromarray(lines))
        pv.convert("RGB").save(preview_path)

    return np.array(mask) > 128


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, TRIM, "example", OUT_NAME)
    # The preview is a working aid, so it lands in the working directory rather
    # than alongside the wraps.
    preview = "1sat_preview.png" if "--preview" in sys.argv else None
    painted = build(os.path.join(root, TRIM, "template.png"), out, preview)

    art = np.array(Image.open(out).convert("RGB"))[painted]
    total = len(art)
    for name, color in (("core", CORE), ("middle", MIDDLE), ("outer", OUTER)):
        n = (np.abs(art.astype(int) - color).sum(axis=1) < 30).sum()
        print(f"  {name:6s} {n / total:6.1%} of painted area")
    print(f"{TRIM}: {os.path.getsize(out):,} bytes")


if __name__ == "__main__":
    main()
