#!/usr/bin/env python3
"""Generate the 1Sat Ordinals race wrap for the Model Y (2025+) Premium template.

A mostly-black motorsport livery: twin yellow centre stripes down the spine of
the car, a 1SAT wordmark across the hood and both front doors, and the 1Sat
roundel on the rear quarters as a competition number disc.

Orientation notes, all of which follow from the template being a UV unwrap
seen from above with the nose at the top of the image:

* The car's left side is the left of the image. Looking down with the nose
  pointing away from you, the driver's side falls on your left.
* On a side panel, the edge nearer the middle of the image is the top of the
  car and the outer edge is the rocker, because the unwrap peels down the
  flanks from the roof.
* For a wordmark to read correctly on both flanks it has to be mirrored in
  world space, which puts its first letter toward the nose on the left side and
  toward the tail on the right. That is how vehicle lettering actually works,
  and cybertruck/example/Graffiti_orange.png in this repo is drawn the same
  way. On this template the flanks are unwrapped as two columns that peel in
  opposite directions, so that world mirror comes out as a 180 degree rotation
  of the artwork rather than a horizontal flip -- flipping it horizontally
  makes the right-hand door read backwards.
* Hood lettering is rotated 180 degrees in the unwrap so it reads to someone
  standing in front of the car, the way motorsport hood graphics normally do.
  Set HOOD_READS_FROM to "rear" to have it read from the windshield instead.

Note that the Model Y's roof is glass and therefore unpainted, so the stripes
run over the hood, break across the roof, and pick up again on the tailgate.

Usage:  python3 tools/gen_1sat_race_wrap.py [--verify]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

TRIM = "modely-2025-premium"
OUT_NAME = "1Sat_Ordinals_Race.png"

# Brand colors, sampled from https://1satordinals.com/images/logo-light.png
YELLOW = (240, 187, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

HOOD_READS_FROM = "front"

FONTS = [
    "/mnt/skills/examples/canvas-design/canvas-fonts/BigShoulders-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Panel boxes measured off modely-2025-premium/template.png, as (x0, y0, x1, y1).
HOOD = (375, 110, 644, 338)
DOOR_L, DOOR_R = (82, 348, 258, 594), (762, 350, 937, 595)
QUARTER_L, QUARTER_R = (124, 779, 264, 949), (757, 782, 898, 952)

CENTER_X = 511.5
# Half-widths from the centreline: the yellow stripes and the white pinstripes
# that edge them.
STRIPE_IN, STRIPE_OUT = 24, 64
PIN_IN, PIN_OUT = 70, 78

SKEW = 0.18  # forward lean on the lettering


def load_font(size):
    for path in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise SystemExit("no usable font found")


def render_text(text, cap_h, fill, outline=None, stroke=0):
    """Draw text at roughly cap_h pixels tall, sheared forward, tightly cropped."""
    font = load_font(int(cap_h * 1.35))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    pad = int(cap_h * 0.6)
    img = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill=fill,
                             stroke_width=stroke, stroke_fill=outline)
    # Shear about the baseline so the lean reads as speed rather than a tilt.
    w, h = img.size
    img = img.transform((w + int(h * SKEW), h), Image.AFFINE,
                        (1, SKEW, -SKEW * h, 0, 1, 0), resample=Image.BICUBIC)
    return np.array(img.crop(img.getbbox()))


def over(dst, src, x, y):
    """Alpha-composite src (RGBA array) onto dst (RGB array) at top-left x, y."""
    x, y = int(round(x)), int(round(y))
    h, w = src.shape[:2]
    sx, sy = max(0, -x), max(0, -y)
    x, y = max(0, x), max(0, y)
    h, w = min(h - sy, dst.shape[0] - y), min(w - sx, dst.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    src = src[sy:sy + h, sx:sx + w]
    region = dst[y:y + h, x:x + w]
    a = src[..., 3:4] / 255.0
    region[...] = (src[..., :3] * a + region * (1 - a)).astype(np.uint8)


def panel_labels(tpl):
    """Label each panel interior. Interiors are the opaque light areas of the
    template; the dark outlines separate one panel from the next."""
    t = np.array(tpl)
    interior = (t[..., 3] > 128) & (t[..., :3].mean(axis=2) > 128)
    return ndimage.label(interior)[0]


def panel_at(labels, box):
    """Mask of the panel occupying the given box, by largest area within it."""
    x0, y0, x1, y1 = box
    vals, counts = np.unique(labels[y0:y1, x0:x1], return_counts=True)
    keep = vals > 0
    return labels == vals[keep][counts[keep].argmax()]


def anchor(panel):
    """Centre of mass of a panel -- a better anchor than the bounding box
    centre, since none of these panels are rectangles."""
    cy, cx = ndimage.center_of_mass(panel)
    return cx, cy


def disc(art, panel, want_r, ss=4):
    """1Sat roundel, dark-background variant: a white outer ring, then a black
    ring, then the yellow core, at the logo's own radius ratios.

    The mark ships in two versions. The light-background one puts black on the
    outside; the dark-background one swaps that for white, so the ring reads
    against the body instead of disappearing into it. This car is black, so it
    takes the dark version -- an outer black ring here would vanish into the
    bodywork and leave the white ring looking like the edge of the mark.

    Sits at the point of the panel furthest from any edge, sized so the whole
    roundel clears the panel rather than being cut by the mask."""
    clear = ndimage.distance_transform_edt(panel)
    cy, cx = np.unravel_index(clear.argmax(), clear.shape)
    radius = min(want_r, clear[cy, cx] - 3)
    r = int(radius) + 2
    yy, xx = np.mgrid[-r * ss:r * ss, -r * ss:r * ss]
    d = np.sqrt(xx ** 2 + yy ** 2) / ss
    rgba = np.zeros((2 * r * ss, 2 * r * ss, 4), np.uint8)
    rgba[d <= radius] = (*WHITE, 255)
    rgba[d <= radius * 0.734] = (*BLACK, 255)
    rgba[d <= radius * 0.602] = (*YELLOW, 255)
    small = np.array(Image.fromarray(rgba).resize((2 * r, 2 * r), Image.LANCZOS))
    over(art, small, cx - r, cy - r)
    return radius


def build(template_path, out_path):
    tpl = Image.open(template_path).convert("RGBA")
    W, H = tpl.size

    mask = tpl.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(3))

    painted = np.array(mask) > 128
    labels = panel_labels(tpl)
    art = np.zeros((H, W, 3), np.uint8)  # black base

    # Twin centre stripes, full length of the car.
    off = np.abs(np.arange(W) - CENTER_X)
    art[:, (off >= STRIPE_IN) & (off <= STRIPE_OUT)] = YELLOW
    art[:, (off >= PIN_IN) & (off <= PIN_OUT)] = WHITE

    def place(block, box, label):
        """Sit a block on its panel, starting from the panel's centre of mass
        and nudging until none of it spills onto neighbouring bodywork. Panels
        are irregular, so the centre of mass is a starting guess, not an answer."""
        panel = panel_at(labels, box)
        cx, cy = anchor(panel)
        bh, bw = block.shape[:2]
        ink = block[..., 3] > 8
        best = None
        for dy in range(-16, 17, 2):
            for dx in range(-16, 17, 2):
                x, y = int(round(cx - bw / 2)) + dx, int(round(cy - bh / 2)) + dy
                if x < 0 or y < 0 or x + bw > W or y + bh > H:
                    continue
                spill = int((ink & painted[y:y + bh, x:x + bw]
                             & ~panel[y:y + bh, x:x + bw]).sum())
                score = (spill, dx * dx + dy * dy)
                if best is None or score < best[0]:
                    best = (score, x, y)
        (spill, _), x, y = best
        over(art, block, x, y)
        print(f"  {label:12s} {bw}x{bh}px, {spill} px onto other panels")

    # Hood wordmark: white so it stays legible where it crosses the stripes,
    # with a black stroke to hold it off the yellow.
    hood = render_text("1SAT", cap_h=80, fill=WHITE, outline=BLACK, stroke=5)
    if HOOD_READS_FROM == "front":
        hood = hood[::-1, ::-1]
    place(hood, HOOD, "hood")

    # Door wordmarks. Built for the left flank, then mirrored in world space
    # for the right so each reads correctly on its own side. Because the two
    # flanks peel in opposite directions, that mirror is a 180 degree rotation.
    door = render_text("1SAT", cap_h=92, fill=YELLOW)
    # The text runs along the car (image y) and stands up across it (image x).
    left = np.flip(door.transpose(1, 0, 2), axis=1)
    place(left, DOOR_L, "door left")
    place(left[::-1, ::-1], DOOR_R, "door right")

    # Roundels on the rear quarters, as competition number discs.
    for box, side in ((QUARTER_L, "left"), (QUARTER_R, "right")):
        r = disc(art, panel_at(labels, box), 52)
        print(f"  roundel {side:5s} radius {r:.0f}px")

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(Image.fromarray(art), (0, 0), mask)
    out.save(out_path, optimize=True)
    return painted


def verify(out_path):
    """Write flank and hood views as they are actually seen, to check that every
    wordmark reads the right way round."""
    w = np.array(Image.open(out_path).convert("RGB"))
    x0, y0, x1, y1 = DOOR_L
    lf = np.flip(w[y0:y1, x0:x1].transpose(1, 0, 2), axis=0)
    x0, y0, x1, y1 = DOOR_R
    rf = np.flip(w[y0:y1, x0:x1].transpose(1, 0, 2), axis=1)
    x0, y0, x1, y1 = HOOD
    hd = w[y0:y1, x0:x1][::-1, ::-1] if HOOD_READS_FROM == "front" else w[y0:y1, x0:x1]

    pad = 12
    width = max(lf.shape[1], rf.shape[1], hd.shape[1]) + 2 * pad
    height = lf.shape[0] + rf.shape[0] + hd.shape[0] + 4 * pad
    sheet = np.full((height, width, 3), 90, np.uint8)
    y = pad
    for block in (lf, rf, hd):
        sheet[y:y + block.shape[0], pad:pad + block.shape[1]] = block
        y += block.shape[0] + pad
    Image.fromarray(sheet).save("1sat_race_verify.png")
    print("wrote 1sat_race_verify.png (left flank, right flank, hood-from-front)")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, TRIM, "example", OUT_NAME)
    painted = build(os.path.join(root, TRIM, "template.png"), out)

    art = np.array(Image.open(out).convert("RGB"))[painted]
    for name, color in (("black", BLACK), ("yellow", YELLOW), ("white", WHITE)):
        n = (np.abs(art.astype(int) - color).sum(axis=1) < 30).sum()
        print(f"  {name:6s} {n / len(art):6.1%} of painted area")
    print(f"{TRIM}: {os.path.getsize(out):,} bytes")
    if "--verify" in sys.argv:
        verify(out)


if __name__ == "__main__":
    main()
