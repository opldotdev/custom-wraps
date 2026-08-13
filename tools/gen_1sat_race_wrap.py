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
import urllib.request

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

# Sponsor marks from the BSV ecosystem, fetched from each project's own site and
# cached outside version control rather than vendored here. They are third-party
# trademarks: fine on your own car, not on anything you sell.
HERE = os.path.dirname(os.path.abspath(__file__))
SPONSORS = {
    "gorillapool": "https://gorillapool.io/logo.svg",
    "nchain": "https://nchain.com/wp-content/uploads/2025/04/nchain-logo.svg",
    "bsv": "https://bsvblockchain.org/wp-content/uploads/2025/10/logo-bsvb.svg",
    "babbage": "https://projectbabbage.com/babb-logo-dark.svg",
}

FONTS = [
    "/mnt/skills/examples/canvas-design/canvas-fonts/BigShoulders-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Panel boxes measured off modely-2025-premium/template.png, as (x0, y0, x1, y1).
HOOD = (375, 110, 644, 338)
DOOR_L, DOOR_R = (82, 348, 258, 594), (762, 350, 937, 595)
REAR_DOOR_L, REAR_DOOR_R = (69, 570, 234, 791), (784, 573, 949, 795)
REAR_BUMPER_L, REAR_BUMPER_R = (55, 885, 173, 1013), (853, 888, 968, 1016)
FENDER_L, FENDER_R = (111, 114, 331, 366), (694, 114, 911, 367)
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


def sponsor(name, width):
    """Fetch a sponsor mark and render it white at the given pixel width.

    Race liveries print sponsor marks in a single color, and white is the only
    one that reads on a black car -- BSV's navy in particular disappears into
    it. Recoloring by alpha rather than by pixel keeps every counter and gap in
    the marks intact instead of flattening them into blobs."""
    cache = os.path.join(HERE, f".sponsor_{name}.svg")
    if not os.path.exists(cache):
        print(f"  fetching {name} from {SPONSORS[name]}")
        try:
            req = urllib.request.Request(
                SPONSORS[name], headers={"User-Agent": "custom-wraps-generator/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r, open(cache, "wb") as f:
                f.write(r.read())
        except Exception as exc:
            raise SystemExit(f"could not fetch the {name} mark ({exc}).\n"
                             f"Download {SPONSORS[name]} to {cache} and re-run.")
    import cairosvg
    png = cache.replace(".svg", ".png")
    cairosvg.svg2png(url=cache, write_to=png, output_width=width * 4, background_color=None)
    im = Image.open(png).convert("RGBA")
    im = im.resize((width, max(1, round(width * im.height / im.width))), Image.LANCZOS)
    white = Image.new("RGBA", im.size, WHITE + (0,))
    white.putalpha(im.getchannel("A"))
    return np.array(white)


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

    def place(block, box, label, side=None, along=None, up=None):
        """Sit a block on its panel and nudge until none of it spills onto
        neighbouring bodywork. Panels are irregular, so the starting point is a
        guess, not an answer.

        Without a spot it centres on the panel's centre of mass. With one, along
        runs 0 at the panel's front edge to 1 at its rear, and up runs 0 at the
        rocker to 1 at the top of the car -- which is +x on the left flank and
        -x on the right, since the two columns peel in opposite directions."""
        panel = panel_at(labels, box)
        if side is None:
            cx, cy = anchor(panel)
        else:
            x0, y0, x1, y1 = box
            fx = up if side == "left" else 1.0 - up
            cx, cy = x0 + fx * (x1 - x0), y0 + along * (y1 - y0)
        bh, bw = block.shape[:2]
        ink = block[..., 3] > 8
        best = None
        span = 28 if side is None else 24
        for dy in range(-span, span + 1, 2):
            for dx in range(-span, span + 1, 2):
                x, y = int(round(cx - bw / 2)) + dx, int(round(cy - bh / 2)) + dy
                if x < 0 or y < 0 or x + bw > W or y + bh > H:
                    continue
                # Score on everything that misses the panel, not just what lands
                # on a neighbour. Pixels that fall into the gaps between panels
                # are painted nowhere and get clipped away by the mask, which
                # silently trims a mark rather than moving it.
                outside = int((ink & ~panel[y:y + bh, x:x + bw]).sum())
                score = (outside, dx * dx + dy * dy)
                if best is None or score < best[0]:
                    best = (score, x, y)
        (outside, _), x, y = best
        over(art, block, x, y)
        print(f"  {label:12s} {bw}x{bh}px, {outside} px off its panel")

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

    # Sponsor marks, scattered rather than stacked. On a real car these are
    # small decals spread over the body -- roughly 20-45cm at this template's
    # ~5mm per pixel -- not one block on a door, which reads as a decal sheet.
    # Each is built for the left flank and turned 180 degrees for the right,
    # the same world mirror the wordmarks use, so every mark sits upright and
    # unmirrored viewed from its own side.
    for name, width, box_l, box_r, along, up in (
            ("nchain", 84, FENDER_L, FENDER_R, 0.72, 0.74),
            ("gorillapool", 76, REAR_DOOR_L, REAR_DOOR_R, 0.26, 0.76),
            ("bsv", 118, REAR_DOOR_L, REAR_DOOR_R, 0.48, 0.16),
            ("babbage", 40, REAR_BUMPER_L, REAR_BUMPER_R, 0.42, 0.62)):
        mark = np.flip(sponsor(name, width).transpose(1, 0, 2), axis=1)
        place(mark, box_l, f"{name} L", "left", along, up)
        place(mark[::-1, ::-1], box_r, f"{name} R", "right", along, up)

    # Roundels on the rear quarters, as competition number discs.
    for box, side in ((QUARTER_L, "left"), (QUARTER_R, "right")):
        r = disc(art, panel_at(labels, box), 52)
        print(f"  roundel {side:5s} radius {r:.0f}px")

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(Image.fromarray(art), (0, 0), mask)
    out.save(out_path, optimize=True)
    return painted


def verify(out_path):
    """Write every lettered panel as it is actually seen, so each wordmark and
    sponsor mark can be checked upright and the right way round. Anything that
    carries a mark belongs here -- the sponsor panels were added blind the first
    time because this only covered the front doors and the hood."""
    w = np.array(Image.open(out_path).convert("RGB"))

    def flank(box, side):
        x0, y0, x1, y1 = box
        block = w[y0:y1, x0:x1].transpose(1, 0, 2)
        return np.flip(block, axis=0 if side == "left" else 1)

    x0, y0, x1, y1 = HOOD
    blocks = [
        flank(DOOR_L, "left"), flank(DOOR_R, "right"),
        flank(REAR_DOOR_L, "left"), flank(REAR_DOOR_R, "right"),
        flank(FENDER_L, "left"), flank(FENDER_R, "right"),
        flank(REAR_BUMPER_L, "left"), flank(REAR_BUMPER_R, "right"),
        w[y0:y1, x0:x1][::-1, ::-1] if HOOD_READS_FROM == "front" else w[y0:y1, x0:x1],
    ]
    pad = 12
    width = max(b.shape[1] for b in blocks) + 2 * pad
    height = sum(b.shape[0] for b in blocks) + pad * (len(blocks) + 1)
    sheet = np.full((height, width, 3), 90, np.uint8)
    y = pad
    for block in blocks:
        sheet[y:y + block.shape[0], pad:pad + block.shape[1]] = block
        y += block.shape[0] + pad
    Image.fromarray(sheet).save("1sat_race_verify.png")
    print("wrote 1sat_race_verify.png (front doors, rear doors, fenders, hood)")


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
