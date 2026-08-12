#!/usr/bin/env python3
"""Generate the Miami Barbie beach wrap for the Model Y (2025+) Premium template.

A sunset beach scene wrapped along both flanks: palm trees over sand and ocean
under a Miami sunset, with a sunbather reclining across the left side of the
car and the Barbie wordmark across the right.

Each flank is painted as one continuous scene rather than panel by panel, so
the horizon and the palms run unbroken from the front fender to the rear
bumper the way an actual printed wrap would.

Orientation, which follows from the template being a UV unwrap seen from above
with the nose at the top of the image:

* The car's left side is the left of the image, and on a side panel the edge
  nearer the middle of the image is the top of the car.
* A flank scene is drawn the way that flank is actually seen -- sky at the top,
  sand at the bottom, length of the car across. The two flanks peel in opposite
  directions, so each maps onto its column with a different transform, and the
  nose ends up at opposite ends of the two scenes. Getting this wrong is what
  leaves lettering reading backwards on one side of the car.
* Positions along the car are therefore given as a fraction from the nose, and
  each flank resolves that to its own pixel column.

The wordmark is the Barbie logotype, fetched as vector outlines from Wikimedia
Commons, where it is held to be below the threshold of originality and so not
copyrightable. It remains a Mattel trademark, which is fine for a wrap on your
own car but is not something to put on anything you sell. It is downloaded on
demand rather than vendored into this repository.

Usage:  python3 tools/gen_barbie_beach_wrap.py [--verify]
"""
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TRIM = "modely-2025-premium"
OUT_NAME = "Miami_Barbie_Beach.png"

LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/0/0b/Barbie_Logo.svg"
LOGO_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".barbie_logo.svg")

# Barbie pink is taken from the logo file itself; the rest is a Miami sunset.
PINK = (236, 67, 153)
HOT = (255, 105, 170)
ORANGE = (255, 150, 74)
GOLD = (255, 214, 94)
TEAL = (38, 194, 200)
DEEP_TEAL = (16, 104, 136)
PURPLE = (146, 48, 168)
NIGHT = (44, 26, 88)
SAND = (255, 226, 192)
SHADOW = (58, 20, 74)
WHITE = (255, 255, 255)

# Sunset along the length of the car, nose to tail. Used on the panels that are
# not part of a flank scene: the fascia, hood, roof rails and tail.
SUNSET = [(0.00, GOLD), (0.14, ORANGE), (0.34, HOT), (0.52, PINK),
          (0.74, PURPLE), (1.00, NIGHT)]

# Flank extents in template pixels, (x0, y0, x1, y1).
FLANKS = {"left": (55, 114, 341, 1013), "right": (679, 114, 965, 1016)}
HOOD = (375, 110, 644, 338)

HORIZON, SHORE = 0.40, 0.66  # as fractions of flank height
SS = 2  # supersample for the hand-drawn scenery


def to_flank(scene, side):
    """Map a scene drawn as the flank is seen onto that flank's column."""
    m = np.asarray(scene).transpose(1, 0, 2)
    return np.flip(m, axis=1) if side == "left" else np.flip(m, axis=0)


def along(side, s, length):
    """Column in a flank scene for position s along the car, nose at s=0."""
    return (s if side == "left" else 1.0 - s) * (length - 1)


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def ramp(stops, t):
    if t <= stops[0][0]:
        return stops[0][1]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            return lerp(c0, c1, (t - t0) / (t1 - t0))
    return stops[-1][1]


def vertical_ramp(draw, x0, x1, y0, y1, stops):
    for y in range(int(y0), int(y1)):
        draw.line([(x0, y), (x1, y)], fill=ramp(stops, (y - y0) / max(1, y1 - y0)))


def capsule(d, p0, p1, w, color):
    """A thick line with round ends -- the building block for the figure."""
    d.line([p0, p1], fill=color, width=int(w))
    for x, y in (p0, p1):
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=color)


def bezier(p0, p1, p2, n=44):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / (n - 1) for i in range(n))]


def ribbon(d, spine, widths, color):
    """Fill a tapered band either side of a curve: trunks and palm fronds."""
    left, right = [], []
    for i, (x, y) in enumerate(spine):
        j, k = min(i + 1, len(spine) - 1), max(i - 1, 0)
        dx, dy = spine[j][0] - spine[k][0], spine[j][1] - spine[k][1]
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny, half = -dy / n, dx / n, widths[i] / 2
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    d.polygon(left + right[::-1], fill=color)


def palm(d, x, ground, height, lean, color):
    n = 44
    spine = bezier((x, ground), (x + lean * 0.35, ground - height * 0.62),
                   (x + lean, ground - height), n)
    ribbon(d, spine, [height * 0.048 * (1 - 0.6 * i / (n - 1)) for i in range(n)], color)
    cx, cy = spine[-1]
    for k in range(8):
        a = -2.85 + k * 0.66
        reach = height * (0.40 if k % 2 else 0.50)
        mid = (cx + reach * 0.55 * np.cos(a), cy + reach * 0.55 * np.sin(a) - reach * 0.18)
        tip = (cx + reach * np.cos(a), cy + reach * np.sin(a) + reach * 0.34)
        widths = [height * 0.085 * np.sin(np.pi * (i / (n - 1)) ** 0.7) + 1 for i in range(n)]
        ribbon(d, bezier((cx, cy), mid, tip, n), widths, color)
    for dx, dy in ((-0.045, 0.04), (0.045, 0.06), (0, 0.10)):
        r = height * 0.028
        d.ellipse([cx + dx * height - r, cy + dy * height - r,
                   cx + dx * height + r, cy + dy * height + r], fill=color)


def sunbather(d, cx, cy, k):
    """A reclining figure in flat silhouette, hair and swimsuit picked out in
    color. The design is laid out about 340 wide by 130 tall, then scaled by k
    and centred on cx, cy. Head toward the nose of the car."""
    def p(a, b):
        return (cx + (a - 170) * k, cy + (b - 64) * k)

    d.rounded_rectangle([p(-34, 104), p(334, 128)], radius=12 * k, fill=WHITE)
    for i in range(8):
        d.rectangle([p(-24 + i * 48, 104), p(-2 + i * 48, 128)], fill=HOT)

    capsule(d, p(168, 100), p(300, 110), 30 * k, SHADOW)      # far leg
    capsule(d, p(70, 56), p(166, 96), 46 * k, SHADOW)         # torso
    capsule(d, p(166, 96), p(236, 46), 34 * k, SHADOW)        # thigh
    capsule(d, p(236, 46), p(288, 104), 24 * k, SHADOW)       # calf
    d.ellipse([p(280, 94), p(306, 116)], fill=SHADOW)         # foot
    capsule(d, p(74, 62), p(40, 100), 18 * k, SHADOW)         # propping arm
    capsule(d, p(58, 50), p(74, 60), 17 * k, SHADOW)          # neck

    d.ellipse([p(8, 42), p(68, 100)], fill=GOLD)              # hair behind
    d.ellipse([p(28, 20), p(72, 64)], fill=SHADOW)            # head
    d.ellipse([p(10, 28), p(44, 72)], fill=GOLD)              # hair in front
    d.ellipse([p(-4, 4), p(94, 40)], fill=HOT)                # sun hat brim
    d.ellipse([p(26, 0), p(70, 30)], fill=PINK)               # hat crown

    d.polygon([p(96, 52), p(128, 46), p(122, 80), p(92, 78)], fill=PINK)
    d.polygon([p(150, 78), p(188, 70), p(192, 102), p(152, 104)], fill=PINK)


def sliced_sun(w, h, cx, cy, r, cut_below=None):
    """A synthwave sun: a vertical gradient disc with slots that widen downward,
    optionally cut off at a horizon."""
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    rgba = np.zeros((h, w, 4), np.uint8)
    t = np.clip((yy - (cy - r)) / (2 * r), 0, 1)
    # White-hot at the top through gold into pink, so the disc separates from
    # the sky behind it instead of blending into the same gradient.
    core, mid = (255, 250, 224), GOLD
    grad = np.where(t[..., None] < 0.5,
                    np.stack([core[i] + (mid[i] - core[i]) * (t / 0.5) for i in range(3)], axis=2),
                    np.stack([mid[i] + (HOT[i] - mid[i]) * ((t - 0.5) / 0.5) for i in range(3)], axis=2))
    rgba[..., :3] = grad.astype(np.uint8)
    rgba[..., 3] = inside * 255
    for i in range(9):
        y0 = cy - r * 0.52 + i * r * 0.17
        rgba[(yy >= y0) & (yy < y0 + 1.5 + i * 1.7), 3] = 0
    if cut_below is not None:
        rgba[yy > cut_below, 3] = 0
    return rgba


def flank_scene(length, height, side, wordmark):
    """One flank of the car, drawn as it is seen: length runs along the car."""
    L, H = length * SS, height * SS
    horizon, shore = int(H * HORIZON), int(H * SHORE)
    img = Image.new("RGBA", (L, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    vertical_ramp(d, 0, L, 0, horizon, [(0.0, NIGHT), (0.28, PURPLE), (0.60, PINK),
                                        (0.84, ORANGE), (1.0, GOLD)])
    vertical_ramp(d, 0, L, horizon, shore, [(0.0, ORANGE), (0.16, TEAL), (1.0, DEEP_TEAL)])
    vertical_ramp(d, 0, L, shore, H, [(0.0, (255, 234, 202)), (1.0, (247, 172, 146))])

    sx, r = along(side, 0.10, L), H * 0.40
    img.alpha_composite(Image.fromarray(sliced_sun(L, H, sx, horizon, r, cut_below=horizon)))
    d = ImageDraw.Draw(img)

    for i in range(8):  # glitter path on the water beneath the sun
        y = horizon + 3 + i * (shore - horizon) / 8
        half = r * (0.14 + i * 0.085)
        d.rounded_rectangle([sx - half, y, sx + half, y + 2.5 * SS],
                            radius=2 * SS, fill=(255, 240, 210))

    for s, hgt, lean in ((0.135, 0.58, -0.19), (0.70, 0.52, 0.16), (0.92, 0.38, -0.11)):
        palm(d, along(side, s, L), H * 0.92, H * hgt, lean * H, SHADOW)

    if side == "left":
        sunbather(d, along(side, 0.50, L), H * 0.66, H * 0.0031)
    else:
        target_h = int(H * 0.52)
        mark = wordmark.resize((int(target_h * wordmark.width / wordmark.height), target_h),
                               Image.LANCZOS)
        halo = mark.getchannel("A").filter(ImageFilter.MaxFilter(9))
        plate = Image.new("RGBA", mark.size, WHITE + (0,))
        plate.putalpha(halo)
        px = int(along(side, 0.50, L) - mark.width / 2)
        py = int(H * 0.40 - mark.height / 2)
        img.alpha_composite(plate, (px, py))
        img.alpha_composite(mark, (px, py))

    return img.resize((length, height), Image.LANCZOS)


def load_wordmark():
    if not os.path.exists(LOGO_CACHE):
        print(f"fetching wordmark from {LOGO_URL}")
        try:
            # Wikimedia refuses requests that do not identify themselves.
            req = urllib.request.Request(
                LOGO_URL, headers={"User-Agent": "custom-wraps-generator/1.0"})
            with urllib.request.urlopen(req) as r, open(LOGO_CACHE, "wb") as f:
                f.write(r.read())
        except Exception as exc:
            raise SystemExit(f"could not fetch the wordmark ({exc}).\n"
                             f"Download {LOGO_URL} to {LOGO_CACHE} and re-run.")
    import cairosvg
    png = LOGO_CACHE.replace(".svg", ".png")
    cairosvg.svg2png(url=LOGO_CACHE, write_to=png, output_width=1600, background_color=None)
    return Image.open(png).convert("RGBA")


def build(template_path, out_path):
    tpl = Image.open(template_path).convert("RGBA")
    W, H = tpl.size
    mask = tpl.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(3))

    art = Image.new("RGB", (W, H))
    vertical_ramp(ImageDraw.Draw(art), 0, W, 0, H, SUNSET)

    wordmark = load_wordmark()
    for side, (x0, y0, x1, y1) in FLANKS.items():
        scene = flank_scene(y1 - y0, x1 - x0, side, wordmark)
        block = to_flank(scene, side)
        art.paste(Image.fromarray(block[..., :3]), (x0, y0),
                  Image.fromarray(block[..., 3], "L"))

    # A sun on the hood as well, so the nose reads as sunset from the front.
    hw, hh = HOOD[2] - HOOD[0], HOOD[3] - HOOD[1]
    sun = sliced_sun(hw, hh, hw / 2, hh * 0.52, min(hw, hh) * 0.44)
    art.paste(Image.fromarray(sun[..., :3]), (HOOD[0], HOOD[1]),
              Image.fromarray(sun[..., 3], "L"))

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(art, (0, 0), mask)
    out.save(out_path, optimize=True)
    return np.array(mask) > 128


def verify(out_path):
    """Write both flanks as they are actually seen, to check the scene sits the
    right way up and the wordmark reads correctly."""
    w = np.array(Image.open(out_path).convert("RGB"))
    views = []
    for side, (x0, y0, x1, y1) in FLANKS.items():
        block = w[y0:y1, x0:x1].transpose(1, 0, 2)
        views.append(np.flip(block, axis=0) if side == "left" else np.flip(block, axis=1))
    pad = 12
    width = max(v.shape[1] for v in views) + 2 * pad
    height = sum(v.shape[0] for v in views) + (len(views) + 1) * pad
    sheet = np.full((height, width, 3), 90, np.uint8)
    y = pad
    for v in views:
        sheet[y:y + v.shape[0], pad:pad + v.shape[1]] = v
        y += v.shape[0] + pad
    Image.fromarray(sheet).save("barbie_verify.png")
    print("wrote barbie_verify.png (left flank, right flank, as seen)")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, TRIM, "example", OUT_NAME)
    build(os.path.join(root, TRIM, "template.png"), out)
    print(f"{TRIM}: {os.path.getsize(out):,} bytes")
    if "--verify" in sys.argv:
        verify(out)


if __name__ == "__main__":
    main()
