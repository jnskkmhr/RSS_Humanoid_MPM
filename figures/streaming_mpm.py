"""Fig. 1: streaming MPM schematic (particle streaming, tile cache, adaptive particle activation).

Regenerate from the repository root:

    uv run --with matplotlib --with numpy python figures/streaming_mpm.py

Writes figures/streaming_mpm.pdf (vector, included by the paper). Pass a second path to also
write a PNG preview:  ... figures/streaming_mpm.py preview.png

The drawing is laid out in centimetres: one data unit is one centimetre at \\textwidth (18.2 cm).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

# ----------------------------------------------------------------------------- style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

W, H = 18.2, 4.95  # figure size [cm]
CM = 1.0 / 2.54

NAVY = "#1B4F72"  # active particle
GRAY = "#B9C2CB"  # inactive or unloaded particle
SAND = "#FAF6EE"  # tile fill
GRID = "#CFC8BA"  # tile lines
EDGE = "#8F897D"  # terrain outline
TRACK = "#E6D8BD"  # deformed track
CUR = "#D62728"  # tracked body, current
PREV = "#F4A582"  # tracked body, previous
GREEN = "#2E8B57"
GREEN_TINT = "#E1F0DE"
RED = "#C0392B"
VIOLET = "#7B5EA7"
VIOLET_TINT = "#EFEAF6"
INK = "#222222"
MUTED = "#6B665C"

FS = 6.6  # label font size [pt]
FS_TITLE = 7.6

fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


# ----------------------------------------------------------------------------- helpers
def lattice(x0, y0, w, h, nx, ny, rng, jitter=0.27):
    """Jittered lattice of particle positions inside a rectangle."""
    sx, sy = w / nx, h / ny
    gx = x0 + (np.arange(nx) + 0.5) * sx
    gy = y0 + (np.arange(ny) + 0.5) * sy
    X, Y = np.meshgrid(gx, gy)
    X = X + rng.uniform(-jitter, jitter, X.shape) * sx
    Y = Y + rng.uniform(-jitter, jitter, Y.shape) * sy
    return X.ravel(), Y.ravel()


def dots(x, y, r, color, z=4):
    if len(x) == 0:
        return
    ax.add_collection(
        PatchCollection([Circle((a, b), r) for a, b in zip(x, y)], facecolor=color, edgecolor="none", zorder=z)
    )


def rect(x, y, w, h, fc="none", ec="none", lw=0.6, ls="-", z=1, **kw):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z, **kw)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color=INK, lw=0.8, rad=0.0, ls="-", z=6, scale=6.5, style="-|>"):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=scale,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(a)
    return a


def text(x, y, s, size=FS, color=INK, ha="center", va="center", weight="normal", style="normal", z=8, **kw):
    return ax.text(
        x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style, zorder=z, **kw
    )


def text_width(s, size=FS, weight="normal"):
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    fig.canvas.draw()
    w = t.get_window_extent().width / fig.dpi * 2.54
    t.remove()
    return w


def lw_cm(width_cm):
    """Line width in points for a stroke that is `width_cm` wide."""
    return width_cm / 2.54 * 72.0


Y0 = 0.55  # bottom of the panel drawings
YT = 3.65  # top of the panel drawings
Y_HEAD = 3.90  # column headers
Y_TITLE = 0.17  # panel titles

# ============================================================================= (a) particle streaming
u = 0.62  # tile size
AX0 = 0.25
NC, NR = 8, 5
rng = np.random.default_rng(3)

rect(AX0, Y0, NC * u, NR * u, fc=SAND, z=0)
# tiles that leave (unloaded) and enter (loaded)
rect(AX0 + 2 * u, Y0 + u, u, 3 * u, fc="#ECEAE6", z=1)
rect(AX0 + 5 * u, Y0 + u, u, 3 * u, fc=GREEN_TINT, z=1)
for i in range(NC + 1):
    ax.plot([AX0 + i * u] * 2, [Y0, Y0 + NR * u], color=GRID, lw=0.45, zorder=2)
for j in range(NR + 1):
    ax.plot([AX0, AX0 + NC * u], [Y0 + j * u] * 2, color=GRID, lw=0.45, zorder=2)
rect(AX0, Y0, NC * u, NR * u, ec=EDGE, lw=0.7, z=3)

yc = Y0 + 2.5 * u  # row of the tracked body
x_prev = AX0 + 3.5 * u
x_cur = AX0 + 4.5 * u
hw = 0.085  # half width of the track left by the body

# deformed track behind the body
ax.plot(
    [AX0 + 2 * u, x_cur],
    [yc, yc],
    color=TRACK,
    lw=lw_cm(2 * hw),
    solid_capstyle="butt",
    zorder=2.5,
)
px, py = lattice(AX0 + 2 * u, Y0 + u, 4 * u, 3 * u, 20, 15, rng)
keep = ~((np.abs(py - yc) < hw + 0.03) & (px < x_cur + 0.08))
px, py = px[keep], py[keep]
leaving = px < AX0 + 3 * u
dots(px[leaving], py[leaving], 0.033, GRAY)
dots(px[~leaving], py[~leaving], 0.033, NAVY)

# pool at the previous and the current step
rect(AX0 + 2 * u, Y0 + u, 3 * u, 3 * u, ec="#6E6E6E", lw=0.9, ls=(0, (3, 2)), z=5)
rect(AX0 + 3 * u, Y0 + u, 3 * u, 3 * u, ec="black", lw=1.3, z=5.5)

# tracked body
arrow((x_prev + 0.10, yc), (x_cur - 0.10, yc), lw=0.8, z=7, scale=5.5)
ax.add_patch(Circle((x_prev, yc), 0.085, facecolor=PREV, edgecolor="white", linewidth=0.5, zorder=7))
ax.add_patch(Circle((x_cur, yc), 0.085, facecolor=CUR, edgecolor="white", linewidth=0.5, zorder=7))

# recycling of particles from the leaving tiles into the entering tiles
y_pool_top = Y0 + 4 * u
arrow(
    (AX0 + 2.5 * u, y_pool_top + 0.05),
    (AX0 + 5.5 * u, y_pool_top + 0.05),
    color=GREEN,
    lw=0.9,
    rad=-0.52,
    ls=(0, (3, 1.6)),
    z=7,
)
text(AX0 + 4 * u, YT + 0.22, "recycle particles", color=GREEN)
text(AX0 + 2.5 * u, Y0 + 0.5 * u, "unload", color=MUTED)
text(AX0 + 5.5 * u, Y0 + 0.5 * u, "load", color=GREEN)
text(AX0 + 7 * u, Y0 + 2.5 * u, "not\nsimulated", color="#9A9488", style="italic", linespacing=1.05)
text(AX0 + 1 * u, Y0 + 2.5 * u, "not\nsimulated", color="#9A9488", style="italic", linespacing=1.05)

text(AX0 + NC * u / 2, Y_TITLE, "(a) Particle streaming", size=FS_TITLE, weight="bold")

# ============================================================================= (b) tile cache
BX0 = 6.42  # left tile
S = 1.15  # tile size in this panel
BX1 = 10.62  # right tile
ROW1 = YT - S  # with cache
ROW2 = Y0  # without cache


def tile(x0, y0, s, seed, track, n=8, r=0.04, lw=0.9, ec="#4A4A4A"):
    """One terrain tile seen from above, with or without a deformed track."""
    box = rect(x0, y0, s, s, fc=SAND, ec="none", z=1)
    a = np.array([x0, y0 + 0.30 * s])
    b = np.array([x0 + s, y0 + 0.74 * s])
    half = 0.105 * s / 1.15
    local = np.random.default_rng(seed)
    x, y = lattice(x0, y0, s, s, n, n, local)
    if track:
        (line,) = ax.plot(
            [a[0] - 0.2, b[0] + 0.2],
            [a[1] - 0.2 * (b[1] - a[1]) / s, b[1] + 0.2 * (b[1] - a[1]) / s],
            color=TRACK,
            lw=lw_cm(2 * half),
            solid_capstyle="butt",
            zorder=2,
        )
        line.set_clip_path(box)
        d = b - a
        dist = np.abs(d[0] * (y - a[1]) - d[1] * (x - a[0])) / np.hypot(*d)
        x, y = x[dist > half + 0.035], y[dist > half + 0.035]
    dots(x, y, r, NAVY)
    rect(x0, y0, s, s, ec=ec, lw=lw, z=5)


# column headers and row labels
text(BX0 + S / 2, Y_HEAD, "Leaving tile")
text(BX1 + S / 2, Y_HEAD, "Re-entering tile")
text(BX0 - 0.22, ROW1 + S / 2, "With cache", rotation=90, weight="bold")
text(BX0 - 0.22, ROW2 + S / 2, "Without cache", rotation=90, weight="bold")

# --- with cache: save, then restore
tile(BX0, ROW1, S, seed=11, track=True)
tile(BX1, ROW1, S, seed=11, track=True)
ym = ROW1 + S / 2
cx, cs = 8.70, 0.50  # cache icon
for k in (2, 1, 0):
    ox = cx + 0.10 * k
    oy = ym - cs / 2 - 0.05 + 0.10 * k - 0.05
    ax.add_patch(
        FancyBboxPatch(
            (ox, oy),
            cs,
            cs,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=VIOLET_TINT,
            edgecolor=VIOLET,
            linewidth=0.8,
            zorder=5 - k * 0.1,
        )
    )
ox, oy = cx, ym - cs / 2 - 0.10
ax.plot([ox + 0.04, ox + cs - 0.04], [oy + 0.17, oy + 0.37], color=VIOLET, lw=1.6, alpha=0.55, zorder=6)
arrow((BX0 + S + 0.08, ym), (cx - 0.08, ym))
arrow((cx + cs + 0.30, ym), (BX1 - 0.08, ym))
text((BX0 + S + cx) / 2, ym + 0.17, "save")
text((cx + cs + 0.30 + BX1) / 2, ym + 0.17, "restore")
text(cx + cs / 2 + 0.10, ROW1 - 0.02, "Tile cache", color=VIOLET, va="top")
text(BX1 + S / 2, ROW1 - 0.17, "deformation kept", color=GREEN)

# --- without cache: discard, then spawn from the template
tile(BX0, ROW2, S, seed=11, track=True)
tile(BX1, ROW2, S, seed=29, track=False)
ym = ROW2 + S / 2
xx = 8.66  # discard mark
arrow((BX0 + S + 0.08, ym), (xx - 0.19, ym))
for sgn in (1, -1):
    ax.plot([xx - 0.09, xx + 0.09], [ym - 0.09 * sgn, ym + 0.09 * sgn], color=RED, lw=1.3, zorder=7)
text((BX0 + S + 0.08 + xx - 0.19) / 2, ym + 0.17, "discard")
ts = 0.56  # template icon
tx0 = 9.06
tile(tx0, ym - ts / 2, ts, seed=5, track=False, n=4, r=0.036, lw=0.8, ec="#4A4A4A")
rect(tx0, ym - ts / 2, ts, ts, ec="#4A4A4A", lw=0.8, ls=(0, (2.2, 1.5)), fc="none", z=5.2)
arrow((tx0 + ts + 0.08, ym), (BX1 - 0.08, ym))
text((tx0 + ts + BX1) / 2, ym + 0.17, "spawn")
text(tx0 + ts / 2, ym - ts / 2 - 0.05, "Template", color=MUTED, va="top")
text(BX1 + S / 2, ROW2 - 0.17, "deformation lost", color=RED)

text((BX0 - 0.35 + BX1 + S) / 2, Y_TITLE, "(b) Tile cache", size=FS_TITLE, weight="bold")

# ============================================================================= (c) adaptive particle activation
CX0 = 12.50
P = YT - Y0  # pool drawn at 3.1 cm
rng = np.random.default_rng(8)
rect(CX0, Y0, P, P, fc=SAND, z=0)
for k in (1, 2):
    ax.plot([CX0 + k * P / 3] * 2, [Y0, Y0 + P], color=GRID, lw=0.45, zorder=1)
    ax.plot([CX0, CX0 + P], [Y0 + k * P / 3] * 2, color=GRID, lw=0.45, zorder=1)

bx, by = CX0 + P / 2, Y0 + P / 2 - 0.08  # tracked body
bw, bh = 0.44, 0.86
rw, rh = 2.05, 2.30  # activation region
ex, ey = 0.80, 1.00  # strained zone (semi-axes)

ax.add_patch(Ellipse((bx, by), 2 * ex, 2 * ey, facecolor=NAVY, edgecolor="none", alpha=0.10, zorder=1.5))
px, py = lattice(CX0, Y0, P, P, 20, 20, rng)
# material pushed away from the body
dx, dy = px - bx, py - by
rr = np.hypot(dx / (bw / 2 + 0.05), dy / (bh / 2 + 0.05))
push = 0.09 * np.exp(-((np.hypot(dx, dy) / 0.55) ** 2))
norm = np.maximum(np.hypot(dx, dy), 1e-6)
px, py = px + push * dx / norm, py + push * dy / norm
dx, dy = px - bx, py - by
under = (np.abs(dx) < bw / 2 + 0.05) & (np.abs(dy) < bh / 2 + 0.05)
px, py, dx, dy = px[~under], py[~under], dx[~under], dy[~under]
in_region = (np.abs(dx) < rw / 2) & (np.abs(dy) < rh / 2)
strained = (dx / ex) ** 2 + (dy / ey) ** 2 < 1.0
active = in_region & strained
dots(px[~active], py[~active], 0.043, GRAY)
dots(px[active], py[active], 0.043, NAVY)

rect(bx - rw / 2, by - rh / 2, rw, rh, ec=RED, lw=0.9, ls=(0, (3, 1.8)), z=5)
rect(CX0, Y0, P, P, ec="black", lw=1.3, z=6)
ax.add_patch(
    FancyBboxPatch(
        (bx - bw / 2, by - bh / 2),
        bw,
        bh,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor="#D9D9D9",
        edgecolor="#4A4A4A",
        linewidth=0.8,
        zorder=6.5,
    )
)
ax.add_patch(Circle((bx, by), 0.085, facecolor=CUR, edgecolor="white", linewidth=0.5, zorder=7))
text(
    bx,
    by + rh / 2,
    "activation region",
    color=RED,
    bbox=dict(boxstyle="round,pad=0.12", facecolor=SAND, edgecolor="none"),
    z=7,
)
text(CX0 + P / 2, Y_HEAD, "Particle pool")

# MPM update applied to the active particles only
FX = 17.02  # centre of the flow column
BWD, BHT = 1.74, 0.38
ys = [2.47, 1.93, 1.39]
labels = ["P2G", "Rheology solve", "G2P"]
GX0, GX1 = FX - BWD / 2 - 0.13, FX + BWD / 2 + 0.13  # group box
GY0, GY1 = ys[-1] - BHT / 2 - 0.13, ys[0] + BHT / 2 + 0.42
ax.add_patch(
    FancyBboxPatch(
        (GX0, GY0),
        GX1 - GX0,
        GY1 - GY0,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor="#F1F5F9",
        edgecolor=NAVY,
        linewidth=0.8,
        zorder=1,
    )
)
text(FX, GY1 - 0.20, "MPM update", color=NAVY, weight="bold")
for yb, lab in zip(ys, labels):
    ax.add_patch(
        FancyBboxPatch(
            (FX - BWD / 2, yb - BHT / 2),
            BWD,
            BHT,
            boxstyle="round,pad=0,rounding_size=0.07",
            facecolor="white",
            edgecolor=INK,
            linewidth=0.7,
            zorder=2,
        )
    )
    text(FX, yb, lab)
for y_a, y_b in zip(ys[:-1], ys[1:]):
    arrow((FX, y_a - BHT / 2), (FX, y_b + BHT / 2), lw=0.7, scale=5.0)
XL = GX0 + 0.06  # left end of the two state labels
y_act = 3.42
ax.add_patch(Circle((XL, y_act), 0.055, facecolor=NAVY, edgecolor="none", zorder=7))
text(XL + 0.13, y_act, "Active particles", ha="left", color=NAVY)
arrow((FX, y_act - 0.15), (FX, GY1), color=NAVY, lw=0.8, scale=5.5)
y_in = 0.76
ax.add_patch(Circle((XL, y_in), 0.055, facecolor=GRAY, edgecolor="none", zorder=7))
text(XL + 0.13, y_in, "Inactive: skipped", ha="left", color=MUTED)

# leaders from the pool to the two particle states
arrow((bx + 0.60, by + 0.34), (XL - 0.09, y_act), color=NAVY, lw=0.6, style="-", z=7)
arrow((CX0 + P - 0.30, Y0 + 0.34), (XL - 0.09, y_in), color="#8A939C", lw=0.6, style="-", z=7)

text((CX0 + GX1) / 2, Y_TITLE, "(c) Adaptive particle activation", size=FS_TITLE, weight="bold")

# ============================================================================= legend
items = [
    ("dots2", "Tracked body (previous, current)"),
    ("pool2", "Particle pool (previous, current)"),
    ("tile", "Terrain tile"),
    ("active", "Active particle"),
    ("inactive", "Inactive or unloaded particle"),
]
SW = {"dots2": 0.42, "pool2": 0.74, "tile": 0.30, "active": 0.14, "inactive": 0.14}
GAP_IN, GAP_OUT = 0.12, 0.48
widths = [SW[k] + GAP_IN + text_width(lab) for k, lab in items]
total = sum(widths) + GAP_OUT * (len(items) - 1)
PAD = 0.22
YL = H - 0.30
x = (W - total) / 2
ax.add_patch(
    FancyBboxPatch(
        (x - PAD, YL - 0.20),
        total + 2 * PAD,
        0.40,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor="white",
        edgecolor="#9A9A9A",
        linewidth=0.6,
        zorder=1,
    )
)
for (kind, lab), wd in zip(items, widths):
    if kind == "dots2":
        ax.add_patch(Circle((x + 0.09, YL), 0.075, facecolor=PREV, edgecolor="none", zorder=7))
        ax.add_patch(Circle((x + 0.32, YL), 0.075, facecolor=CUR, edgecolor="none", zorder=7))
    elif kind == "pool2":
        rect(x, YL - 0.11, 0.32, 0.22, ec="#6E6E6E", lw=0.9, ls=(0, (3, 2)), z=7)
        rect(x + 0.42, YL - 0.11, 0.32, 0.22, ec="black", lw=1.3, z=7)
    elif kind == "tile":
        rect(x, YL - 0.13, 0.26, 0.26, fc=SAND, ec=GRID, lw=0.8, z=7)
    elif kind == "active":
        ax.add_patch(Circle((x + 0.07, YL), 0.06, facecolor=NAVY, edgecolor="none", zorder=7))
    elif kind == "inactive":
        ax.add_patch(Circle((x + 0.07, YL), 0.06, facecolor=GRAY, edgecolor="none", zorder=7))
    text(x + SW[kind] + GAP_IN, YL - 0.005, lab, ha="left")
    x += wd + GAP_OUT

# ----------------------------------------------------------------------------- output
out = Path(__file__).with_suffix(".pdf")
fig.savefig(out)
print("wrote", out)
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], dpi=300)
    print("wrote", sys.argv[1])
