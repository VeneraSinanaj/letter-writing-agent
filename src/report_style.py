"""
Figure style for the report: dark, muted colours, Times font, white background, no titles inside the figures
(the caption of the report gives the title). Used by notebooks/10_report_figures.ipynb.
The notebooks 01-09 keep their own, brighter style (src/visualization.py), useful on screen.
"""
import matplotlib.pyplot as plt
import numpy as np

from .letters import GRID_SIZE

# ------------------------------------------------------------------ palette
NAVY = "#1f3a5f"     # main colour: the agent's drawing, the most important series
TEAL = "#2a6f6b"
PLUM = "#6b4c7a"
OCHRE = "#9c7a2b"
SLATE = "#6b7280"
WINE = "#8b2e3c"     # errors only (segments drawn outside the letter)
SERIES = [NAVY, TEAL, PLUM, OCHRE, SLATE]
LINESTYLES = ["-", "--", ":", "-."]

TEXT = "#22262c"
AXIS = "#3a3f47"
GRID_LINE = "#e6e8eb"
TEMPLATE = "#dde3ea"   # segments of the letter not drawn yet
LETTER = "#7d8ea3"     # letters shown on their own (figure of the alphabet)
DOT = "#b8bec7"
FRAME = "#c9ced6"

WIDTH = 7.0            # inches: full text width of an A4 report, figures are scaled to ~16 cm


def apply():
    """Global matplotlib settings for the report figures."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
        "mathtext.fontset": "stix",   # formulas in a Times-like font
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": TEXT, "axes.titlesize": 9,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def axes_style(ax, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_LINE, lw=0.6)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ------------------------------------------------------------------ drawings on the grid
def _grid(ax, framed_color=FRAME, frame_width=0.6):
    r, c = np.meshgrid(range(GRID_SIZE), range(GRID_SIZE))
    ax.scatter(c.ravel(), r.ravel(), s=3, color=DOT, zorder=1)
    ax.set_xlim(-0.6, GRID_SIZE - 0.4)
    ax.set_ylim(GRID_SIZE - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(framed_color)
        spine.set_linewidth(frame_width)


def draw_letter(ax, letter, title=None, highlight=False):
    """A letter on its own, with its natural starting points (white rings)."""
    _grid(ax, NAVY if highlight else FRAME, 1.2 if highlight else 0.6)
    for (r1, c1), (r2, c2) in letter.segments:
        ax.plot([c1, c2], [r1, r2], color=LETTER, lw=3.2, solid_capstyle="round", zorder=2)
    for r, c in letter.start_points:
        ax.plot(c, r, "o", ms=5.5, mfc="white", mec=TEAL, mew=1.3, zorder=3)
    if title:
        ax.set_title(title, color=TEXT, pad=3)


def draw_episode(ax, letter, history, title=None):
    """An episode on top of the letter: drawn segments in navy, errors in wine, pen-up moves dotted."""
    _grid(ax)
    for (r1, c1), (r2, c2) in letter.segments:
        ax.plot([c1, c2], [r1, r2], color=TEMPLATE, lw=6, solid_capstyle="round", zorder=2)
    moves = [info for info in history if info["event"] == "move"]
    for info in moves:
        (r1, c1), (r2, c2) = info["from"], info["to"]
        if not info["pen_down"]:
            ax.plot([c1, c2], [r1, r2], color=SLATE, lw=0.8, ls=(0, (2, 2)), zorder=3)
        elif info["segment"] == "off_letter":
            ax.plot([c1, c2], [r1, r2], color=WINE, lw=1.9, zorder=3)
        else:
            ax.plot([c1, c2], [r1, r2], color=NAVY, lw=2.1, solid_capstyle="round", zorder=4)
    if moves:
        r0, c0 = moves[0]["from"]
        ax.plot(c0, r0, "o", ms=6.5, mfc="white", mec=TEAL, mew=1.4, zorder=5)
        r1, c1 = moves[-1]["to"]
        ax.plot(c1, r1, "o", ms=3.8, color=NAVY, zorder=6)
    if title:
        ax.set_title(title, color=TEXT, pad=3)


def panel_grid(n, ncols, panel=0.95):
    """A figure with n small square panels (unused panels hidden)."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel, nrows * (panel + 0.18)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


DRAWING_LEGEND = ("white ring: start   ·   navy: segment drawn   ·   wine: segment outside the letter   ·   "
                  "dotted: pen up   ·   navy dot: final position")


def add_caption_line(fig, text=DRAWING_LEGEND):
    fig.text(0.5, -0.01, text, ha="center", va="top", fontsize=7.5, color=SLATE)


# ------------------------------------------------------------------ blank sheet (step 9)
def draw_canvas(ax, target, zone, history, title=None):
    """A drawing on the 28x28 canvas: zone (very light), skeleton (light blue-grey), ink in the zone (navy),
    ink outside the zone (wine), pen-up moves (dotted), start (white ring), final position (navy dot)."""
    from matplotlib.colors import to_rgb
    from .free_draw import line_pixels
    size = target.shape[0]
    img = np.ones((size, size, 3))
    img[zone] = to_rgb("#eef1f5")
    img[target] = to_rgb(TEMPLATE)
    moves = [info for info in history if info.get("event") == "move"]
    for info in moves:
        if info["pen_down"]:
            for p in line_pixels(info["start"], info["end"]):
                img[p] = to_rgb(NAVY if zone[p] else WINE)
    ax.imshow(img, interpolation="nearest")
    for info in moves:
        if not info["pen_down"]:
            (r1, c1), (r2, c2) = info["start"], info["end"]
            ax.plot([c1, c2], [r1, r2], color=SLATE, lw=0.7, ls=(0, (2, 2)))
    if moves:
        r0, c0 = moves[0]["start"]
        ax.plot(c0, r0, "o", ms=6, mfc="white", mec=TEAL, mew=1.3)
        r1, c1 = moves[-1]["end"]
        ax.plot(c1, r1, "o", ms=3.5, color=NAVY)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME)
        spine.set_linewidth(0.6)
    if title:
        ax.set_title(title, color=TEXT, pad=3)


CANVAS_LEGEND = ("white ring: start   ·   navy: ink on the letter   ·   wine: ink outside   ·   "
                 "dotted: pen up   ·   navy dot: final position")
