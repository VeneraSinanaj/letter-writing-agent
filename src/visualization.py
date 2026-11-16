"""Drawing functions: the letters and the episodes played by an agent."""
import matplotlib.pyplot as plt
import numpy as np

from .letters import GRID_SIZE, Letter

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID_DOT = "#c3c2b7"
TEMPLATE = "#e4e3df"   # segments of the letter not drawn yet
DRAWN = "#2a78d6"      # segments of the letter drawn by the agent
OFF_LETTER = "#e34948" # segments drawn outside the letter
PEN_UP = "#8a8984"     # moves with the pen up
START = "#eb6834"      # natural starting points
PEN = "#0d366b"


def _setup(ax, title=None):
    ax.set_facecolor(SURFACE)
    rows, cols = np.meshgrid(range(GRID_SIZE), range(GRID_SIZE))
    ax.scatter(cols.ravel(), rows.ravel(), s=8, color=GRID_DOT, zorder=1)
    ax.set_xlim(-0.6, GRID_SIZE - 0.4)
    ax.set_ylim(GRID_SIZE - 0.4, -0.6)  # row 0 at the top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d6d5d0")
    if title:
        ax.set_title(title, fontsize=11, color=TEXT)


def _segment(ax, a, b, **kwargs):
    (r1, c1), (r2, c2) = a, b
    ax.plot([c1, c2], [r1, r2], solid_capstyle="round", **kwargs)


def draw_letter(ax, letter: Letter, show_starts=True, title=None):
    """The letter as a light template, with its natural starting points."""
    _setup(ax, title if title is not None else f"{letter.name}  ({letter.n_segments} segments)")
    for a, b in letter.segments:
        _segment(ax, a, b, color=TEMPLATE, lw=9, zorder=2)
    if show_starts:
        for r, c in letter.start_points:
            ax.plot(c, r, "o", ms=9, mfc=START, mec=SURFACE, mew=1.5, zorder=4)


def draw_episode(ax, letter: Letter, history, title=None, n_steps=None):
    """An episode on top of the letter template. history = env.history (list of info dicts)."""
    draw_letter(ax, letter, show_starts=False, title=title if title is not None else letter.name)
    steps = history if n_steps is None else history[:n_steps]
    moves = [info for info in history if info["event"] == "move"]
    if moves:  # starting point of the episode
        r, c = moves[0]["from"]
        ax.plot(c, r, "o", ms=9, mfc=START, mec=SURFACE, mew=1.5, zorder=4)
    pen_pos = None
    for info in steps:
        if info["event"] != "move":
            continue
        if not info["pen_down"]:
            _segment(ax, info["from"], info["to"], color=PEN_UP, lw=1.2, ls=(0, (2, 2)), zorder=3)
        elif info["segment"] == "off_letter":
            _segment(ax, info["from"], info["to"], color=OFF_LETTER, lw=3, zorder=3)
        else:
            _segment(ax, info["from"], info["to"], color=DRAWN, lw=4, zorder=3)
        pen_pos = info["to"]
    if pen_pos is not None:
        ax.plot(pen_pos[1], pen_pos[0], "o", ms=8, mfc=PEN, mec=SURFACE, mew=1.5, zorder=5)


def plot_episodes(panels, ncols=5, path=None, caption=None):
    """Several episodes side by side. panels = list of (letter, history, title)."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows + 0.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, (letter, history, title) in zip(axes, panels):
        draw_episode(ax, letter, history, title=title)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.text(0.5, 0.01, caption or "orange: start   blue: letter segment drawn   red: outside the letter   "
             "dashed: pen up   dark dot: final pen position", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def animate_episodes(panels, path, ncols=5, fps=4):
    """GIF of several episodes played step by step at the same time (panels as in plot_episodes)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows + 0.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    n_frames = max(len(h) for _, h, _ in panels) + 1

    def update(frame):
        for ax, (letter, history, title) in zip(axes, panels):
            ax.clear()
            draw_episode(ax, letter, history, title=title, n_steps=min(frame, len(history)))
        return []

    fig.tight_layout()
    hold = 2 * fps  # keep the final image for about 2 seconds
    anim = FuncAnimation(fig, update, frames=list(range(n_frames)) + [n_frames - 1] * hold, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_letters(letters, ncols=5, path=None):
    """All the letters side by side, with their starting points."""
    nrows = int(np.ceil(len(letters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows + 0.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, letter in zip(axes, letters):
        draw_letter(ax, letter)
    for ax in axes[len(letters):]:
        ax.axis("off")
    fig.text(0.5, 0.01, "orange dots: natural starting points", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig
