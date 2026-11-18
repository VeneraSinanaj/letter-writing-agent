"""
"Blank sheet" environment (step 9): the agent draws freely on a 28x28 pixel canvas.

Compared with the 5x5 grid of LetterWritingEnv:
- the pen is on any pixel of the canvas, not on 25 points;
- the target is an IMAGE of the letter, not a list of segments;
- a move goes in one of 16 directions with one of 3 lengths (1, 2 or 5 pixels), pen up or pen down;
  some short moves fall on the same pixel, which leaves 40 distinct moves -> 40 x 2 + STOP = 81 actions;
- the reward measures the resemblance between the drawing and the target image.

Target image: the letter's strokes are drawn as 1-pixel lines (the "skeleton"). A drawing is judged with a
tolerance of 1 pixel: a skeleton pixel is "covered" when the agent has inked a pixel at most 1 pixel away,
and an inked pixel is "good" when it lies in the target "zone" (skeleton widened by 1 pixel).

Reward (same spirit as the grid):
    +0.2 per newly covered skeleton pixel      (a grid segment is 5 pixels long -> about +1 per segment)
    -0.2 per newly inked pixel outside the zone
    -0.05 per move, -0.3 per new stroke, -0.2 for trying to leave the canvas
    STOP: +2 if at least 95 % of the skeleton is covered, and -0.2 per skeleton pixel still uncovered
"""
from dataclasses import dataclass

import numpy as np

from .letters import GRID_SIZE, Letter, make_letter

SIZE = 28                 # canvas: 28 x 28 pixels
MARGIN, SPACING = 4, 5    # grid point (r, c) of the letters -> pixel (4 + 5r, 4 + 5c)
LENGTHS = (1, 2, 5)       # move lengths in pixels
N_DIRECTIONS = 16


def _direction_vectors():
    """16 directions every 22.5 degrees, scaled so that the largest coordinate is 1
    (so that a 45-degree move of length 5 goes exactly from one grid point to the next)."""
    vecs = []
    for k in range(N_DIRECTIONS):
        angle = 2 * np.pi * k / N_DIRECTIONS
        dr, dc = -np.cos(angle), np.sin(angle)          # k = 0: up, then clockwise
        scale = max(abs(dr), abs(dc))
        vecs.append((dr / scale, dc / scale))
    return vecs


DIRECTIONS = _direction_vectors()
MOVES, MOVE_LABELS = [], []           # distinct moves (row change, col change) and (angle, length)
for k, (dr, dc) in enumerate(DIRECTIONS):
    for L in LENGTHS:
        move = (int(round(L * dr)), int(round(L * dc)))
        if move not in MOVES:
            MOVES.append(move)
            MOVE_LABELS.append((360 * k / N_DIRECTIONS, L))
N_MOVES = len(MOVES)                  # 40
STOP = 2 * N_MOVES                    # actions 0-39: pen up, 40-79: pen down, 80: STOP
N_ACTIONS = STOP + 1                  # 81


def action_name(a):
    if a == STOP:
        return "STOP"
    angle, L = MOVE_LABELS[a % N_MOVES]
    return f"{angle:5.1f}° x {L} px ({'pen down' if a >= N_MOVES else 'pen up'})"


def to_pixel(cell):
    r, c = cell
    return MARGIN + SPACING * r, MARGIN + SPACING * c


def line_pixels(p0, p1):
    """Pixels of the straight line from p0 to p1 (Bresenham), both ends included."""
    (r0, c0), (r1, c1) = p0, p1
    pixels = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    r, c = r0, c0
    while True:
        pixels.append((r, c))
        if (r, c) == (r1, c1):
            return pixels
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def dilate(mask):
    """Widen a boolean image by 1 pixel in every direction (3x3 neighbourhood)."""
    out = mask.copy()
    padded = np.pad(mask, 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[1 + dr:1 + dr + SIZE, 1 + dc:1 + dc + SIZE]
    return out


def letter_image(letter: Letter):
    """Skeleton image (1-pixel lines) of a letter defined on the 5x5 grid."""
    img = np.zeros((SIZE, SIZE), dtype=bool)
    for a, b in letter.segments:
        for r, c in line_pixels(to_pixel(a), to_pixel(b)):
            img[r, c] = True
    return img


@dataclass(frozen=True)
class FreeRewards:
    cover: float = 0.2          # per newly covered skeleton pixel
    off_zone: float = -0.2      # per newly inked pixel outside the target zone
    move: float = -0.05
    new_stroke: float = -0.3
    off_canvas: float = -0.2
    stop_bonus: float = 2.0     # STOP with at least `complete` of the skeleton covered
    stop_missing: float = -0.2  # STOP: per skeleton pixel still uncovered
    complete: float = 0.95


class FreeDrawEnv:
    def __init__(self, letter, rewards: FreeRewards = FreeRewards(), max_steps: int = 60, seed=None):
        self.letter = make_letter(letter) if isinstance(letter, str) else letter
        self.rewards = rewards
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.target = letter_image(self.letter)          # skeleton
        self.zone = dilate(self.target)                   # tolerance zone
        self.n_target = int(self.target.sum())
        self.n_actions = N_ACTIONS
        self.start_pixels = [to_pixel(cell) for cell in self.letter.start_points]

    # ------------------------------------------------------------------ episode
    def reset(self, start=None):
        """New episode: pen up on a natural starting point (random if not given), blank canvas."""
        if start is None:
            start = self.start_pixels[self.rng.integers(len(self.start_pixels))]
        self.pen = tuple(start)
        self.pen_down = False
        self.ink = np.zeros((SIZE, SIZE), dtype=bool)      # what the agent has drawn
        self.covered = np.zeros((SIZE, SIZE), dtype=bool)  # skeleton pixels already covered
        self.steps = 0
        self.history = []
        return self.observation()

    def observation(self):
        """4 images of 28x28: target, drawing, pen position, pen down (constant image)."""
        obs = np.zeros((4, SIZE, SIZE), dtype=np.float32)
        obs[0] = self.target
        obs[1] = self.ink
        obs[2][self.pen] = 1.0
        obs[3] = float(self.pen_down)
        return obs

    def step(self, a):
        R = self.rewards
        self.steps += 1
        info = {"action": a}
        if a == STOP:
            missing = self.n_target - int(self.covered.sum())
            reward = R.stop_missing * missing
            if self.coverage() >= R.complete:
                reward += R.stop_bonus
            info.update(event="stop", missing=missing)
            self.history.append(info)
            return self.observation(), reward, True, False, info

        dr, dc = MOVES[a % N_MOVES]
        draw = a >= N_MOVES
        r, c = self.pen
        r2, c2 = r + dr, c + dc
        if not (0 <= r2 < SIZE and 0 <= c2 < SIZE):
            info.update(event="off_canvas")
            self.history.append(info)
            return self.observation(), R.off_canvas, False, self.steps >= self.max_steps, info

        reward = R.move
        new_cover = off = 0
        if draw:
            if not self.pen_down:
                reward += R.new_stroke
            for p in line_pixels((r, c), (r2, c2)):
                if self.ink[p]:
                    continue
                self.ink[p] = True
                if not self.zone[p]:
                    off += 1
                # skeleton pixels within 1 pixel of p become covered
                r0, r1 = max(p[0] - 1, 0), min(p[0] + 2, SIZE)
                c0, c1 = max(p[1] - 1, 0), min(p[1] + 2, SIZE)
                window = self.target[r0:r1, c0:c1] & ~self.covered[r0:r1, c0:c1]
                new_cover += int(window.sum())
                self.covered[r0:r1, c0:c1] |= self.target[r0:r1, c0:c1]
            reward += R.cover * new_cover + R.off_zone * off
        self.pen = (r2, c2)
        self.pen_down = draw
        info.update(event="move", start=(r, c), end=(r2, c2), pen_down=draw, new_cover=new_cover, off=off)
        self.history.append(info)
        return self.observation(), reward, False, self.steps >= self.max_steps, info

    # ------------------------------------------------------------------ measures
    def coverage(self):
        """Fraction of the skeleton covered (recall)."""
        return float(self.covered.sum()) / self.n_target

    def precision(self):
        """Fraction of the inked pixels that lie in the target zone."""
        n = int(self.ink.sum())
        return 1.0 if n == 0 else float((self.ink & self.zone).sum()) / n

    def perfect(self, terminated):
        return bool(terminated and self.coverage() >= self.rewards.complete and self.precision() >= 0.95)


def scripted_actions(letter: Letter):
    """A hand-written policy that follows the strokes of the letter with 5-pixel moves
    (pen up between strokes), then STOP. Used to check that every letter can be drawn perfectly."""
    from .letters import ALL_LETTER_STROKES, polyline_to_segments  # noqa: F401
    strokes = ALL_LETTER_STROKES[letter.name]
    actions, pen = [], to_pixel(letter.start_points[0])
    for stroke in strokes:
        points = [to_pixel(cell) for cell in stroke]
        # pen-up travel to the start of the stroke, with 5- then 1-pixel moves
        actions += _travel(pen, points[0])
        for p0, p1 in zip(points[:-1], points[1:]):
            n = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) // SPACING
            step = ((p1[0] - p0[0]) // n, (p1[1] - p0[1]) // n)
            actions += [N_MOVES + MOVES.index(step)] * n
        pen = points[-1]
    return actions + [STOP]


def _travel(p0, p1):
    actions, (r, c) = [], p0
    while (r, c) != tuple(p1):
        dr, dc = p1[0] - r, p1[1] - c
        best = min(range(N_MOVES), key=lambda m: (abs(dr - MOVES[m][0]) + abs(dc - MOVES[m][1]), -abs(MOVES[m][0]) - abs(MOVES[m][1])))
        actions.append(best)
        r, c = r + MOVES[best][0], c + MOVES[best][1]
    return actions
