"""
Random stroke figures, used as training targets so that the DQN cannot memorise each target (step 7c).

A figure is built like a letter: 1 to 3 strokes, each stroke made of 1 to 3 straight runs (horizontal,
vertical or 45-degree) of 1 to 4 moves. The second and third strokes start, most of the time, on a cell
of the figure already drawn (like the bars of E or the crossbar of H). The natural starting points are the
ends of the strokes. Figures identical to one of the 26 letters are rejected, so that the letters stay
truly unseen during training.
"""
import numpy as np

from .letters import GRID_SIZE, Letter, make_letter, polyline_to_segments, TRAINING_LETTERS, TEST_LETTERS

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
LETTER_SHAPES = {frozenset(make_letter(name).segments) for name in TRAINING_LETTERS + TEST_LETTERS}


def _random_stroke(rng, start):
    """A polyline of 1 to 3 straight runs starting at `start`, staying inside the grid."""
    points = [start]
    for _ in range(rng.integers(1, 4)):
        r, c = points[-1]
        options = []
        for dr, dc in DIRECTIONS:
            max_len = 0
            while 0 <= r + (max_len + 1) * dr < GRID_SIZE and 0 <= c + (max_len + 1) * dc < GRID_SIZE and max_len < 4:
                max_len += 1
            if max_len > 0:
                options.append((dr, dc, max_len))
        dr, dc, max_len = options[rng.integers(len(options))]
        length = int(rng.integers(1, max_len + 1))
        points.append((r + length * dr, c + length * dc))
    return points


def random_figure(rng, min_segments=4, max_segments=14, max_strokes=3, connect_prob=0.7):
    """Draw a random figure. Returns a Letter named 'random'."""
    while True:
        segments, starts = [], []
        for i in range(rng.integers(1, max_strokes + 1)):
            cells = sorted({cell for seg in segments for cell in seg})
            if i > 0 and cells and rng.random() < connect_prob:
                start = cells[rng.integers(len(cells))]
            else:
                start = (int(rng.integers(GRID_SIZE)), int(rng.integers(GRID_SIZE)))
            points = _random_stroke(rng, start)
            for seg in polyline_to_segments(points):
                if seg not in segments:
                    segments.append(seg)
            starts += [points[0], points[-1]]
        if not (min_segments <= len(segments) <= max_segments):
            continue
        if frozenset(segments) in LETTER_SHAPES:
            continue
        unique_starts = tuple(dict.fromkeys(starts))
        return Letter("random", tuple(segments), unique_starts)
