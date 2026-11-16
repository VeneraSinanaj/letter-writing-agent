"""
Letter definitions.

A letter is described by its strokes. Each stroke is a polyline: a list of cells
(row, col) on the 5x5 grid, with (0, 0) at the top-left corner. Two consecutive
points of a polyline must be joined by a horizontal, vertical or 45-degree line,
because these are the only directions the pen can move in.

The strokes are then cut into unit segments (one segment = one move between two
neighbouring cells). The environment rewards the agent segment by segment.
"""
from dataclasses import dataclass

GRID_SIZE = 5

Cell = tuple[int, int]
Segment = tuple[Cell, Cell]

# Strokes of each letter (polylines).
LETTER_STROKES: dict[str, list[list[Cell]]] = {
    "I": [[(0, 1), (0, 3)], [(0, 2), (4, 2)], [(4, 1), (4, 3)]],
    "L": [[(0, 0), (4, 0), (4, 3)]],
    "T": [[(0, 0), (0, 4)], [(0, 2), (4, 2)]],
    "E": [[(0, 3), (0, 0), (4, 0), (4, 3)], [(2, 0), (2, 2)]],
    "F": [[(0, 3), (0, 0), (4, 0)], [(2, 0), (2, 2)]],
    "H": [[(0, 0), (4, 0)], [(0, 4), (4, 4)], [(2, 0), (2, 4)]],
    "N": [[(4, 0), (0, 0), (4, 4), (0, 4)]],
    "Z": [[(0, 0), (0, 4), (4, 0), (4, 4)]],
    # On a grid, the arms of the V cannot be perfect diagonals: they alternate
    # straight and diagonal moves.
    "V": [[(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (3, 3), (2, 3), (1, 4), (0, 4)]],
    "X": [[(0, 0), (4, 4)], [(0, 4), (4, 0)]],
}

# Natural starting points: the ends of the strokes, where a person would start writing.
START_POINTS: dict[str, list[Cell]] = {
    "I": [(0, 1), (0, 3), (4, 1), (4, 3)],
    "L": [(0, 0), (4, 3)],
    "T": [(0, 0), (0, 4), (4, 2)],
    "E": [(0, 3), (2, 2), (4, 3)],
    "F": [(0, 3), (2, 2), (4, 0)],
    "H": [(0, 0), (4, 0), (0, 4), (4, 4)],
    "N": [(4, 0), (0, 4)],
    "Z": [(0, 0), (4, 4)],
    "V": [(0, 0), (0, 4)],
    "X": [(0, 0), (0, 4), (4, 0), (4, 4)],
}

TRAINING_LETTERS = list(LETTER_STROKES)  # the 10 letters of the tabular part


def normalize(a: Cell, b: Cell) -> Segment:
    """A segment has no direction: (a, b) and (b, a) are the same segment."""
    return (a, b) if a <= b else (b, a)


def polyline_to_segments(points: list[Cell]) -> list[Segment]:
    """Cut a polyline into unit segments between neighbouring cells."""
    segments = []
    for (r1, c1), (r2, c2) in zip(points[:-1], points[1:]):
        dr, dc = r2 - r1, c2 - c1
        if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
            raise ValueError(f"({r1},{c1})->({r2},{c2}) is not horizontal, vertical or diagonal")
        n = max(abs(dr), abs(dc))
        step_r, step_c = (dr > 0) - (dr < 0), (dc > 0) - (dc < 0)
        for i in range(n):
            a = (r1 + i * step_r, c1 + i * step_c)
            b = (r1 + (i + 1) * step_r, c1 + (i + 1) * step_c)
            segments.append(normalize(a, b))
    return segments


@dataclass(frozen=True)
class Letter:
    name: str
    segments: tuple[Segment, ...]  # unit segments, each one listed once
    start_points: tuple[Cell, ...]

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def make_letter(name: str) -> Letter:
    segments = []
    for stroke in LETTER_STROKES[name]:
        for seg in polyline_to_segments(stroke):
            if seg not in segments:
                segments.append(seg)
    for r, c in START_POINTS[name]:
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
            raise ValueError(f"start point {(r, c)} of {name} is outside the grid")
    return Letter(name, tuple(segments), tuple(START_POINTS[name]))
