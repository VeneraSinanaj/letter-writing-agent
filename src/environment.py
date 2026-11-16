"""
The letter-writing environment (a finite, deterministic MDP).

State  = (pen row, pen col, pen down at the previous move?, segments of the letter already drawn)
Action = 8 directions x {pen up, pen down}  +  STOP                      -> 17 actions
Reward = see the `Rewards` class below (same values as in the README).

The interface follows Gymnasium: reset() starts an episode, step(action) plays one move.
Model-free methods (Q-learning, SARSA, Monte Carlo) only use reset() and step().
Dynamic programming needs the full model: build_model() gives it, by calling the
same transition() function as step(), so that both play exactly the same game.
"""
from dataclasses import dataclass

import numpy as np

from .letters import GRID_SIZE, Cell, Letter, make_letter, normalize

# Directions: (row change, col change). Row 0 is the top of the grid.
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRECTION_NAMES = ["up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right"]
N_MOVES = 2 * len(DIRECTIONS)  # actions 0-7: pen up, actions 8-15: pen down
STOP = N_MOVES                 # action 16
N_ACTIONS = N_MOVES + 1


def make_action(direction: str, pen_down: bool) -> int:
    """Action number from a readable description, e.g. make_action("down", pen_down=True) -> 9."""
    return DIRECTION_NAMES.index(direction) + (8 if pen_down else 0)


def action_name(a: int) -> str:
    if a == STOP:
        return "STOP"
    return f"{DIRECTION_NAMES[a % 8]} ({'pen down' if a >= 8 else 'pen up'})"


@dataclass(frozen=True)
class Rewards:
    new_segment: float = 1.0    # drawing a segment of the letter for the first time
    off_letter: float = -1.0    # drawing a segment that is not part of the letter
    move: float = -0.05         # every move
    new_stroke: float = -0.3    # putting the pen down to start a new stroke
    off_grid: float = -0.2      # trying to move outside the grid (the pen does not move)
    stop_complete: float = 2.0  # STOP when every segment has been drawn
    stop_missing: float = -1.0  # STOP too early: per missing segment


class LetterWritingEnv:
    def __init__(self, letter: Letter | str, rewards: Rewards = Rewards(), max_steps: int = 50, seed=None):
        self.letter = make_letter(letter) if isinstance(letter, str) else letter
        self.rewards = rewards
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        # each segment of the letter gets a bit number in the "drawn" mask
        self.segment_bit = {seg: i for i, seg in enumerate(self.letter.segments)}
        self.k = self.letter.n_segments
        self.full_mask = (1 << self.k) - 1
        self.n_cells = GRID_SIZE * GRID_SIZE
        self.n_states = self.n_cells * 2 * (1 << self.k)
        self.n_actions = N_ACTIONS
        self.state = None

    # ------------------------------------------------------------------ state <-> integer
    # Tabular methods need each state to be a single integer (a row of the Q-table).
    def encode(self, row: int, col: int, pen_down: int, mask: int) -> int:
        return (mask * self.n_cells + row * GRID_SIZE + col) * 2 + pen_down

    def decode(self, s: int) -> tuple[int, int, int, int]:
        s = int(s)  # works for numpy integers too
        pen_down = s % 2
        cell = (s // 2) % self.n_cells
        mask = s // (2 * self.n_cells)
        return cell // GRID_SIZE, cell % GRID_SIZE, pen_down, mask

    def start_states(self) -> list[int]:
        return [self.encode(r, c, 0, 0) for r, c in self.letter.start_points]

    # ------------------------------------------------------------------ the rules
    def transition(self, s: int, a: int) -> tuple[int, float, bool, dict]:
        """Result of action a in state s: (next state, reward, terminated, info).
        Pure function: it does not change the environment. All the rules are here."""
        R = self.rewards
        row, col, pen_down, mask = self.decode(s)

        if a == STOP:
            missing = self.k - bin(mask).count("1")
            reward = R.stop_complete if missing == 0 else R.stop_missing * missing
            return s, reward, True, {"event": "stop", "missing": missing}

        dr, dc = DIRECTIONS[a % 8]
        draw = a >= 8
        new_row, new_col = row + dr, col + dc
        if not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE):
            return s, R.off_grid, False, {"event": "off_grid"}

        reward = R.move
        info = {"event": "move", "from": (row, col), "to": (new_row, new_col), "pen_down": draw}
        if draw:
            if not pen_down:
                reward += R.new_stroke
            seg = normalize((row, col), (new_row, new_col))
            bit = self.segment_bit.get(seg)
            if bit is None:                   # not part of the letter
                reward += R.off_letter
                info["segment"] = "off_letter"
            elif not mask & (1 << bit):       # part of the letter, drawn for the first time
                reward += R.new_segment
                mask |= 1 << bit
                info["segment"] = "new"
            else:                             # already drawn: going back over it
                info["segment"] = "redrawn"
        return self.encode(new_row, new_col, int(draw), mask), reward, False, info

    # ------------------------------------------------------------------ Gymnasium-style interface
    def reset(self, start: Cell | None = None) -> int:
        """Start a new episode on a natural starting point (random if not given), pen up."""
        if start is None:
            start = self.letter.start_points[self.rng.integers(len(self.letter.start_points))]
        self.state = self.encode(start[0], start[1], 0, 0)
        self.steps = 0
        self.history = []  # list of info dicts, used to draw the episode
        return self.state

    def step(self, a: int) -> tuple[int, float, bool, bool, dict]:
        """Play action a. Returns (next state, reward, terminated, truncated, info).
        terminated = the agent pressed STOP (a real end of the MDP).
        truncated  = the step limit was reached (the episode is cut, but the state is not terminal)."""
        next_s, reward, terminated, info = self.transition(self.state, a)
        self.state = next_s
        self.steps += 1
        truncated = not terminated and self.steps >= self.max_steps
        self.history.append(info)
        return next_s, reward, terminated, truncated, info

    def drawn_segments(self) -> int:
        return bin(self.decode(self.state)[3]).count("1")

    # ------------------------------------------------------------------ full model (for dynamic programming)
    def build_model(self):
        """For every state s and action a: next_state[s, a], reward[s, a], terminated[s, a]."""
        next_state = np.empty((self.n_states, self.n_actions), dtype=np.int64)
        reward = np.empty((self.n_states, self.n_actions), dtype=np.float64)
        terminated = np.zeros((self.n_states, self.n_actions), dtype=bool)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_state[s, a], reward[s, a], terminated[s, a], _ = self.transition(s, a)
        return next_state, reward, terminated
