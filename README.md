# Letter-Writing Agent

Reinforcement Learning course project (N. Mayer, University of Strasbourg, 2026-27), group of 3.

An agent learns, by trial and error, to draw letters on a 5×5 grid.
We compare several methods from the course: dynamic programming (value iteration),
Q-learning, SARSA, Monte Carlo, and finally a DQN able to draw letters it has never seen.

## Repository structure

| Folder | Content |
|---|---|
| `src/` | Reusable code (`.py`): the environment, the algorithms, the visualisation functions |
| `notebooks/` | The experiments (`.ipynb`): each notebook uses the code in `src/` and discusses the results |
| `results/` | Figures, animations and tables produced by the notebooks |

## Installation

```bash
python -m venv ../.venv
../.venv/Scripts/python -m pip install -r requirements.txt
```

The virtual environment is created **next to** the repository (`../.venv`), not inside it, so that it is not pushed to GitHub.

---

## Project choices (MDP definition)

### Environment
- 5×5 grid. A letter = a **set of segments** between neighbouring cells (diagonals included).
- 10 letters, each with at most 12 segments:
  - straight strokes: I, L, T, E, F, H;
  - diagonals: N, Z, V, X.

### State
The state contains three pieces of information:
- the pen position (25 positions);
- is the pen down? (yes / no);
- which segments of the letter have already been drawn (at most 2¹² combinations).

In total: up to 25 × 2 × 4,096 ≈ **205,000 states**, which fits in a table.

### Actions
**17 actions**:
- 8 directions × (pen up or pen down) = 16 moves;
- the STOP action ("I'm done"), which ends the episode.

### Transitions
Deterministic: the same action in the same state always gives the same result.

### Start
The agent starts on one of the letter's 2 to 4 **natural starting points** (the end of a stroke), drawn at random, with the pen up.

### Rewards

| Event | Reward | Purpose |
|---|---|---|
| Drawing a new segment of the letter | +1 | Main objective |
| Drawing a segment outside the letter | −1 | Cancels the gain of a correct segment |
| Each move | −0.05 | Encourages efficient drawing |
| Starting a new stroke (putting the pen down) | −0.3 | Lifting the pen costs as much as 6 moves |
| Moving off the grid | −0.2 | Useless move |
| STOP with the letter complete | +2 | Rewards finishing |
| STOP with missing segments | −1 per segment | Prevents stopping too early |

### Parameters
- Discount factor **γ = 0.99**.
- **50 steps maximum** per episode.

### Rationale
- **Segments rather than cells:** if we reward coloured cells, the agent can cut corners diagonally (the cells are the same, but the movement is unnatural). With segments, a diagonal that cuts a corner is not part of the letter, so it is penalised.
- **The −0.3 / −0.05 ratio:** it sets the agent's writing "style". Up to 6 moves, the agent prefers going back over a stroke rather than lifting the pen. We will vary this ratio in the experiments.
- **5×5 grid:** the number of states doubles with each added segment. On a 7×7 grid, a letter like E would have about 18 segments, i.e. about 26 million states: the table no longer fits in memory (about 1.7 GB). This is what motivates moving to a DQN.

### Letters used
- **Tabular part** (DP, Q-learning, SARSA, Monte Carlo): the 10 letters above. Each letter has its own Q-table, so adding more letters would show nothing new.
- **DQN part**: training on the 10 letters, then testing on the 16 others, never seen during training. This checks whether the agent has learned to write in general.
- **Limitation:** curved letters (C, O, S…) are very rough on a 5×5 grid; they will be used to discuss the limits of the model.

### Do we give data to the agent?
No. The letter is part of the environment: the environment knows it and uses it to compute the reward. It is not an example to imitate.

Nobody shows the agent how to draw. It knows neither the stroke order, nor the ideal starting point, nor the right move at each step. It only receives two things: **the state it is in and a reward** after each action. It discovers everything else by trial and error. This is why it scribbles at first.

Nuance for the discussion: the reward (+1 per correct segment) is built from the letter. This is *reward shaping*, human knowledge slipped into the reward. "Purer" versions exist (a reward only at the end, or a classifier that judges the drawing, as in SPIRAL by Ganin et al., 2018), but they are much harder to learn.

---

## Progress

| Step | Status |
|---|---|
| 1. MDP definition | Done |
| 2. Repository structure | Done |
| 3. Environment (`src/environment.py`) | Done |
| 4. Value iteration (reference optimum) | To do |
| 5. Q-learning, SARSA, Monte Carlo + comparison | To do |
| 6. Reward experiments | To do |
| 7. DQN + test on unseen letters | To do |
| 8. Figures, report, slides | To do |
