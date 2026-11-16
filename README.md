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

## Key results so far

### Step 3 — Environment (`notebooks/01_environment.ipynb`)
- A random agent scribbles: on E it draws 3 of the 12 segments, with several segments outside the letter, and gets a total reward of about −10.6.
- 7 hand-computed checks confirm every reward rule (e.g. drawing the whole L then STOP gives −0.3 + 7 × (−0.05) + 7 + 2 = +8.35).

### Step 4 — Value iteration (`notebooks/02_value_iteration.ipynb`, `src/dynamic_programming.py`)
- Value iteration uses the Bellman optimality equation; since the environment is deterministic, it reduces to v(s) = max_a [r(s, a) + γ v(s')]. All states are updated at once with numpy arrays (two-array version, Sutton & Barto §4.1), whereas the course notebook updates them one by one (in-place version); both converge to the same v*.
- **Every letter is drawn completely, with no segment outside the letter, from every starting point.** The optimal returns per starting point are saved in `results/02_optimal_returns.csv` (reference for the next steps).
- **The optimal agent never lifts the pen: every letter is drawn in a single stroke**, going back over segments already drawn (−0.05 per move) rather than starting a new stroke (−0.3). The reward defines the writing style; this will be studied in step 6.
- **The starting point matters**: for E, starting from the middle bar costs 18 steps instead of 15.
- **Convergence is exact and fast**: 13 to 23 iterations, more for letters with more segments. Δ stays close to 2 and then drops to exactly 0 once the number of iterations exceeds the longest optimal path (no gradual decrease, unlike Frozen Lake, because there is no randomness).
- Building the model takes up to ~6 s for the 12-segment letters, value iteration ~1–2 s; this grows with 2^(segments) (curse of dimensionality).

### Step 5a — Q-learning (`notebooks/03_q_learning.ipynb`, `src/tabular_agents.py`)
- Update rule (Sutton & Barto eq. 6.8): Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) − Q(S, A)], target = R after STOP. ε-greedy exploration, ε from 1 to 0.05 over 80 % of the episodes. α = 0.5, γ = 0.99, 30,000 episodes per letter, 5 runs (seeds) per letter.
- **Every letter is learned**: in all 50 runs, the final greedy policy draws every letter perfectly from every starting point. First perfect drawing after ~5,000 (L) to ~12,500 (H) episodes (median).
- **Fast**: 2–4 s per run; 50 runs in about 2.5 minutes.
- **Good but not optimal**: 93 % (H) to 100 % (L) of the optimum v*. Only L reaches the optimum in all runs. Two imperfections: *detours* (from some starts, the agent travels pen-up to the starting point it knows best: I, T, N, X) and *lifting the pen* (2 strokes for E, F, H, where the optimum uses 1 stroke and goes back over segments).
- **Cause: exploration.** Random ε-greedy exploration rarely tries the long sequences of unrewarded "going back" moves of the optimal drawings. With 100,000 episodes on H: 97 % of v* (vs 93 %), and only ~12 % of the states visited.
- Runs agree at the end (std ≤ 1.4 % of v*) but differ a lot during learning.

### Step 5b — SARSA and Monte Carlo (`notebooks/04_sarsa_monte_carlo.ipynb`)
- Same conditions as Q-learning (letters, 30,000 episodes, ε schedule, 5 seeds, evaluation); only the update rule and the step size change. SARSA: Q(S, A) ← Q(S, A) + α [R + γ Q(S', A') − Q(S, A)]. Monte Carlo: on-policy, first-visit, Q(s, a) ← Q(s, a) + step × [G − Q(s, a)], with step = 1/N (sample average, course formula) or a constant α.
- **Step sizes must be chosen per algorithm**: SARSA with α = 0.5 learns and then degrades (its target depends on a sometimes random next action, so it is noisy); α = 0.1 works. Monte Carlo fails whatever α (0.02–0.1); we keep α = 0.05 for the constant-step version.
- **SARSA ≈ Q-learning in final quality** (84–100 % of v*, perfect drawings in 90–100 % of the cases) but **≈ 1.5× slower** (first perfect drawing after 6,500–19,500 episodes vs 5,000–12,500) and **less stable** (sudden drops of the greedy return, even at the end).
- **"SARSA is more cautious" is not confirmed**: at the end the training returns are almost equal (L: 8.00 vs 7.95). The cliff-walking effect needs very costly exploratory mistakes; in our environment a random move costs at most −1.35 and can be repaired, so the safest drawing is also the optimal one.
- **Monte Carlo fails on every letter (both versions): the "giving up" trap.** The agent learns to draw one to five segments and press STOP (V, X: STOP immediately). From the start, playing at random is worth about −11 on average, stopping immediately only −7 (L, −1 per missing segment). Judged by the returns of its own mostly random policy, "continue" looks worse than "stop", and the agent never gets out. Q-learning escapes thanks to the max, SARSA thanks to bootstrapping.
- **With a stronger early-stop penalty (−3 per missing segment)**, Monte Carlo learns L, Z and V perfectly and part of the others, but still fails on X; Q-learning is unaffected (93–100 %). The reward and the algorithm interact; Monte Carlo remains much less sample-efficient (one return per episode, high variance, updates only at the end).

### Step 6a — Reward and writing style (`notebooks/05_reward_style.ipynb`)
- New reward parameter `redrawn` (penalty for going back over a segment already drawn); default 0, so steps 3–5 are unchanged.
- **Trade-off**: at the end of a stroke, the agent lifts the pen if and only if m·|r_move| + |r_stroke| < k·(|r_move| + |r_redrawn|), where k = segments to go back over and m = pen-up moves. On a grid m = k very often, so with r_redrawn = 0 lifting the pen is never strictly better.
- **The stroke penalty alone cannot produce a human style**: with −0.1, −0.3 or −1, every letter is drawn in one stroke, going back over 2–4 segments. With 0, lifting and going back cost the same: the strokes are decided by a tie.
- **The going-back penalty changes the style letter by letter, as the inequality predicts**: −0.1 → only X lifts the pen; −0.2 → T, E, F, H, X lift it; −0.5 → no segment is ever gone back over (I: 3 strokes; T, E, F, X: 2; H: 3; L, N, Z, V: 1) — the strokes a person would use.
- **Q-learning adopts the same style**: with r_redrawn = −0.5 it uses the optimal number of strokes for 9 letters out of 10, never goes back, draws every letter perfectly, 97–100 % of the new optimum (X: 4 strokes instead of 2, 90 %).
- Discussion: the reward defines not only *what* to do but *how*; a natural behaviour appears only if the reward makes it strictly better (link with reward shaping theory, Ng, Harada & Russell, 1999).

---

## Algorithm comparison (for the discussion)

Filled in step by step with the results of our own experiments.

| Algorithm | Advantages | Disadvantages | What our experiments show |
|---|---|---|---|
| **Value iteration** (DP, ch. 4) | Exact optimum; guaranteed convergence; very few iterations | Needs the full model p(s′, r \| s, a); must sweep **all** states, even useless ones; memory and time grow with 2^(segments) (curse of dimensionality) | Optimum for every letter in 13–23 iterations (1–2 s), but building the model takes up to 6 s and would be impossible on a 7×7 grid (~26 million states) |
| **Q-learning** (TD, off-policy, ch. 6) | Model-free (only plays); visits only the useful states; off-policy: learns the greedy policy while exploring; the max values the best continuation, which protects it from the "giving up" trap; fast | No guarantee of reaching the optimum in limited time; depends on exploration and parameters (α, ε, episodes); still a table: no generalisation to unseen letters | All letters drawn perfectly, 93–100 % of v*; fastest learner (first perfect drawing after 5,000–12,500 episodes); ~12 % of the states of H visited; 2–4 s per run |
| **SARSA** (TD, on-policy, ch. 6) | Learns the value of the policy it actually follows (exploration included); bootstrapping lets it escape the "giving up" trap | Target depends on the (sometimes random) next action: noisier, needs a smaller α; optimal only if ε → 0; slower | Same final quality as Q-learning (84–100 % of v*) but ≈1.5× more episodes and unstable curves; with α = 0.5 it degrades. Expected "caution" not observed: our environment has no costly "cliff" |
| **Monte Carlo** (on-policy, ch. 5) | No bootstrapping, so unbiased estimates of the returns of the current policy; simplest idea (average of returns) | Waits for the end of the episode; high variance; needs terminating episodes (our 50-step cut is an approximation); judges actions by the returns of its current, mostly random, policy | Fails on every letter (both versions): learns to draw a few segments and STOP ("giving up" trap). With a stronger early-stop penalty it learns 3 letters perfectly, still far behind Q-learning |
| **DQN** (deep RL) | *Expected:* generalises across states and letters (unseen letters); scales to large grids | *Expected:* unstable, no convergence guarantee, many hyperparameters, slower, less interpretable | To do (step 7) |

---

## Progress

| Step | Status |
|---|---|
| 1. MDP definition | Done |
| 2. Repository structure | Done |
| 3. Environment (`src/environment.py`) | Done |
| 4. Value iteration (reference optimum) | Done |
| 5a. Q-learning | Done |
| 5b. SARSA, Monte Carlo + comparison | Done |
| 6a. Reward and writing style | Done |
| 6b. Reward and learnability (early-stop penalty, sparse reward) | To do |
| 7. DQN + test on unseen letters | To do |
| 8. Figures, report, slides | To do |
