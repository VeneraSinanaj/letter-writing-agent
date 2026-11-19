# Letter-Writing Agent

Reinforcement Learning course project (N. Mayer, University of Strasbourg, 2026-27), group of 3.

An agent learns, by trial and error, to draw letters on a 5×5 grid.
We compare several methods from the course: dynamic programming (value iteration),
Q-learning, SARSA, Monte Carlo, and finally a DQN able to draw letters it has never seen.
As an extension, the agent learns to draw freely on a blank 28×28 pixel sheet, where a convolutional DQN trained
only on random figures draws 96 % of the 26 letters perfectly.

**Contents:** [Repository structure](#repository-structure) · [Installation](#installation) · [How to reproduce](#how-to-reproduce) · [Project choices (MDP)](#project-choices-mdp-definition) · [Key results](#key-results-so-far) · [Algorithm comparison](#algorithm-comparison-for-the-discussion) · [Progress](#progress) · [Report outline](#report-outline) · [References](#references)

## Repository structure

| Folder / file | Content |
|---|---|
| `src/letters.py` | The 26 letters: strokes on the grid and natural starting points (10 training letters + 16 test letters) |
| `src/environment.py` | The MDP: `LetterWritingEnv` (`reset`, `step`, `transition`, `build_model`) and the `Rewards` |
| `src/dynamic_programming.py` | Value iteration |
| `src/tabular_agents.py` | Q-learning, SARSA, Monte Carlo (sample average or constant step), ε-greedy exploration |
| `src/dqn.py` | Deep Q-Network: observation (basic 170 / rich 266), network, replay memory, training, evaluation |
| `src/figures.py` | Random stroke figures used to train the generalising DQN |
| `src/free_draw.py` | Step 9, the "blank sheet": free drawing on a 28×28 pixel canvas (`FreeDrawEnv`) |
| `src/free_dqn.py` | Steps 9b–9c: DQN with a convolutional network for the blank sheet (Double DQN, compact replay memory, optional egocentric view) |
| `src/evaluation.py` | Playing a policy and measuring it (return, steps, strokes, errors, completion) |
| `src/visualization.py` | Drawing letters and episodes, animations |
| `notebooks/` | The experiments, numbered in the order of the project (see [How to reproduce](#how-to-reproduce)) |
| `results/` | Figures, animations, tables and trained networks produced by the notebooks |
| `figures/` | The final figures of the report, in the report style: `fig01` to `fig14` (grid, notebook 10), `fig15` to `fig17` (blank sheet, notebooks 12 and 13) |
| `src/report_style.py` | The style of the report figures: dark muted colours, Times font, no titles inside the figures |

## Installation

```bash
python -m venv ../.venv
../.venv/Scripts/python -m pip install -r requirements.txt
```

The virtual environment is created **next to** the repository (`../.venv`), not inside it, so that it is not pushed to GitHub.
In VS Code or Jupyter, select this environment as the kernel (`.venv`): PyTorch is installed there, not in a global Python.

## How to reproduce

Run the notebooks in this order (Run All). Times measured on a laptop CPU (no GPU); they may vary.

| Notebook | Step | What it does | Needs | Time |
|---|---|---|---|---|
| `01_environment` | 3 | Letters, random agent, 7 checks of the reward rules, full model | — | < 1 min |
| `02_value_iteration` | 4 | Optimal drawings (value iteration), convergence; saves `02_optimal_returns.csv` | — | ~1 min |
| `03_q_learning` | 5a | Q-learning, 10 letters × 5 seeds; saves `03_q_learning_summary.csv` and `03_q_learning_curves.npz` | 02 | ~3 min |
| `04_sarsa_monte_carlo` | 5b | SARSA, Monte Carlo (2 versions), comparison with Q-learning, "giving up" trap | 02, 03 | ~11 min |
| `05_reward_style` | 6a | Stroke and going-back penalties: writing style | — | ~2.5 min |
| `06_reward_learnability` | 6b | Early-stop penalty, delayed reward | 03, 04 | ~6 min |
| `07_dqn` | 7a | DQN on the 10 letters, 3 seeds; saves the networks `07_dqn_seed*.pt` | 02, 03 | 15–25 min (seconds if the networks are already in `results/`) |
| `08_generalisation` | 7b | The 16 unseen letters: test of the networks of 07 | 07 | ~1 min |
| `09_better_generalisation` | 7c | Random figures + rich observation (2 × 2 design) | 07 | ~1.5 min with the saved networks (~1.5 h with `RETRAIN = True`) |
| `11_free_draw_environment` | 9a | The blank-sheet environment: target images, moves, rule checks, every letter drawable | — | < 1 min |
| `12_free_draw_dqn` | 9b | Convolutional DQN on the blank sheet: whole sheet vs sheet seen from the pen, the 10 letters, first look at the 16 unseen letters; report figure `fig15` | 11 | < 1 min with the saved network (~75 min with `RETRAIN = True`) |
| `13_free_draw_generalisation` | 9c | The same agent trained on random figures only, tested on the 26 letters; comparison with notebook 12; failures; report figures `fig16`, `fig17` | 12 | ~1.5 min with the saved network (~9 h with `RETRAIN = True`) |
| `10_report_figures` | 8b | The 14 final figures of the report in `figures/`, in the report style (recomputes a few curves, no network training) | 02–09 | ~7 min |

Random seeds are fixed: the tabular results are exactly reproducible; PyTorch on CPU is not always bit-for-bit
reproducible, so DQN numbers can differ slightly after retraining.

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
- **The comparison of the course (ch. 6) against our results**: "Q-learning learns faster" → confirmed (≈ 1.5×); "Q-learning can learn from experience replay (off-policy)" → this is what our DQN does (step 7a); "SARSA is safer" → not observed (no cliff, see above); "SARSA gets more reward while learning" → not observed either: the training return of Q-learning is equal (L) or higher (E) during learning (`results/04_on_vs_off_policy.png`).
- **Monte Carlo fails on every letter (both versions): the "giving up" trap.** The agent learns to draw one to five segments and press STOP (V, X: STOP immediately). From the start, playing at random is worth about −11 on average, stopping immediately only −7 (L, −1 per missing segment). Judged by the returns of its own mostly random policy, "continue" looks worse than "stop", and the agent never gets out. Q-learning escapes thanks to the max, SARSA thanks to bootstrapping.
- **With a stronger early-stop penalty (−3 per missing segment)**, Monte Carlo learns L, Z and V perfectly and part of the others, but still fails on X; Q-learning is unaffected (93–100 %). The reward and the algorithm interact; Monte Carlo remains much less sample-efficient (one return per episode, high variance, updates only at the end).

### Step 6a — Reward and writing style (`notebooks/05_reward_style.ipynb`)
- New reward parameter `redrawn` (penalty for going back over a segment already drawn); default 0, so steps 3–5 are unchanged.
- **Trade-off**: at the end of a stroke, the agent lifts the pen if and only if m·|r_move| + |r_stroke| < k·(|r_move| + |r_redrawn|), where k = segments to go back over and m = pen-up moves. On a grid m = k very often, so with r_redrawn = 0 lifting the pen is never strictly better.
- **The stroke penalty alone cannot produce a human style**: with −0.1, −0.3 or −1, every letter is drawn in one stroke, going back over 2–4 segments. With 0, lifting and going back cost the same: the strokes are decided by a tie.
- **The going-back penalty changes the style letter by letter, as the inequality predicts**: −0.1 → only X lifts the pen; −0.2 → T, E, F, H, X lift it; −0.5 → no segment is ever gone back over (I: 3 strokes; T, E, F, X: 2; H: 3; L, N, Z, V: 1) — the strokes a person would use.
- **Q-learning adopts the same style**: with r_redrawn = −0.5 it uses the optimal number of strokes for 9 letters out of 10, never goes back, draws every letter perfectly, 97–100 % of the new optimum (X: 4 strokes instead of 2, 90 %).
- Discussion: the reward defines not only *what* to do but *how*; a natural behaviour appears only if the reward makes it strictly better (link with reward shaping theory, Ng, Harada & Russell, 1999).

### Step 6b — Reward and learnability (`notebooks/06_reward_learnability.ipynb`)
- **Early-stop penalty** (−1, −2, −3, −5 per missing segment; L, T, E, X; Monte Carlo 3 runs, Q-learning 1 run): a stronger penalty lets Monte Carlo learn, letter by letter: L perfect from −2, T only at −5 (78 % perfect drawings), X only at −5 (50 %), E partial (22 %) whatever the penalty. Q-learning is unaffected (96–100 % of v*, 100 % perfect for every penalty).
- **Simple prediction** ("STOP immediately" vs "play at random", estimated by simulation): right trend, not the exact threshold. Important correction: random play also ends with a random STOP that pays the same penalty, so a stronger penalty makes both options worse. Where "ok" is predicted, Monte Carlo always starts to learn; where "trap" is predicted it mostly fails, but L is learned at −2 and −3 anyway (the prediction only looks at the very beginning, when the agent is fully random).
- **Delayed reward** (no +1 per segment, signal only at STOP): Q-learning still learns 8 letters out of 10 perfectly (3 runs each), but 2–3× more slowly (first perfect drawing after 13,000–27,000 episodes vs 5,000–12,500), and fails on E and H, the letters with the longest optimal drawings (15 and 17 steps). Perfect drawings appear only once ε is small: with strong exploration the agent almost never completes the letter, so it almost never sees the final reward. SARSA learns only L (V in some runs), Monte Carlo nothing.
- Discussion: reward shaping (+1 per segment) does not change the length of the optimal drawings, but makes them much easier to discover; a badly chosen penalty can create a trap that some algorithms (Monte Carlo) cannot escape while others (Q-learning) are robust.

### Step 7a — Deep Q-Network on the 10 letters (`notebooks/07_dqn.ipynb`, `src/dqn.py`)
- **One network for all letters.** Observation of 170 numbers: target letter (72 bits, one per possible segment of the 5×5 grid), segments already drawn (72), pen position (25, one-hot), pen down (1). MLP 170 → 256 → 256 → 17. Target y = R + γ max_a Q(o′, a; θ⁻) (y = R after STOP), Huber loss, Adam (lr 0.0005). Experience replay (100,000 transitions, mini-batches of 64, one gradient step every 4 steps), target network updated every 1,000 steps. 300,000 steps on the 10 letters together (random letter at each episode), ε from 1 to 0.05 over 60 % of the steps, 3 seeds. Trained networks saved in `results/07_dqn_seed*.pt`.
- **Link with the course (ch. 7)**: the course writes the loss L(θ) = E[(R + γ max_a Q(S′, a; θ) − Q(S, A; θ))²]. We add the two stabilisers of Mnih et al. (2015): a *target network* θ⁻ (a copy of the network, updated every 1,000 steps, so that the target does not move at every step) and the *Huber loss* (squared error for small errors, absolute error for large ones, which limits large corrections). DQN is built on Q-learning, not SARSA, because off-policy learning allows experience replay (old transitions were played by an older policy). Exploration (a question asked in the course): ε-greedy when choosing the action during training, greedy policy for evaluation.
- **It learns the 10 letters**: 87–100 % of v* per letter (mean over 3 runs), perfect drawings 100 % for 8 letters (F 89 %, V 67 %); 2 runs out of 3 draw every letter perfectly at the end.
- **Much more sample-efficient than tabular Q-learning**: ~1,150–1,200 episodes per letter until every letter is perfect, vs 5,000–12,500 per letter. Reasons: experience replay (each transition reused many times) and generalisation (shared sub-skills between letters, values inferred for similar states).
- **Closer to the optimum on several letters** (I, T, E, H: 99–100 % vs 93–97 % for Q-learning): states never visited get values from similar visited states.
- **Slower per step** (4–5.5 min per run vs ~35 s for the 10 Q-learning tables) and **unstable**: the fraction of perfect drawings oscillates strongly even at the end (seed 1: 90 → 71 → 51 → 90 → 43 → 87 % over the last 50,000 steps) — the "deadly triad" (function approximation + bootstrapping + off-policy, Sutton & Barto §11.3). Possible improvements: smaller learning rate at the end, Double DQN, keeping the best network, longer training.

### Step 7b — Generalisation to unseen letters (`notebooks/08_generalisation.ipynb`)
- The 16 other letters (A, B, C, D, G, J, K, M, O, P, Q, R, S, U, W, Y) are defined in `src/letters.py` (curves approximated on the grid; 6 to 14 segments). The 3 trained networks are tested **without any new training**.
- **No generalisation**: 0 % perfect drawings, 0–33 % of the segments drawn, and the agent **never presses STOP** (all episodes reach the 50-step limit) — vs 96 % perfect, 99 % coverage and 96 % STOP on the training letters. A random agent draws 5–8 % of the segments: the DQN does better on D, K, P, U (~30 %) but worse on G, Q, Y (0 %).
- **It replays memorised routines**: for D or K it draws the vertical stem from the top-left corner as for L, E, F, then gets lost or draws outside (A, O, P, U: 8–13 segments outside on average). Similarity to the closest training letter helps only weakly (correlation 0.30; W ≈ N with Jaccard 0.71, yet only 6 % drawn).
- **Why**: with only 10 different targets, the target bits act as a letter *identifier*; the network learns 10 separate behaviours rather than the general rule "draw the missing target segments, STOP when drawn = target". RL agents overfit to their training tasks and need many diverse tasks to generalise (Cobbe et al., 2019).
- Possible improvements: train on thousands of random stroke figures; give the *missing* segments as input or use a convolutional network (local rule shared between positions); train on more letters.

### Step 7c — Better generalisation (`notebooks/09_better_generalisation.ipynb`, `src/figures.py`)
- Two ideas, tested in a 2 × 2 design: **diverse targets** (random stroke figures, a new one at each episode, figures identical to a letter rejected, so that *no letter is ever seen during training*) and a **rich observation** (266 numbers: + the 72 *missing* segments + a 24-bit *local view* around the pen: for each direction, is the segment missing / drawn / outside the grid). A and B: 1,000,000 steps; C: 300,000 steps. Trained networks saved in `results/09_dqn_*.pt`.

| | 10 first letters | 16 other letters | STOP on the 16 others |
|---|---|---|---|
| Notebook 07: basic obs, trained on the 10 letters | 96 % | 0 % | 0 % |
| A: basic obs, random figures | 0 % | 0 % | 0 % |
| C: rich obs, trained on the 10 letters | 100 % | 0 % | 0 % |
| **B: rich obs, random figures (2 seeds)** | **94 %** | **92 %** | **92 %** |

- **Only B generalises**: it draws 92–94 % of the letters perfectly although it has never seen any of them (90–92 % of held-out random figures). Both seeds agree.
- **Diversity alone is not enough** (A learns nothing, even on random figures: the general rule is too hard to discover from the basic observation once memorising is impossible); **a good observation alone is not enough** (C memorises its 10 letters again). Generalisation needs both: diversity *forces* the general rule, the representation makes it *learnable* (Cobbe et al., 2019).
- **Remaining failures** (11 of 140 episodes): always at junctions in the middle of a stroke (J, K, T, H, I, W, Q): after one stroke, the next one starts far from the pen, the local view shows nothing missing, and the agent wanders until the 50-step limit.
- Cost: ~20–35 min per B run. Discussion: choosing the state representation is part of the agent's design, like the reward; it gives no information about the solution, yet it takes the same algorithm from 0 % to 92 % on unseen letters.

### Step 9a — The "blank sheet" environment (`notebooks/11_free_draw_environment.ipynb`, `src/free_draw.py`)
- Extension of the project: the agent draws **freely on a 28×28 pixel canvas** (no grid points, no ready-made segments). Target = an **image** of the letter (strokes drawn as 1-pixel lines, the "skeleton"). Moves: 16 directions × 3 lengths (1, 2, 5 px) → 40 distinct moves, pen up or down, + STOP = **81 actions**. Observation: 4 images of 28×28 (target, drawing, pen position, pen down).
- A drawing is judged with a **1-pixel tolerance**: a skeleton pixel is *covered* if a pixel at most 1 pixel away is inked; an inked pixel is *good* if it lies in the skeleton widened by 1 pixel. Rewards keep the spirit of the grid: +0.2 per newly covered skeleton pixel (≈ +1 per 5-pixel piece), −0.2 per inked pixel outside the zone, −0.05 per move, −0.3 per stroke, −0.2 off canvas; STOP: +2 if ≥ 95 % covered, −0.2 per uncovered pixel.
- Checks: 6 hand-computed rule checks pass (e.g. whole L then STOP = 36 × 0.2 − 0.3 − 7 × 0.05 + 2 = +8.55); a random agent scribbles (5 % coverage, −27.7); a hand-written policy draws **all 26 letters perfectly**, so the task is feasible with these moves.
- Why 28×28: already a real blank sheet (the pen goes anywhere), the standard size of handwritten characters (MNIST / EMNIST), and small enough for CPU training (256×256 would be ~80× more pixels).
- Why discrete moves: DQN cannot handle continuous actions (course ch. 7), so the free moves (any direction, any length) are discretised into 16 directions × 3 lengths.

### Step 9b — Learning to draw on the blank sheet (`notebooks/12_free_draw_dqn.ipynb`, `src/free_dqn.py`)
- Agent: **convolutional Double DQN** (van Hasselt et al., 2016): 3 convolutions (the first one at full resolution, the lines are 1 pixel wide) + 2 dense layers → 81 action values; replay memory stored as bits; **stabilised settings** (learning rate 0.0001, target network every 5,000 steps, gradient norm clipped at 10, memory of 200,000 transitions) because with the settings of step 7 learning on several letters rose and then collapsed (as the course warns for policy gradient, ch. 7, too large a learning rate makes learning diverge).
- **The way of seeing decides everything** (L and T, 100,000 steps): whole sheet (4 images of 28×28) → coverage up to 86 % but **0 % perfect drawings**; **egocentric view** = a 41×41 window centred on the pen (what is still missing, the drawing, the inside of the sheet, pen down) → **100 % perfect drawings**. Centred on the pen, "a piece is missing just to my right" always looks the same, wherever the pen is: the image version of the local view of step 7c. Numbers in `results/12_trials_two_letters.csv`.
- **On the 10 letters: 100 % perfect drawings** from every natural starting point (100 % covered, 100 % of the ink on the letter, ended by STOP), after 500,000 steps (74 min on a laptop CPU). Network and learning curves saved in `results/12_free_dqn_10letters.*`.
- Learning curves: coverage jumps to 81 % in 50,000 steps (the +0.2 per new pixel is immediate), dips to ~50 % while ε decreases, then rises; perfect drawings appear only after 150,000 steps and follow the STOP curve almost exactly: **learning when to stop is the hard part** (delayed reward, step 6b). The curve is still rising at the end (88 % at 475,000 steps, 100 % at 500,000): more steps would make the result more robust.
- Style: the strokes follow the letters exactly, but some pen-up moves wander (E, H): a pen-up move costs only −0.05, so nothing pushes towards the shortest path.
- **No generalisation yet**: on the 16 unseen letters, 0 % perfect, 21 % coverage on average. On most of them the greedy agent repeats a move that would leave the sheet (refused, so the observation does not change) until the 60-step limit; on M it covers 98 % but goes back and forth and never stops. Same lesson as step 7b: 10 training targets → 10 memorised behaviours. Step 9c applies the remedy of step 7c (random figures).

### Step 9c — Generalisation on the blank sheet (`notebooks/13_free_draw_generalisation.ipynb`)
- **Same agent and settings as step 9b, only the training targets change**: a new random figure at each episode (the generator of step 7c, `src/figures.py`, drawn on the canvas), 2,000,000 steps (8.9 h on a laptop CPU). Figures identical to a letter are rejected, so **all 26 letters are unseen**, including the 10 letters of the project.
- **Model selection on a validation set**: learning oscillates, so the network is saved at each evaluation (every 50,000 steps) and we keep the best one on 50 held-out random figures (98 %, step 1,550,000); the letters are never used to choose, they stay a true test set. Files: `results/13_free_dqn_figures_best.pt` (network kept), `_last.pt` (step 2,000,000), `_curves.npz`.
- **Result: 96 % of the 26 letters drawn perfectly** (63 episodes out of 66; 24 letters perfect from every starting point, including O, C, S, G, Q, B, K, R), coverage 99 %, precision 100 %.
- **Specialist vs generalist** (perfect drawings): trained on the 10 letters (9b) → 100 % on the 10 letters, **0 %** on the 16 others, 8 % on random figures; trained on random figures (9c) → 95 %, **97 %**, 98 %. Same recipe as step 7c (diverse targets + an observation that makes the local rule learnable), here on pixels with 81 actions instead of a 5×5 grid with 17 actions (92 %).
- Learning curves: no perfect drawing for 750,000 steps (coverage first collapses to 6 %, then rises as ε decreases), then a sudden take-off between 800,000 and 1,100,000 steps (0 → ~90 %), then oscillations between 80 and 98 % until the end. The network of step 1,400,000 would have scored only 80 % on the letters: choosing on held-out figures matters. Validation and letter curves move together, so random figures are a good proxy.
- **Failures** (3 episodes): X from 2 of its 4 starting points and A from 1: the agent ends up going back and forth on a piece already drawn until the 60-step limit (on X, it switches to the other diagonal at the crossing). A deterministic greedy policy that sends two observations to each other loops forever: the "deterministic policy" limit of DQN in the course (ch. 7), in a given state the agent always chooses the same action. As in step 7c, the hard places are where strokes meet or cross.

---

## Algorithm comparison (for the discussion)

Filled in step by step with the results of our own experiments.

Course vocabulary (ch. 6): **bootstrapping** = updating an estimate from other estimates; **sampling** = learning
from experience instead of a known model.

| | Bootstrapping | Sampling |
|---|---|---|
| DP (value iteration) | yes | no (needs the model) |
| Monte Carlo | no | yes |
| TD (Q-learning, SARSA), DQN | yes | yes |

Our results illustrate it: without bootstrapping, Monte Carlo falls into the "giving up" trap; Q-learning and SARSA
escape it.

| Algorithm | Advantages | Disadvantages | What our experiments show |
|---|---|---|---|
| **Value iteration** (DP, ch. 4) | Exact optimum; guaranteed convergence; very few iterations | Needs the full model p(s′, r \| s, a); must sweep **all** states, even useless ones; memory and time grow with 2^(segments) (curse of dimensionality) | Optimum for every letter in 13–23 iterations (1–2 s), but building the model takes up to 6 s and would be impossible on a 7×7 grid (~26 million states) |
| **Q-learning** (TD, off-policy, ch. 6) | Model-free (only plays); visits only the useful states; off-policy: learns the greedy policy while exploring; the max values the best continuation, which protects it from the "giving up" trap; fast; off-policy, so it can learn from old transitions (experience replay, course ch. 6): the basis of DQN | No guarantee of reaching the optimum in limited time; depends on exploration and parameters (α, ε, episodes); still a table: no generalisation to unseen letters | All letters drawn perfectly, 93–100 % of v*; fastest learner (first perfect drawing after 5,000–12,500 episodes); ~12 % of the states of H visited; 2–4 s per run |
| **SARSA** (TD, on-policy, ch. 6) | Learns the value of the policy it actually follows (exploration included); bootstrapping lets it escape the "giving up" trap | Target depends on the (sometimes random) next action: noisier, needs a smaller α; optimal only if ε → 0; slower | Same final quality as Q-learning (84–100 % of v*) but ≈1.5× more episodes and unstable curves; with α = 0.5 it degrades. Expected "caution" not observed: our environment has no costly "cliff". The other claim of the course, more reward while learning, is not observed either: the training return of Q-learning is equal (L) or higher (E) (notebook 04) |
| **Monte Carlo** (on-policy, ch. 5) | No bootstrapping, so unbiased estimates of the returns of the current policy; simplest idea (average of returns) | Waits for the end of the episode; high variance; needs terminating episodes (our 50-step cut is an approximation); judges actions by the returns of its current, mostly random, policy | Fails on every letter (both versions): learns to draw a few segments and STOP ("giving up" trap). With a stronger early-stop penalty it learns 3 letters perfectly, still far behind Q-learning |
| **DQN** (deep RL, ch. 7) | One network for all letters; generalises between similar states and letters (values for states never visited); experience replay reuses each transition; scales to large state spaces | Unstable (deadly triad), no convergence guarantee; many hyperparameters; more computation per step; harder to interpret; discrete actions only; deterministic greedy policy (course ch. 7) | 10 letters with one network, 87–100 % of v*; ~1,200 episodes per letter (vs 5,000–12,500 for Q-learning); 99–100 % on letters where Q-learning stays at 93–97 %; but strong oscillations even at the end (1 run out of 3 not perfect at the end); 4–5.5 min per run. **Unseen letters: no generalisation** (0 % perfect, ≤ 33 % of the segments, never STOP): with 10 training targets it memorises 10 behaviours instead of a general rule. **With random training figures + a rich observation (step 7c): 92 % of the unseen letters drawn perfectly**; either ingredient alone gives 0 %. **Blank sheet (step 9b, 81 actions, image observation): a convolutional Double DQN with a view centred on the pen draws the 10 letters perfectly** (500,000 steps, 74 min); the whole-sheet view gives 0 % perfect drawings. **Trained on random figures only (step 9c): 96 % of the 26 letters, all unseen, drawn perfectly** (2,000,000 steps, 8.9 h). The two limits of DQN given in the course appear: the free moves had to be discretised (16 directions × 3 lengths), and the deterministic greedy policy can loop (steps 9b, 9c) |

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
| 6b. Reward and learnability (early-stop penalty, delayed reward) | Done |
| 7a. DQN on the 10 letters | Done |
| 7b. DQN on unseen letters (generalisation) | Done |
| 7c. Better generalisation (random figures + rich observation) | Done |
| 8a. README as a report dossier (reproduction, outline, references) | Done |
| 8b. Final report figures (`notebooks/10_report_figures.ipynb`) | Done |
| 8c. Presentation (20 January 2027) | To do |
| 9a. Blank sheet: environment (`src/free_draw.py`, notebook 11) | Done |
| 9b. Blank sheet: DQN with a convolutional network (`src/free_dqn.py`, notebook 12) | Done — egocentric view: the 10 letters 100 % perfect; unseen letters 0 % |
| 9c. Blank sheet: training on random figures, test on the 26 letters (notebook 13) | Done — 96 % of the 26 unseen letters drawn perfectly |

---

## Report outline

Guideline: PDF of **12–15 pages maximum for a group of 3** (excluding references and appendix), 10–12 pt, structure
Introduction / Methodology / Results / Discussion / Conclusion, **at least 8 references, all cited in the text**
(APA or Chicago), sources of inspiration cited in the introduction. Deadline: **10 January 2027, 23:59**
(PDF + code to npromwang@unistra.fr); presentation on 20 January 2027.

**How to use this outline.** For each part: what it must show, the key numbers, the figures, and where the
details are in this README. It says *what* to write, not *how*: the writing is yours. Two parts: the **5×5 grid**
(all the algorithms of the course, steps 1–8), then the **blank sheet** (free drawing on pixels, DQN only, step 9).
All the report figures are in `figures/`: `fig01`–`fig14` for the grid (notebook 10), `fig15`–`fig17` for the
blank sheet (notebooks 12 and 13). Page budget ≈ 14.5 pages.

### 1. Introduction (≈ 1 p)
- **Show:** the problem, and why it is reinforcement learning, not supervised learning.
- **Key points:** an agent learns to draw letters stroke by stroke, by trial and error; nobody shows it how (no
  stroke order, no start, no correct move), it only gets states and rewards (see "Do we give data to the agent?").
- **Contributions** (one sentence each): an MDP for letter writing; DP, Q-learning, SARSA, Monte Carlo and DQN
  compared with the exact optimum; the reward shapes the style and the ability to learn; a DQN that draws 92 % of
  unseen letters (grid); on the blank sheet, 96 % of the 26 unseen letters.

### 2. Literature review (≈ 1.25 p)
- **Show:** where the project comes from. For each theme: what these works do, and what we take from them (or how
  we differ). References in "References" below.
- **Themes:**
  - MDPs and dynamic programming: Bellman (1957), Howard (1960), Puterman (1994), Sutton & Barto (2018) → our
    reference optimum (value iteration).
  - Learning without a model: Monte Carlo (Metropolis & Ulam, 1949; Singh & Sutton, 1996), TD learning (Sutton,
    1988), Q-learning (Watkins & Dayan, 1992), SARSA (Rummery & Niranjan, 1994), reward shaping (Ng et al., 1999)
    → the algorithms we compare.
  - Deep RL: experience replay (Lin, 1992), DQN (Mnih et al., 2015), Double DQN (van Hasselt et al., 2016),
    convolutional networks (LeCun et al., 1998), gradient clipping (Pascanu et al., 2013) → our DQNs. Policy
    gradient (Williams, 1992) and PPO (Schulman et al., 2017): not used, an opening for continuous moves.
  - Generalisation: the goal as an input of the network (Schaul et al., 2015), diverse training tasks (Cobbe et
    al., 2019) → training on random figures.
  - Agents that draw, handwriting: SPIRAL (Ganin et al., 2018), Learning to Paint (Huang et al., 2019), sketching
    with a DQN (Zhou et al., 2018), characters as strokes (Lake et al., 2015), MNIST / EMNIST (LeCun et al., 1998;
    Cohen et al., 2017) → our inspiration. Difference to point out: they work at large scale (GPU, continuous
    strokes); we start from a small MDP solved exactly, then a pixel version trained on a laptop CPU.

### 3. Methodology (≈ 3.75 p)
- **3.1 The MDP (≈ 1 p).** 5×5 grid, letter = set of segments, state = (pen position, pen down, segments drawn),
  17 actions, deterministic, γ = 0.99, 50-step limit, natural starting points, reward table. To justify: segments
  rather than cells, why 5×5 (2^segments states), segments outside the letter not stored. Details: "Project choices
  (MDP definition)". *Figure:* `fig01_letters.png`.
- **3.2 Algorithms (≈ 1.25 p).** Value iteration; Q-learning (eq. 6.8); SARSA (eq. 6.7); first-visit Monte Carlo
  (average or constant step); ε-greedy; DQN (replay memory, target network, Huber loss, 170-number observation);
  random figures + rich observation (266 numbers). Use the course vocabulary (bootstrapping / sampling, table in
  "Algorithm comparison") and link the DQN loss to the course's (target network and Huber loss added, see
  "Step 7a"). Pseudo-codes in the appendix.
- **3.3 Experimental setup (≈ 0.75 p).** Hyperparameter table, episodes / steps, seeds; evaluation = greedy policy
  from every starting point, % of the optimum v*, "perfect drawing" = complete + nothing outside + STOP; laptop CPU
  and computation times. Details: "Key results", one section per step.
- **3.4 The blank sheet (≈ 0.75 p).** 28×28 canvas; 81 actions (16 directions × 1, 2 or 5 px, pen up or down, +
  STOP); target = image of the letter, 1-pixel tolerance; rewards (+0.2 per new pixel covered, −0.2 per pixel
  outside…); convolutional Double DQN with stabilised settings; view centred on the pen (41×41); network chosen on 50
  held-out figures. Details: "Step 9a" to "Step 9c".

### 4. Results (≈ 6 p)
- **4.1 The optimum (≈ 0.5 p).** Every letter perfect, in a single stroke; exact convergence in 13–23 iterations.
  *Figures:* `fig02_optimal_drawings.png`, `fig03_convergence.png`.
- **4.2 Tabular methods (≈ 1.25 p).** Q-learning: every letter, 93–100 % of v*, first perfect drawing after
  5,000–12,500 episodes. SARSA: same quality, 1.5× slower, unstable. Monte Carlo: fails on every letter ("giving up"
  trap: draws a few segments and STOP). *Figures:* `fig04_q_learning_curves.png`, `fig05_tabular_comparison.png`,
  `fig06_monte_carlo_trap.png`, `fig07_method_summary.png`.
- **4.3 Reward experiments (≈ 1 p).** Going-back penalty −0.5 → human-like strokes (Q-learning: right number of
  strokes for 9 letters out of 10). Early-stop penalty: Monte Carlo learns L from −2, T and X only at −5. Delayed
  reward: Q-learning 8 letters out of 10, 2–3× slower; SARSA only L; Monte Carlo nothing. *Figures:*
  `fig08_writing_style.png`, `fig09_early_stop_penalty.png`, `fig10_delayed_reward.png`.
- **4.4 DQN on the 10 letters (≈ 0.5 p).** One network for all letters, 87–100 % of v*, ~1,200 episodes per letter
  (vs 5,000–12,500), but unstable (90 → 43 → 87 % at the end). *Figure:* `fig11_dqn_learning.png`.
- **4.5 Generalisation on the grid (≈ 1.25 p).** Unseen letters: 0 % (never STOP). 2×2 design: only random figures +
  rich observation generalise, 92 % of the unseen letters; either one alone 0 %. *Figures:*
  `fig14_unseen_failures_07.png`, `fig12_generalisation.png`, `fig13_unseen_letters.png`.
- **4.6 The blank sheet (≈ 1.5 p).** Whole sheet 0 % perfect vs sheet seen from the pen 100 % (L, T). Trained on the
  10 letters: 100 % on them, 0 % on the 16 others. Trained on random figures only: 96 % of the 26 unseen letters
  (63 episodes out of 66); 3 failures (X, A: loops). *Figures:* `fig15_blank_sheet_views.png`,
  `fig16_blank_sheet_generalisation.png`, `fig17_blank_sheet_letters.png`.

### 5. Discussion (≈ 2 p)
- **Algorithms:** advantages and disadvantages of each, with our evidence: table "Algorithm comparison" above.
- **Points to discuss:** the Monte Carlo trap; SARSA's caution not observed (no "cliff" in our task); the reward
  defines the style; DQN instability (deadly triad); generalisation needs diverse targets *and* a good
  representation (true on the grid and on the blank sheet); specialist vs generalist; choosing the network on a
  validation set; cost (74 min for 10 letters vs 8.9 h for random figures); the claims of the course on SARSA vs
  Q-learning against our results ("Step 5b"); the two limits of DQN given in the course, seen in our results
  (discrete moves, deterministic loops); sensitivity to the learning rate ("Step 9b").
- **Limitations:** small grid, hand-made letters and starting points, few seeds, CPU budget, hand-designed
  observations, blank-sheet targets that are not real handwriting, discrete moves, greedy loops.
- **Improvements:** several seeds; real handwriting (EMNIST); n-step TD or TD(λ) for the delayed reward (course
  ch. 6); continuous moves and stochastic policies with policy gradient (REINFORCE; Williams, 1992), actor-critic or
  PPO (Schulman et al., 2017) (course ch. 7); curriculum learning; loop detection.

### 6. Conclusion (≈ 0.5 p)
- Answer the question of the introduction in a few lines: one sentence per main result, one opening.

### References and appendix (not counted)
- **Appendix ideas:** pseudo-codes, per-letter tables, the 16 test letters, extra drawings, the random training
  figures and the failures of the blank sheet.
- **Statement on the use of AI** (the guideline allows AI for code and brainstorming): say that the code was
  written with the help of an AI assistant, then checked, run and explained by the group.

---

## References

APA style. ⚠️ Compiled from memory: **check every entry** (authors, year, title, pages) before putting it in the
report; entries marked *(to verify)* are the least certain. Every reference must be cited in the text.

**Textbook**
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press. — the
  course textbook; equations 6.7 (SARSA) and 6.8 (Q-learning), §4.1 (value iteration), §5.4 (Monte Carlo control),
  §11.3 (deadly triad), Example 6.6 (cliff walking).

**Dynamic programming**
- Bellman, R. (1957). *Dynamic programming*. Princeton University Press.
- Howard, R. A. (1960). *Dynamic programming and Markov processes*. MIT Press.
- Puterman, M. L. (1994). *Markov decision processes: Discrete stochastic dynamic programming*. Wiley.

**Monte Carlo**
- Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. *Journal of the American Statistical Association,
  44*(247), 335–341.
- Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. *Machine Learning,
  22*, 123–158. *(to verify)* — first-visit vs every-visit Monte Carlo.
- Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature, 550*(7676), 354–359. — cited
  in the course (MCTS).

**Temporal-difference learning**
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning, 3*(1), 9–44.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3–4), 279–292.
- Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems* (Technical Report
  CUED/F-INFENG/TR 166). University of Cambridge. — origin of SARSA.

**Reward shaping**
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and
  application to reward shaping. In *Proceedings of the 16th International Conference on Machine Learning (ICML)*.

**Deep reinforcement learning and generalisation**
- Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching.
  *Machine Learning, 8*, 293–321. — experience replay. *(to verify)*
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533. — DQN.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. In
  *Proceedings of the AAAI Conference on Artificial Intelligence*. — Double DQN, used on the blank sheet (step 9b).
- Schaul, T., Horgan, D., Gregor, K., & Silver, D. (2015). Universal value function approximators. In *Proceedings
  of the 32nd International Conference on Machine Learning (ICML)*. — a network that receives the goal as input.
- Cobbe, K., Klimov, O., Hesse, C., Kim, T., & Schulman, J. (2019). Quantifying generalization in reinforcement
  learning. In *Proceedings of the 36th International Conference on Machine Learning (ICML)*. — diverse training
  tasks are needed to generalise.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement
  learning. *Machine Learning, 8*(3–4), 229–256. — REINFORCE, the basic policy-gradient algorithm (course ch. 7);
  cited as an improvement.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization
  algorithms. *arXiv:1707.06347*. — PPO (course ch. 7); cited as an improvement.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on
  Learning Representations (ICLR)*. — the optimiser of the DQN.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. In
  *Proceedings of the 30th International Conference on Machine Learning (ICML)*. — gradient norm clipping, used to
  stabilise the blank-sheet DQN.

**Images and handwriting (blank sheet)**
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document
  recognition. *Proceedings of the IEEE, 86*(11), 2278–2324. — convolutional networks and the 28×28 MNIST digits.
- Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: An extension of MNIST to handwritten letters.
  In *2017 International Joint Conference on Neural Networks (IJCNN)* (pp. 2921–2926). *(to verify)* — 28×28
  handwritten letters, a possible next step.

**Inspiration (cite in the introduction)**
- Ganin, Y., Kulkarni, T., Babuschkin, I., Eslami, S. M. A., & Vinyals, O. (2018). Synthesizing programs for images
  using reinforced adversarial learning. In *Proceedings of the 35th International Conference on Machine Learning
  (ICML)*. — SPIRAL, an RL agent that draws.
- Huang, Z., Heng, W., & Zhou, S. (2019). Learning to paint with model-based deep reinforcement learning. In
  *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.
- Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic
  program induction. *Science, 350*(6266), 1332–1338. — characters drawn stroke by stroke (Omniglot).
- Zhou, T., et al. (2018). Learning to sketch with deep Q networks and demonstrated strokes. *arXiv:1810.05977*.
  *(to verify)*

**Software**
- Harris, C. R., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362.
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95.
- Kluyver, T., et al. (2016). Jupyter Notebooks – a publishing format for reproducible computational workflows. In
  *Positioning and Power in Academic Publishing* (pp. 87–90). IOS Press.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in
  Neural Information Processing Systems 32 (NeurIPS)*.
- Towers, M., et al. (2024). Gymnasium: A standard interface for reinforcement learning environments.
  *arXiv:2407.17032*. — our `reset()` / `step()` interface follows it.

**Course and online resources**
- Mayer, N. (2026). *Reinforcement learning* [Lecture slides, chapters 1–7]. Université de Strasbourg.
- Weng, L. (2018). *A (long) peek into reinforcement learning*. Lil'Log. *(check the URL)*
- Achiam, J. (2018). *Spinning Up in deep RL*. OpenAI. *(check the URL)*
