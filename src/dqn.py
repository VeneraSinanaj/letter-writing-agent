"""
Deep Q-Network (Mnih et al., 2015) for the letter-writing task: ONE network for ALL letters.

The table Q(s, a) is replaced by a neural network Q(o, a; theta). The observation o describes the state
in a way that does not depend on the letter, so that the same network can be used for every letter:
    - the target letter: 72 bits, one per possible unit segment of the 5x5 grid (1 = part of the letter);
    - the segments of the letter already drawn: 72 bits;
    - the pen position: 25 bits (one-hot);
    - pen down at the previous move: 1 bit.
    -> 170 numbers. The network outputs the 17 action values.

Same target as Q-learning:   y = R + gamma * max_a Q_target(o', a)     (y = R after STOP)
Loss (Huber) between Q(o, A; theta) and y, minimised by gradient descent (Adam).

Step 7c adds an optional "rich" observation (266 numbers) and training on random figures:
    - the missing segments (target minus drawn): 72 bits;
    - a local view around the pen: for each of the 8 directions, is the segment in that direction
      missing / already drawn / outside the grid: 24 bits.

Two ingredients make it stable (Mnih et al., 2015):
    - experience replay: transitions (o, a, r, o', terminated) are stored in a memory and the network
      learns on random mini-batches from it, which breaks the correlation between successive steps;
    - target network: a copy of the network, updated only every `target_update` steps, computes the
      target y, so that the target does not move at every update.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .letters import GRID_SIZE, normalize
from .environment import DIRECTIONS, N_ACTIONS, LetterWritingEnv

# ------------------------------------------------------------------ observation
ALL_SEGMENTS = sorted({normalize((r, c), (r + dr, c + dc))
                       for r in range(GRID_SIZE) for c in range(GRID_SIZE) for dr, dc in DIRECTIONS
                       if 0 <= r + dr < GRID_SIZE and 0 <= c + dc < GRID_SIZE})
SEGMENT_INDEX = {seg: i for i, seg in enumerate(ALL_SEGMENTS)}
N_SEGMENTS = len(ALL_SEGMENTS)                        # 72
N_CELLS = GRID_SIZE * GRID_SIZE                       # 25
OBS_SIZE = 2 * N_SEGMENTS + N_CELLS + 1               # 170 (basic observation)
LOCAL_SIZE = 3 * len(DIRECTIONS)                      # 24
RICH_OBS_SIZE = OBS_SIZE + N_SEGMENTS + LOCAL_SIZE    # 266 (rich observation, step 7c)


def obs_size(rich=False):
    return RICH_OBS_SIZE if rich else OBS_SIZE


class Observer:
    """Turns a state of a LetterWritingEnv into the observation (170 numbers, or 266 if rich)."""

    def __init__(self, env, rich=False):
        self.env = env
        self.rich = rich
        self.size = obs_size(rich)
        self.target = np.zeros(N_SEGMENTS, dtype=np.float32)
        self.bit_to_index = np.array([SEGMENT_INDEX[seg] for seg in env.letter.segments])
        self.target[self.bit_to_index] = 1.0

    def __call__(self, state):
        row, col, pen_down, mask = self.env.decode(state)
        obs = np.zeros(self.size, dtype=np.float32)
        obs[:N_SEGMENTS] = self.target
        drawn_bits = [i for i in range(self.env.k) if mask >> i & 1]
        obs[N_SEGMENTS + self.bit_to_index[drawn_bits]] = 1.0
        obs[2 * N_SEGMENTS + row * GRID_SIZE + col] = 1.0
        obs[OBS_SIZE - 1] = pen_down
        if self.rich:
            # missing segments = target segments not drawn yet
            missing = obs[OBS_SIZE:OBS_SIZE + N_SEGMENTS]
            missing[:] = self.target
            missing[self.bit_to_index[drawn_bits]] = 0.0
            # local view: for each direction, [missing, already drawn, outside the grid]
            for d, (dr, dc) in enumerate(DIRECTIONS):
                base = OBS_SIZE + N_SEGMENTS + 3 * d
                r2, c2 = row + dr, col + dc
                if not (0 <= r2 < GRID_SIZE and 0 <= c2 < GRID_SIZE):
                    obs[base + 2] = 1.0
                    continue
                bit = self.env.segment_bit.get(normalize((row, col), (r2, c2)))
                if bit is not None:
                    obs[base + (1 if mask >> bit & 1 else 0)] = 1.0
        return obs


# ------------------------------------------------------------------ network and memory
class QNetwork(nn.Module):
    def __init__(self, hidden=256, input_size=OBS_SIZE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, obs):
        return self.layers(obs)


class ReplayBuffer:
    """Fixed-size memory of transitions; the oldest ones are overwritten when it is full."""

    def __init__(self, capacity, size=OBS_SIZE):
        self.obs = np.zeros((capacity, size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, size), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i], self.action[i], self.reward[i], self.next_obs[i], self.terminated[i] = \
            obs, action, reward, next_obs, terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(self.size, size=batch_size)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.action[idx]),
                torch.from_numpy(self.reward[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.terminated[idx]))


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def greedy_episode(net, env, start, gamma=0.99, rich=False):
    """Play the greedy policy of the network on env from `start`. Returns the same metrics as play_policy."""
    observe = Observer(env, rich)
    state = env.reset(start=start)
    total, discounted, t, done, terminated = 0.0, 0.0, 0, False, False
    while not done:
        q = net(torch.from_numpy(observe(state)).unsqueeze(0))
        state, r, terminated, truncated, _ = env.step(int(q.argmax()))
        total += r
        discounted += gamma**t * r
        t += 1
        done = terminated or truncated
    strokes, off_letter, pen_was_down = 0, 0, False
    for info in env.history:
        if info["event"] != "move":
            continue
        if info["pen_down"]:
            strokes += not pen_was_down
            off_letter += info["segment"] == "off_letter"
        pen_was_down = info["pen_down"]
    drawn = env.drawn_segments()
    return {"history": list(env.history), "total_reward": total, "discounted_return": discounted,
            "steps": env.steps, "strokes": strokes, "drawn": drawn, "off_letter": off_letter,
            "completed": bool(terminated and drawn == env.k)}


def evaluate(net, letters, rewards, gamma=0.99, rich=False):
    """Greedy policy from every natural starting point of every letter (names or Letter objects).
    Returns {key: (mean discounted return, fraction of perfect drawings)}."""
    out = {}
    for i, letter in enumerate(letters):
        env = LetterWritingEnv(letter, rewards=rewards)
        key = letter if isinstance(letter, str) else f"{letter.name}_{i}"
        eps = [greedy_episode(net, env, start, gamma, rich) for start in env.letter.start_points]
        out[key] = (float(np.mean([e["discounted_return"] for e in eps])),
                    float(np.mean([e["completed"] and e["off_letter"] == 0 for e in eps])))
    return out


# ------------------------------------------------------------------ training
def train_dqn(letters, rewards=None, total_steps=300_000, gamma=0.99, lr=5e-4, batch_size=64,
              buffer_size=100_000, learning_starts=5_000, train_every=4, target_update=1_000,
              eps_start=1.0, eps_end=0.05, eps_fraction=0.6, hidden=256, eval_every=10_000,
              seed=0, verbose=True, rich_obs=False, task_sampler=None, eval_letters=None, eval_figures=None):
    """Train one DQN on several targets at once.
    - letters: the training letters (a random one at each episode), used when task_sampler is None;
    - task_sampler(rng) -> Letter: a new target at each episode (e.g. random figures, step 7c);
    - rich_obs: use the 266-number observation instead of the 170-number one;
    - eval_letters / eval_figures: what the greedy policy is evaluated on (default: the training letters).
    Returns a dict with the trained network and the evaluation curves."""
    from .environment import Rewards
    rewards = rewards or Rewards()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    size = obs_size(rich_obs)
    if task_sampler is None:
        envs = [LetterWritingEnv(name, rewards=rewards, seed=seed + i) for i, name in enumerate(letters)]
        observers = [Observer(env, rich_obs) for env in envs]

        def new_task():
            k = rng.integers(len(envs))
            return envs[k], observers[k]
    else:
        def new_task():
            env = LetterWritingEnv(task_sampler(rng), rewards=rewards)
            env.rng = rng
            return env, Observer(env, rich_obs)
    eval_letters = list(letters) if eval_letters is None else eval_letters

    net, target_net = QNetwork(hidden, size), QNetwork(hidden, size)
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, size)

    eval_steps, eval_results, eval_fig_results, episode_returns, episode_end_steps = [], [], [], [], []
    t0 = time.perf_counter()
    env, observe = new_task()
    obs = observe(env.reset())
    episode_return = 0.0

    for step in range(1, total_steps + 1):
        # --- act (epsilon-greedy) ---
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / (eps_fraction * total_steps))
        if rng.random() < epsilon:
            action = int(rng.integers(N_ACTIONS))
        else:
            with torch.no_grad():
                action = int(net(torch.from_numpy(obs).unsqueeze(0)).argmax())
        state, reward, terminated, truncated, _ = env.step(action)
        next_obs = observe(state)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        episode_return += reward
        if terminated or truncated:          # new episode, on a random letter
            episode_returns.append(episode_return)
            episode_end_steps.append(step)
            episode_return = 0.0
            env, observe = new_task()
            obs = observe(env.reset())

        # --- learn from a random mini-batch of the memory ---
        if step >= learning_starts and step % train_every == 0:
            o, a, r, o2, term = buffer.sample(batch_size, rng)
            with torch.no_grad():
                y = r + gamma * (1.0 - term) * target_net(o2).max(dim=1).values   # target: R + gamma max Q_target
            q = net(o).gather(1, a.unsqueeze(1)).squeeze(1)                       # Q(o, a; theta)
            loss = nn.functional.smooth_l1_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % target_update == 0:
            target_net.load_state_dict(net.state_dict())

        # --- evaluation of the greedy policy ---
        if step % eval_every == 0:
            net.eval()
            res = evaluate(net, eval_letters, rewards, gamma, rich_obs)
            res_fig = evaluate(net, eval_figures, rewards, gamma, rich_obs) if eval_figures else {}
            net.train()
            eval_steps.append(step)
            eval_results.append(res)
            eval_fig_results.append(res_fig)
            if verbose:
                perfect = np.mean([p for _, p in res.values()])
                extra = f"  held-out figures {np.mean([p for _, p in res_fig.values()]):.0%}" if res_fig else ""
                print(f"step {step:>7,}  epsilon {epsilon:.2f}  perfect drawings {perfect:.0%}{extra}  "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)

    return {"net": net, "eval_steps": np.array(eval_steps), "eval_results": eval_results,
            "eval_fig_results": eval_fig_results,
            "episode_returns": np.array(episode_returns), "episode_end_steps": np.array(episode_end_steps),
            "time": time.perf_counter() - t0}
