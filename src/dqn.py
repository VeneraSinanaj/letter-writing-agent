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
OBS_SIZE = 2 * N_SEGMENTS + N_CELLS + 1               # 170


class Observer:
    """Turns a state of a LetterWritingEnv into the 170-number observation."""

    def __init__(self, env):
        self.env = env
        self.target = np.zeros(N_SEGMENTS, dtype=np.float32)
        self.bit_to_index = np.array([SEGMENT_INDEX[seg] for seg in env.letter.segments])
        self.target[self.bit_to_index] = 1.0

    def __call__(self, state):
        row, col, pen_down, mask = self.env.decode(state)
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[:N_SEGMENTS] = self.target
        drawn_bits = [i for i in range(self.env.k) if mask >> i & 1]
        obs[N_SEGMENTS + self.bit_to_index[drawn_bits]] = 1.0
        obs[2 * N_SEGMENTS + row * GRID_SIZE + col] = 1.0
        obs[-1] = pen_down
        return obs


# ------------------------------------------------------------------ network and memory
class QNetwork(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, obs):
        return self.layers(obs)


class ReplayBuffer:
    """Fixed-size memory of transitions; the oldest ones are overwritten when it is full."""

    def __init__(self, capacity):
        self.obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
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
def greedy_episode(net, env, start, gamma=0.99):
    """Play the greedy policy of the network on env from `start`. Returns the same metrics as play_policy."""
    observe = Observer(env)
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


def evaluate(net, letters, rewards, gamma=0.99):
    """Greedy policy from every natural starting point of every letter.
    Returns {letter: (mean discounted return, fraction of perfect drawings)}."""
    out = {}
    for name in letters:
        env = LetterWritingEnv(name, rewards=rewards)
        eps = [greedy_episode(net, env, start, gamma) for start in env.letter.start_points]
        out[name] = (float(np.mean([e["discounted_return"] for e in eps])),
                     float(np.mean([e["completed"] and e["off_letter"] == 0 for e in eps])))
    return out


# ------------------------------------------------------------------ training
def train_dqn(letters, rewards=None, total_steps=300_000, gamma=0.99, lr=5e-4, batch_size=64,
              buffer_size=100_000, learning_starts=5_000, train_every=4, target_update=1_000,
              eps_start=1.0, eps_end=0.05, eps_fraction=0.6, hidden=256, eval_every=10_000,
              seed=0, verbose=True):
    """Train one DQN on several letters at once (a random letter at each episode).
    Returns a dict with the trained network and the evaluation curves."""
    from .environment import Rewards
    rewards = rewards or Rewards()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    envs = [LetterWritingEnv(name, rewards=rewards, seed=seed + i) for i, name in enumerate(letters)]
    observers = [Observer(env) for env in envs]

    net, target_net = QNetwork(hidden), QNetwork(hidden)
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    eval_steps, eval_results, episode_returns, episode_end_steps = [], [], [], []
    t0 = time.perf_counter()
    k = rng.integers(len(envs))
    env, observe = envs[k], observers[k]
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
            k = rng.integers(len(envs))
            env, observe = envs[k], observers[k]
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
            res = evaluate(net, letters, rewards, gamma)
            net.train()
            eval_steps.append(step)
            eval_results.append(res)
            if verbose:
                perfect = np.mean([p for _, p in res.values()])
                print(f"step {step:>7,}  epsilon {epsilon:.2f}  perfect drawings {perfect:.0%}  "
                      f"({time.perf_counter() - t0:.0f} s)", flush=True)

    return {"net": net, "eval_steps": np.array(eval_steps), "eval_results": eval_results,
            "episode_returns": np.array(episode_returns), "episode_end_steps": np.array(episode_end_steps),
            "time": time.perf_counter() - t0}
