"""
DQN with a convolutional network for the blank sheet (step 9b).

The network reads the 4 images of the observation (target, drawing, pen position, pen down) and outputs the
81 action values. Convolutions apply the same small filters everywhere on the canvas: a natural way to learn a
*local* rule ("is a piece of the letter missing next to my pen?"), the lesson of step 7c.

Same algorithm as src/dqn.py (experience replay, target network, epsilon-greedy), plus:
- Double DQN (van Hasselt et al., 2016): the online network chooses the next action, the target network
  evaluates it -> y = R + gamma * Q_target(o', argmax_a Q(o', a)). It reduces the over-estimation of the max.
- a compact replay memory: images are stored as bits (np.packbits) and rebuilt when a mini-batch is drawn;
- optionally, an EGOCENTRIC observation (the sheet seen from the pen, like the local view of step 7c): a 41x41
  window centred on the pen with 4 channels: missing part of the letter, ink, inside-the-canvas mask, pen down.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .free_draw import FreeDrawEnv, FreeRewards, N_ACTIONS, SIZE

N_PIX = SIZE * SIZE


class ConvQNetwork(nn.Module):
    """3 convolutional layers (28x28 -> 28x28 -> 14x14 -> 7x7), then 2 fully connected layers.
    The first convolution keeps the full resolution: a lighter network that halves the resolution at once
    was 2x faster but much worse (61 % vs 99 % of L covered after 20,000 steps): lines are 1 pixel wide."""

    def __init__(self, n_actions=N_ACTIONS, input_size=SIZE):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),            # full resolution
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), # 1/2
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), # 1/4
        )
        side = (((input_size + 1) // 2) + 1) // 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * side * side, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, obs):
        return self.head(self.conv(obs))


# ------------------------------------------------------------------ compact replay memory
def snapshot(env):
    """The state of a FreeDrawEnv in a compact form: (target bits, ink bits, pen row, pen col, pen down)."""
    return (np.packbits(env.target.ravel()), np.packbits(env.ink.ravel()), env.pen[0], env.pen[1], int(env.pen_down))


def rebuild(target_bits, ink_bits, pen, pen_down):
    """Rebuild a batch of observations (B, 4, 28, 28) from compact states."""
    B = len(pen)
    obs = np.zeros((B, 4, SIZE, SIZE), dtype=np.float32)
    obs[:, 0] = np.unpackbits(target_bits, axis=1)[:, :N_PIX].reshape(B, SIZE, SIZE)
    obs[:, 1] = np.unpackbits(ink_bits, axis=1)[:, :N_PIX].reshape(B, SIZE, SIZE)
    obs[np.arange(B), 2, pen[:, 0], pen[:, 1]] = 1.0
    obs[:, 3] = pen_down[:, None, None]
    return torch.from_numpy(obs)


EGO = 41                        # egocentric window, the pen at the centre: sees the whole letter area (pixels 4-24) from any pen position on the letter
HALF = EGO // 2


def rebuild_egocentric(target_bits, ink_bits, pen, pen_down):
    """Batch of egocentric observations (B, 4, 29, 29): the sheet seen from the pen.
    Channels: missing part of the letter (skeleton not yet covered), ink, inside-the-canvas mask, pen down."""
    B = len(pen)
    target = np.unpackbits(target_bits, axis=1)[:, :N_PIX].reshape(B, SIZE, SIZE).astype(np.float32)
    ink = np.unpackbits(ink_bits, axis=1)[:, :N_PIX].reshape(B, SIZE, SIZE).astype(np.float32)
    near_ink = torch.nn.functional.max_pool2d(torch.from_numpy(ink)[:, None], 3, stride=1, padding=1)[:, 0].numpy()
    missing = target * (1.0 - near_ink)          # skeleton pixels not yet covered (1-pixel tolerance)
    full = np.zeros((B, 3, SIZE + 2 * HALF, SIZE + 2 * HALF), dtype=np.float32)
    full[:, 0, HALF:HALF + SIZE, HALF:HALF + SIZE] = missing
    full[:, 1, HALF:HALF + SIZE, HALF:HALF + SIZE] = ink
    full[:, 2, HALF:HALF + SIZE, HALF:HALF + SIZE] = 1.0
    obs = np.zeros((B, 4, EGO, EGO), dtype=np.float32)
    for i in range(B):
        r, c = pen[i]
        obs[i, :3] = full[i, :, r:r + EGO, c:c + EGO]
    obs[:, 3] = pen_down[:, None, None]
    return torch.from_numpy(obs)


def observe(env, egocentric):
    """Observation of the current state of env, as a tensor (1, 4, H, W)."""
    if not egocentric:
        return torch.from_numpy(env.observation()).unsqueeze(0)
    t, ink, r, c, pd = snapshot(env)
    return rebuild_egocentric(t[None], ink[None], np.array([[r, c]]), np.array([pd], dtype=np.float32))


class FreeReplayBuffer:
    def __init__(self, capacity, egocentric=False):
        self.build = rebuild_egocentric if egocentric else rebuild
        nb = (N_PIX + 7) // 8
        self.target = np.zeros((capacity, nb), np.uint8)
        self.ink = np.zeros((capacity, nb), np.uint8)
        self.next_ink = np.zeros((capacity, nb), np.uint8)
        self.pen = np.zeros((capacity, 2), np.int64)
        self.next_pen = np.zeros((capacity, 2), np.int64)
        self.pen_down = np.zeros(capacity, np.float32)
        self.next_pen_down = np.zeros(capacity, np.float32)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, s, action, reward, s_next, terminated):
        i = self.pos
        self.target[i], self.ink[i], self.pen[i, 0], self.pen[i, 1], self.pen_down[i] = s
        _, self.next_ink[i], self.next_pen[i, 0], self.next_pen[i, 1], self.next_pen_down[i] = s_next
        self.action[i], self.reward[i], self.terminated[i] = action, reward, terminated
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(self.size, size=batch_size)
        obs = self.build(self.target[idx], self.ink[idx], self.pen[idx], self.pen_down[idx])
        next_obs = self.build(self.target[idx], self.next_ink[idx], self.next_pen[idx], self.next_pen_down[idx])
        return (obs, torch.from_numpy(self.action[idx]), torch.from_numpy(self.reward[idx]), next_obs,
                torch.from_numpy(self.terminated[idx]))


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def greedy_free_episode(net, env, start=None, gamma=0.99, egocentric=False):
    """Play the greedy policy of the network. Returns coverage, precision, perfect, return, history..."""
    env.reset(start=start)
    total, discounted, t, done, terminated = 0.0, 0.0, 0, False, False
    while not done:
        a = int(net(observe(env, egocentric)).argmax())
        _, r, terminated, truncated, _ = env.step(a)
        total += r
        discounted += gamma**t * r
        t += 1
        done = terminated or truncated
    return {"history": list(env.history), "total_reward": total, "discounted_return": discounted,
            "steps": env.steps, "coverage": env.coverage(), "precision": env.precision(),
            "perfect": env.perfect(terminated), "stopped": bool(terminated)}


def evaluate_free(net, letters, rewards=FreeRewards(), gamma=0.99, egocentric=False):
    """Greedy policy from every natural starting point of every letter (names or Letter objects).
    Returns {key: dict(coverage, precision, perfect, stop)} averaged over the starting points."""
    out = {}
    for i, letter in enumerate(letters):
        env = FreeDrawEnv(letter, rewards=rewards)
        key = letter if isinstance(letter, str) else f"{letter.name}_{i}"
        eps = [greedy_free_episode(net, env, s, gamma, egocentric) for s in env.start_pixels]
        out[key] = {m: float(np.mean([e[m] for e in eps])) for m in ("coverage", "precision", "perfect", "stopped")}
    return out


# ------------------------------------------------------------------ training
def train_free_dqn(letters=None, task_sampler=None, rewards=FreeRewards(), total_steps=300_000, gamma=0.99,
                   lr=5e-4, batch_size=64, buffer_size=100_000, learning_starts=5_000, train_every=4,
                   target_update=2_000, eps_start=1.0, eps_end=0.05, eps_fraction=0.5, double=True,
                   eval_letters=None, eval_figures=None, eval_every=20_000, max_steps=60, seed=0, verbose=True,
                   threads=4, grad_clip=10.0, egocentric=False):
    """Train a convolutional DQN on the blank sheet.
    - letters: training letters (a random one at each episode), used when task_sampler is None;
    - task_sampler(rng) -> Letter: a new target at each episode (random figures, step 9c)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)   # convolutions are faster with a few CPU threads
    if task_sampler is None:
        envs = [FreeDrawEnv(name, rewards=rewards, max_steps=max_steps, seed=seed + i) for i, name in enumerate(letters)]

        def new_env():
            return envs[rng.integers(len(envs))]
    else:
        def new_env():
            env = FreeDrawEnv(task_sampler(rng), rewards=rewards, max_steps=max_steps)
            env.rng = rng
            return env
    eval_letters = list(letters) if eval_letters is None else eval_letters

    size = EGO if egocentric else SIZE
    net, target_net = ConvQNetwork(input_size=size), ConvQNetwork(input_size=size)
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    buffer = FreeReplayBuffer(buffer_size, egocentric)

    eval_steps, eval_results, eval_fig_results, episode_returns = [], [], [], []
    t0 = time.perf_counter()
    env = new_env()
    env.reset()
    s = snapshot(env)
    ep_return = 0.0
    for step in range(1, total_steps + 1):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * step / (eps_fraction * total_steps))
        if rng.random() < epsilon:
            action = int(rng.integers(N_ACTIONS))
        else:
            with torch.no_grad():
                action = int(net(observe(env, egocentric)).argmax())
        _, reward, terminated, truncated, _ = env.step(action)
        s_next = snapshot(env)
        buffer.add(s, action, reward, s_next, float(terminated))
        s = s_next
        ep_return += reward
        if terminated or truncated:
            episode_returns.append(ep_return)
            ep_return = 0.0
            env = new_env()
            env.reset()
            s = snapshot(env)

        if step >= learning_starts and step % train_every == 0:
            o, a, r, o2, term = buffer.sample(batch_size, rng)
            with torch.no_grad():
                if double:   # online network chooses, target network evaluates
                    a2 = net(o2).argmax(dim=1, keepdim=True)
                    q_next = target_net(o2).gather(1, a2).squeeze(1)
                else:
                    q_next = target_net(o2).max(dim=1).values
                y = r + gamma * (1.0 - term) * q_next
            q = net(o).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:   # limit the size of each correction (stabilises learning)
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
        if step % target_update == 0:
            target_net.load_state_dict(net.state_dict())

        if step % eval_every == 0:
            net.eval()
            res = evaluate_free(net, eval_letters, rewards, gamma, egocentric)
            res_fig = evaluate_free(net, eval_figures, rewards, gamma, egocentric) if eval_figures else {}
            net.train()
            eval_steps.append(step)
            eval_results.append(res)
            eval_fig_results.append(res_fig)
            if verbose:
                perfect = np.mean([v["perfect"] for v in res.values()])
                cover = np.mean([v["coverage"] for v in res.values()])
                prec = np.mean([v["precision"] for v in res.values()])
                extra = f"  figures perfect {np.mean([v['perfect'] for v in res_fig.values()]):.0%}" if res_fig else ""
                print(f"step {step:>7,}  eps {epsilon:.2f}  perfect {perfect:.0%}  coverage {cover:.0%}  "
                      f"precision {prec:.0%}{extra}  ({time.perf_counter() - t0:.0f} s)", flush=True)

    return {"net": net, "eval_steps": np.array(eval_steps), "eval_results": eval_results,
            "eval_fig_results": eval_fig_results, "episode_returns": np.array(episode_returns),
            "time": time.perf_counter() - t0}
