"""
Tabular model-free methods (course, chapters 5 and 6; Sutton & Barto, chapters 5 and 6).

The agent does NOT know the model: it only uses env.reset() and env.step(), and learns a
table Q(s, a) = its own estimate of the return obtained by taking action a in state s.

Q-learning (off-policy TD control, Sutton & Barto eq. 6.8), after every step:
    Q(S, A) <- Q(S, A) + alpha * [ R + gamma * max_a Q(S', a) - Q(S, A) ]
    - the term in brackets is the TD error: what was observed minus what was expected;
    - after STOP (terminated) there is no future: the target is just R;
    - after the step limit (truncated) the state is not terminal: we keep gamma * max_a Q(S', a).

Exploration: epsilon-greedy. With probability epsilon a random action, otherwise the greedy
action argmax_a Q(S, a). Epsilon decreases linearly from eps_start to eps_end during the first
decay_fraction of the episodes, then stays at eps_end.
"""
import time

import numpy as np

from .environment import LetterWritingEnv
from .evaluation import play_policy, greedy_policy


def epsilon_schedule(episode, n_episodes, eps_start=1.0, eps_end=0.05, decay_fraction=0.8):
    """Linear decay from eps_start to eps_end over the first decay_fraction of the episodes."""
    progress = min(1.0, episode / (decay_fraction * n_episodes))
    return eps_start + progress * (eps_end - eps_start)


def epsilon_greedy(q_row, epsilon, rng):
    """Random action with probability epsilon, otherwise the best action.
    Ties are broken at random (at the start all Q values are 0: without this, the agent
    would always choose action 0)."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def evaluate_greedy(eval_env, Q, gamma):
    """Play the greedy policy from every natural starting point (no exploration).
    Returns (mean discounted return, fraction of starts where the letter is drawn perfectly)."""
    policy = greedy_policy(Q)
    returns, perfect = [], []
    for start in eval_env.letter.start_points:
        ep = play_policy(eval_env, policy, start, gamma=gamma)
        returns.append(ep["discounted_return"])
        perfect.append(ep["completed"] and ep["off_letter"] == 0)
    return float(np.mean(returns)), float(np.mean(perfect))


def q_learning(env, n_episodes=30_000, alpha=0.5, gamma=0.99, eps_start=1.0, eps_end=0.05,
               decay_fraction=0.8, eval_every=500, snapshot_episodes=(), seed=0):
    """Train a Q-learning agent on env. Returns a dict with:
        Q                  the learned table, shape (n_states, n_actions)
        eval_episodes      episodes at which the greedy policy was evaluated
        eval_return        mean discounted return of the greedy policy over the starting points
        eval_perfect       fraction of starting points where the greedy policy draws the letter perfectly
        train_returns      total reward of each training episode (with exploration)
        snapshots          {episode: greedy episode from the first starting point} (for drawings)
        time               training time in seconds
    """
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed)  # random starting points, reproducible
    eval_env = LetterWritingEnv(env.letter, env.rewards, env.max_steps)  # separate env for evaluation
    Q = np.zeros((env.n_states, env.n_actions))
    eval_episodes, eval_return, eval_perfect, train_returns, snapshots = [], [], [], [], {}

    t0 = time.perf_counter()
    for episode in range(n_episodes + 1):
        # --- evaluation of the current greedy policy (not counted as training) ---
        if episode in snapshot_episodes:
            snapshots[episode] = play_policy(eval_env, greedy_policy(Q), env.letter.start_points[0], gamma)
        if episode % eval_every == 0:
            g, p = evaluate_greedy(eval_env, Q, gamma)
            eval_episodes.append(episode)
            eval_return.append(g)
            eval_perfect.append(p)
        if episode == n_episodes:
            break

        # --- one training episode ---
        epsilon = epsilon_schedule(episode, n_episodes, eps_start, eps_end, decay_fraction)
        s = env.reset()
        total, done = 0.0, False
        while not done:
            a = epsilon_greedy(Q[s], epsilon, rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            # Q-learning target: R + gamma * max_a Q(S', a), or just R after a terminal step
            target = r if terminated else r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
            total += r
            done = terminated or truncated
        train_returns.append(total)

    return {"Q": Q, "eval_episodes": np.array(eval_episodes), "eval_return": np.array(eval_return),
            "eval_perfect": np.array(eval_perfect), "train_returns": np.array(train_returns),
            "snapshots": snapshots, "time": time.perf_counter() - t0}
