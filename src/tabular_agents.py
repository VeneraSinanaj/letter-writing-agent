"""
Tabular model-free methods (course, chapters 5 and 6; Sutton & Barto, chapters 5 and 6).

The agent does NOT know the model: it only uses env.reset() and env.step(), and learns a
table Q(s, a) = its own estimate of the return obtained by taking action a in state s.

Q-learning (off-policy TD control, Sutton & Barto eq. 6.8), after every step:
    Q(S, A) <- Q(S, A) + alpha * [ R + gamma * max_a Q(S', a) - Q(S, A) ]

SARSA (on-policy TD control, Sutton & Barto eq. 6.7), after every step:
    Q(S, A) <- Q(S, A) + alpha * [ R + gamma * Q(S', A') - Q(S, A) ]
    where A' is the action the agent will actually play in S' (chosen epsilon-greedily).

    For both: the term in brackets is the TD error (what was observed minus what was expected);
    after STOP (terminated) there is no future, the target is just R; after the step limit
    (truncated) the state is not terminal, so the target keeps the estimated value of S'.

Monte Carlo control (on-policy, first-visit, course chapter 5; Sutton & Barto section 5.4):
    play a whole episode, then go backwards and compute the return G_t = R_{t+1} + gamma * G_{t+1};
    for the first visit of each pair (s, a) in the episode:
        sample average (course):  N(s, a) += 1,  Q(s, a) <- Q(s, a) + [G - Q(s, a)] / N(s, a)
        constant step:            Q(s, a) <- Q(s, a) + alpha * [G - Q(s, a)]
    The sample average gives the same weight to every return, even to old returns collected by
    a worse policy; the constant step forgets them gradually (course chapter 2, nonstationary problems).
    Episodes cut at the step limit give an incomplete return: an approximation, since Monte Carlo
    assumes terminating episodes.

Exploration (all methods): epsilon-greedy. With probability epsilon a random action, otherwise the
greedy action argmax_a Q(S, a). Epsilon decreases linearly from eps_start to eps_end during the
first decay_fraction of the episodes, then stays at eps_end.
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


def _train(env, run_episode, n_episodes, gamma, eps_start, eps_end, decay_fraction,
           eval_every, snapshot_episodes, seed):
    """Training loop shared by all methods: exploration schedule, evaluation, snapshots.
    run_episode(Q, epsilon, rng) plays one training episode, updates Q and returns its total reward.

    Returns a dict with:
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
        train_returns.append(run_episode(Q, epsilon, rng))

    return {"Q": Q, "eval_episodes": np.array(eval_episodes), "eval_return": np.array(eval_return),
            "eval_perfect": np.array(eval_perfect), "train_returns": np.array(train_returns),
            "snapshots": snapshots, "time": time.perf_counter() - t0}


def q_learning(env, n_episodes=30_000, alpha=0.5, gamma=0.99, eps_start=1.0, eps_end=0.05,
               decay_fraction=0.8, eval_every=500, snapshot_episodes=(), seed=0):
    """Q-learning (off-policy TD control). See the module docstring for the update rule."""

    def run_episode(Q, epsilon, rng):
        s = env.reset()
        total, done = 0.0, False
        while not done:
            a = epsilon_greedy(Q[s], epsilon, rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            # target: R + gamma * max_a Q(S', a), or just R after a terminal step
            target = r if terminated else r + gamma * Q[s_next].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
            total += r
            done = terminated or truncated
        return total

    return _train(env, run_episode, n_episodes, gamma, eps_start, eps_end, decay_fraction,
                  eval_every, snapshot_episodes, seed)


def sarsa(env, n_episodes=30_000, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
          decay_fraction=0.8, eval_every=500, snapshot_episodes=(), seed=0):
    """SARSA (on-policy TD control). See the module docstring for the update rule.
    Default alpha = 0.1 (not 0.5 as for Q-learning): the SARSA target depends on the next action,
    which is sometimes random (exploration), so it is noisy and needs a smaller step size."""

    def run_episode(Q, epsilon, rng):
        s = env.reset()
        a = epsilon_greedy(Q[s], epsilon, rng)
        total = 0.0
        while True:
            s_next, r, terminated, truncated, _ = env.step(a)
            total += r
            if terminated:                      # STOP: no future, target = R
                Q[s, a] += alpha * (r - Q[s, a])
                break
            a_next = epsilon_greedy(Q[s_next], epsilon, rng)   # the action really played next
            # target: R + gamma * Q(S', A')
            Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
            if truncated:
                break
            s, a = s_next, a_next
        return total

    return _train(env, run_episode, n_episodes, gamma, eps_start, eps_end, decay_fraction,
                  eval_every, snapshot_episodes, seed)


def monte_carlo(env, n_episodes=30_000, alpha=None, gamma=0.99, eps_start=1.0, eps_end=0.05,
                decay_fraction=0.8, eval_every=500, snapshot_episodes=(), seed=0):
    """On-policy first-visit Monte Carlo control.
    alpha=None: sample average of the returns (course formula); alpha=a number: constant step size."""
    N = np.zeros((env.n_states, env.n_actions), dtype=np.int64) if alpha is None else None

    def run_episode(Q, epsilon, rng):
        # 1) play a whole episode without learning
        s = env.reset()
        episode, done = [], False
        while not done:
            a = epsilon_greedy(Q[s], epsilon, rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s_next
            done = terminated or truncated
        # 2) first visit of each (state, action) pair in the episode
        first_visit = {}
        for t, (s, a, _) in enumerate(episode):
            first_visit.setdefault((s, a), t)
        # 3) go backwards: G_t = R_{t+1} + gamma * G_{t+1}, update Q at first visits
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G
            if first_visit[(s, a)] == t:
                if alpha is None:
                    N[s, a] += 1
                    Q[s, a] += (G - Q[s, a]) / N[s, a]   # average of all returns observed so far
                else:
                    Q[s, a] += alpha * (G - Q[s, a])     # constant step: recent returns weigh more
        return sum(r for _, _, r in episode)

    return _train(env, run_episode, n_episodes, gamma, eps_start, eps_end, decay_fraction,
                  eval_every, snapshot_episodes, seed)
