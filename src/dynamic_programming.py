"""
Dynamic programming: value iteration (course, chapter 4).

Bellman optimality equation:
    v*(s) = max_a  sum_{s', r} p(s', r | s, a) [ r + gamma * v*(s') ]

Our environment is deterministic: each pair (s, a) leads to a single (s', r) with
probability 1, so the sum has a single term:
    v*(s) = max_a [ r(s, a) + gamma * v*(s'(s, a)) ]
and after STOP (a terminal transition) there is no future: v(s') is replaced by 0.

Value iteration repeats this update for every state until the values stop changing:
    v_{k+1}(s) = max_a [ r(s, a) + gamma * v_k(s'(s, a)) ]       until  max_s |v_{k+1}(s) - v_k(s)| < theta
and the optimal policy is then read off:
    pi*(s) = argmax_a [ r(s, a) + gamma * v*(s'(s, a)) ]

Implementation note: the course notebook (Frozen Lake) loops over the states one by one
and overwrites V(s) immediately ("in-place" version). Here all states are updated at once
from the previous V with numpy arrays ("two-array" version, Sutton & Barto, section 4.1).
Both versions converge to the same v*; the array version is much faster for 200,000 states.
"""
import time

import numpy as np


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10_000, model=None):
    """Returns a dict with:
        V        optimal state values v*(s), shape (n_states,)
        Q        optimal action values q*(s, a) = r(s, a) + gamma * v*(s'), shape (n_states, n_actions)
        policy   optimal action in each state, shape (n_states,)
        deltas   max_s |v_{k+1}(s) - v_k(s)| at each iteration (convergence curve)
        model_time, solve_time   in seconds
    """
    t0 = time.perf_counter()
    next_state, reward, terminated = model if model is not None else env.build_model()
    t1 = time.perf_counter()

    V = np.zeros(env.n_states)  # step 1 of the algorithm: start with v(s) = 0
    deltas = []
    for _ in range(max_iterations):
        # q(s, a) = r(s, a) + gamma * v(s'), with v(s') = 0 after a terminal transition
        Q = reward + gamma * np.where(terminated, 0.0, V[next_state])
        V_new = Q.max(axis=1)                    # v(s) = max_a q(s, a)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        deltas.append(delta)
        if delta < theta:                        # convergence: the values stop changing
            break
    Q = reward + gamma * np.where(terminated, 0.0, V[next_state])
    policy = Q.argmax(axis=1)                    # pi*(s) = argmax_a q*(s, a)
    t2 = time.perf_counter()

    return {"V": V, "Q": Q, "policy": policy, "deltas": deltas,
            "model_time": t1 - t0, "solve_time": t2 - t1}
