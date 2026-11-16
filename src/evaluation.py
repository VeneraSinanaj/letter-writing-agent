"""
Evaluating a policy: play it from a starting point and measure how it draws.
Used by every method (dynamic programming, Q-learning, SARSA, Monte Carlo, DQN).
"""
import numpy as np


def play_policy(env, policy, start, gamma=0.99):
    """Play a deterministic policy (policy[s] = action) from `start`, pen up.

    Returns a dict:
        history          env.history, for drawing the episode
        total_reward     sum of the rewards
        discounted_return  sum of gamma^t * r_t (equals v(start) for the optimal policy)
        steps            number of actions, STOP included
        strokes          number of times the pen was put down
        drawn            segments of the letter drawn, out of env.k
        off_letter       segments drawn outside the letter (counted each time)
        completed        True if the agent pressed STOP with the whole letter drawn
    """
    state = env.reset(start=start)
    total, discounted, t = 0.0, 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        state, reward, terminated, truncated, _ = env.step(int(policy[state]))
        total += reward
        discounted += gamma**t * reward
        t += 1

    strokes, off_letter, pen_was_down = 0, 0, False
    for info in env.history:
        if info["event"] != "move":
            continue  # STOP and off-grid moves do not change the pen state
        if info["pen_down"]:
            strokes += not pen_was_down
            off_letter += info["segment"] == "off_letter"
        pen_was_down = info["pen_down"]

    drawn = env.drawn_segments()
    return {
        "history": list(env.history),
        "total_reward": total,
        "discounted_return": discounted,
        "steps": env.steps,
        "strokes": strokes,
        "drawn": drawn,
        "off_letter": off_letter,
        "completed": bool(terminated and drawn == env.k),
    }


def greedy_policy(Q):
    """Greedy policy from a Q-table: in each state, the action with the highest value."""
    return np.asarray(Q).argmax(axis=1)
