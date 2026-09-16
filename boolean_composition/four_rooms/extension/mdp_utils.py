"""Exact tabular MDP construction and dynamic-programming utilities.

`GridWorld.step` already supports stochastic transitions via `slip_prob`
(see `GridWorld.pertube_action`): with probability `slip_prob` the intended
action is replaced by a uniformly random *other* direction. That method only
exposes a *sampler* though, which is fine for Q-learning but noisy for
evaluation. Here we rebuild the same dynamics in closed form -- by enumerating
the slip distribution and the deterministic core of `GridWorld.step` -- so we
can run classical value iteration and exact policy evaluation instead of
Monte Carlo rollouts.

This is used by `exp_stochastic_sweep.py` to get an exact ground-truth V* and
exact composed-policy values V^pi for the stochastic Rooms sweep.
"""
from collections import defaultdict

import numpy as np

from four_rooms.GridWorld import UP, RIGHT, DOWN, LEFT, STAY
from four_rooms.library import EQ_P, epsilon_greedy_generalised_policy_improvement

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
ABSORBED_STATE = str([None, None])


def _slip_distribution(action, slip_prob, n_actions):
    """The {primitive_action: prob} distribution `pertube_action` samples from."""
    if action == STAY:
        return {STAY: 1.0}
    intended_prob = 1 - slip_prob
    slip_share = slip_prob / (n_actions - 2)  # spread over the 3 other directions
    return {a: (intended_prob if a == action else slip_share) for a in DIRECTIONS}


def _deterministic_transition(env, state, action):
    """Replicates the deterministic core of `GridWorld.step` for an
    already-resolved primitive `action` (no slip, no mutation of `env`).

    Returns (next_state_str, reward, done).
    """
    if state in env.T_states:
        reward = env._get_reward(state, action)
        return ABSORBED_STATE, reward, True

    if action == STAY and [state[1], state[1]] in env.T_states and not state[0]:
        new_state = [state[1], state[1]]
    else:
        x, y = state[1]
        if action == UP:
            x -= 1
        elif action == DOWN:
            x += 1
        elif action == RIGHT:
            y += 1
        elif action == LEFT:
            y -= 1
        new_state = [None, (x, y)]

    reward = env._get_reward(state, action)

    if env._get_grid_value(new_state) == 1:  # bumped into a wall, stay put
        new_state = [None, state[1]]

    return str(new_state), reward, False


def build_states(env):
    """Enumerates every reachable state string for this env config."""
    states = [str([None, pos]) for pos in env.possiblePositions]
    states += [str(t) for t in env.T_states]
    states.append(ABSORBED_STATE)
    return states


def build_transition_model(env):
    """Builds P[state][action] -> list of (next_state, prob, expected_reward),
    matching the exact stochastic dynamics `env.step` samples from (slip
    included). This is the closed-form counterpart of `GridWorld.step`.
    """
    n_actions = env.action_space.n
    P = defaultdict(lambda: defaultdict(list))

    for pos in env.possiblePositions:
        state = [None, pos]
        state_str = str(state)
        for action in range(n_actions):
            outcomes = defaultdict(lambda: [0.0, 0.0])  # next_state -> [prob, prob * reward]
            for primitive_action, prob in _slip_distribution(action, env.slip_prob, n_actions).items():
                next_state, reward, _ = _deterministic_transition(env, state, primitive_action)
                outcomes[next_state][0] += prob
                outcomes[next_state][1] += prob * reward
            for next_state, (prob, weighted_reward) in outcomes.items():
                if prob <= 0.0:  # e.g. slip_prob == 0: the 3 non-intended directions have prob 0
                    continue
                P[state_str][action].append((next_state, prob, weighted_reward / prob))

    # Achieved-goal marker states are absorbing: any action reveals the reward
    # and moves to the virtual terminal state (mirrors the "state in T_states"
    # branch at the top of `GridWorld.step`). No slip applies here since the
    # episode is already over.
    for t_state in env.T_states:
        state_str = str(t_state)
        for action in range(n_actions):
            _, reward, _ = _deterministic_transition(env, t_state, action)
            P[state_str][action].append((ABSORBED_STATE, 1.0, reward))

    for action in range(n_actions):
        P[ABSORBED_STATE][action].append((ABSORBED_STATE, 1.0, 0.0))

    return P


def _q_values(state, P, V, n_actions, gamma):
    return np.array([
        sum(prob * (reward + gamma * V[next_state]) for next_state, prob, reward in P[state][action])
        for action in range(n_actions)
    ])


def value_iteration(states, P, n_actions, gamma=1.0, tol=1e-6, max_iters=20_000, horizon=None):
    """Synchronous value iteration. Returns (V, Q, policy) dicts keyed by state string.

    With `horizon` set, runs exactly that many backward-induction sweeps from
    V = 0 and never takes the convergence break, so V(s) is the exact best
    expected return obtainable within `horizon` steps. That is the mode the
    stochastic sweep uses: an evaluation episode is capped (`utils.evaluate`
    stops at 100 steps), so the horizon-bounded optimum is the quantity the
    learned policies are actually competing against, and it stays finite when
    gamma is 1.
    """
    V = defaultdict(float)
    for _ in range(horizon if horizon is not None else max_iters):
        V_new = defaultdict(float)
        delta = 0.0
        for state in states:
            if state == ABSORBED_STATE:
                continue
            V_new[state] = _q_values(state, P, V, n_actions, gamma).max()
            delta = max(delta, abs(V_new[state] - V[state]))
        V = V_new
        if horizon is None and delta < tol:
            break

    Q = defaultdict(lambda: np.zeros(n_actions))
    policy = defaultdict(int)
    for state in states:
        if state == ABSORBED_STATE:
            continue
        Q[state] = _q_values(state, P, V, n_actions, gamma)
        policy[state] = int(np.argmax(Q[state]))

    return V, Q, policy


def policy_evaluation(states, P, policy, n_actions, gamma=1.0, tol=1e-6, max_iters=20_000,
                      horizon=None):
    """Exact evaluation of a fixed policy (state string -> action int), e.g.
    the output of `library.EQ_P`. States missing from `policy` default to
    action 0, matching `EQ_P`'s `defaultdict(lambda: 0)` behaviour.

    With `horizon` set, runs exactly that many sweeps and never breaks early,
    returning the exact expected `horizon`-step return -- the zero-variance
    counterpart of averaging capped rollouts.

    Pass a horizon whenever gamma is 1. An undiscounted improper policy (one
    that never reaches a goal from some state -- common once slip makes the
    composed policies cycle) has value -inf, so the `delta < tol` break never
    fires and the value returned is just -step_reward * max_iters. That is a
    silent function of the iteration cap rather than a property of the policy.
    """
    V = defaultdict(float)
    for _ in range(horizon if horizon is not None else max_iters):
        V_new = defaultdict(float)
        delta = 0.0
        for state in states:
            if state == ABSORBED_STATE:
                continue
            action = policy[state]
            V_new[state] = sum(
                prob * (reward + gamma * V[next_state]) for next_state, prob, reward in P[state][action]
            )
            delta = max(delta, abs(V_new[state] - V[state]))
        V = V_new
        if horizon is None and delta < tol:
            break
    return V


def expected_return(V, states):
    """Mean value over the start-state distribution.

    `GridWorld.reset` samples the start position uniformly over
    `possiblePositions`, so the mean over states *is* the expected return
    under the actual initial distribution.
    """
    return float(np.mean([V[s] for s in states if s != ABSORBED_STATE]))


def value_gap(V_star, V_pi, states, tol=1e-9):
    """Mean over start states of V*(s) - V^pi(s): expected return given up.

    Units are the env's own -- a step costs 0.1, so a gap of 0.4 reads as
    "about four wasted steps' worth of return". Nothing is normalized, so
    nothing can blow up or flip sign, and at a fixed slip probability the two
    composition methods are measured against the same V* in the same units.

    No absolute value, deliberately: V* dominates V^pi pointwise by
    construction, so `abs` would be a no-op here and would bury a real bug if
    the difference ever came out negative (policy evaluated against a
    different horizon or transition model than V* was solved for).
    """
    gaps = np.array([V_star[s] - V_pi[s] for s in states if s != ABSORBED_STATE])
    if gaps.min() < -tol:
        raise ValueError(
            f"V^pi exceeds V* by {-gaps.min():g}: V* is not optimal for the model "
            "this policy was evaluated on (mismatched horizon or transition model)."
        )
    return float(gaps.mean())


def rollout_returns(env, policy, n_episodes, horizon, seed=None):
    """Actually run `policy` in `env` and return the list of episode returns.

    The Monte Carlo counterpart of `policy_evaluation(..., horizon=horizon)`:
    same policy, same env, same cap, but sampled rather than solved. Reported
    alongside the DP numbers so the closed-form dynamics in this module can be
    checked against the environment they claim to model -- the two means agree
    up to Monte Carlo error if `build_transition_model` is faithful.

    Start states come from `env.reset()`, i.e. uniform over free cells, which
    is the distribution `expected_return` averages over.

    Args:
        env: the GridWorld to roll out in (its `slip_prob` supplies the noise).
        policy: state string -> action int, e.g. the output of `library.EQ_P`.
        n_episodes: how many episodes to run.
        horizon: per-episode step cap, matching the DP horizon.
        seed: optional seed for this policy's episodes.
    """
    # GridWorld.reset and GridWorld.step both draw from the global numpy
    # stream, so seeding the rollouts means seeding that -- save and restore
    # it, or evaluation would silently reshuffle the training stream.
    state_backup = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    try:
        returns = []
        for _ in range(n_episodes):
            state = env.reset()
            total = 0.0
            for _ in range(horizon):
                state, reward, done, _ = env.step(int(policy[state]))
                total += reward
                if done:
                    break
            returns.append(total)
    finally:
        if state_backup is not None:
            np.random.set_state(state_backup)
    return returns


def normalized_suboptimality(V_star, V_pi, states, eps=1e-6):
    """Mean over non-absorbed states of (V*(s) - V^pi(s)) / max(|V*(s)|, eps).

    Kept as the sweep's original headline number so its figure stays
    comparable, but read it with care: the per-state denominator passes
    through zero on tasks with few goals (V*(s) ~ 2 - 0.1 * steps-to-goal is 0
    at ~20 steps), so a handful of states can dominate the mean. `value_gap`
    is the same comparison without that failure mode.
    """
    diffs = [
        (V_star[s] - V_pi[s]) / max(abs(V_star[s]), eps)
        for s in states
        if s != ABSORBED_STATE
    ]
    return float(np.mean(diffs))


def train_goal_oriented_until_convergence(
    env,
    gamma=1.0,
    epsilon=1,
    alpha=0.1,
    max_steps=50_000,
    check_every=1_000,
    tol=None,
    horizon=None,
):
    """Goal-Oriented Q-learning (same update rule as
    `library.Goal_Oriented_Q_learning`), but trained *to convergence* instead
    of for a fixed number of episodes.

    `alpha` defaults to 0.1 here rather than to `library`'s 1. Under
    deterministic transitions alpha=1 is not merely a large step size, it is
    the *exact* Bellman backup -- the successor is a function of (s, a), so
    the update is what value iteration would compute -- which is why it is the
    right default everywhere else in this repo. Under slip the successor is a
    sample from the slip distribution, and alpha=1 overwrites Q with whichever
    successor was drawn last instead of averaging over them, so Q never
    approaches Q*. 0.1 is the standard tabular value; it leaves Q oscillating
    in a small noise ball rather than converging exactly, so `tol` should be
    read as "close enough", not as exact convergence.

    Every `check_every` environment steps, the current greedy policy
    (`EQ_P(Q)`) is evaluated exactly against the exact optimum, both via the
    closed-form dynamics in this module, and the result is recorded. This
    duplicates `Goal_Oriented_Q_learning`'s loop body (rather than calling it
    repeatedly) because that function reinitializes its Q-table on every call,
    which would throw away progress between checks.

    `tol` defaults to None, i.e. that check *reports* but does not *stop*.
    Stopping on it is not sound: `EQ_P(Q)` is greedy w.r.t. max over goals, so
    the check measures the max-over-goals policy, whereas both composition
    methods consume the individual per-goal values `Q[s][g]`. Those converge
    much later. With `tol=1e-3` the universal and empty EQs on the 4-room map
    stop at an own-task suboptimality of 0.000 while still composing to a
    value gap of 2.14 at slip 0 -- where the collapse theorem requires exactly
    0 -- and raising `max_steps` changes nothing, because the stopping rule
    fires at the same step. Training the full budget instead reaches a gap of
    0.0000 (alpha=0.1, 200k steps; alpha=1 needs 50k).

    Pass a float to restore early stopping, remembering it is a lower bound on
    how trained the EQ is, not a statement about its per-goal values.

    Returns:
        Q: the learned EQ (state -> goal -> action-values).
        stats: {"R": per-episode returns, "T": total steps taken}.
        convergence: {"converged": bool, "steps": steps taken (to convergence,
            or max_steps if it never converged), "final_subopt": the exact
            normalized suboptimality at the last check}.
    """
    states = build_states(env)
    P = build_transition_model(env)
    V_star, _, _ = value_iteration(states, P, env.action_space.n, gamma=gamma, horizon=horizon)

    N = min(env.rmin, (env.rmin - env.rmax) * env.diameter)
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    behaviour_policy = epsilon_greedy_generalised_policy_improvement(env, Q, epsilon=epsilon)

    sMem = {}  # Goals memory
    stats = {"R": [0], "T": 0}
    T = 0
    converged_at = None
    final_subopt = None
    state = env.reset()

    while T < max_steps:
        probs = behaviour_policy(state, epsilon=epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        state_, reward, done, _ = env.step(action)

        stats["R"][-1] += reward

        if done:
            sMem[state] = 0

        for goal in sMem.keys():
            if state != goal and done:
                reward_ = N
            else:
                reward_ = reward

            G = 0 if done else np.max(Q[state_][goal])
            TD_target = reward_ + gamma * G
            TD_error = TD_target - Q[state][goal][action]
            Q[state][goal][action] = Q[state][goal][action] + alpha * TD_error

        state = state_
        T += 1

        if done:
            state = env.reset()
            stats["R"].append(0)

        if T % check_every == 0:
            policy = EQ_P(Q)
            V_pi = policy_evaluation(
                states, P, policy, env.action_space.n, gamma=gamma, horizon=horizon
            )
            final_subopt = normalized_suboptimality(V_star, V_pi, states)
            if tol is not None and final_subopt < tol:
                converged_at = T
                break

    stats["T"] = T

    if final_subopt is None:  # max_steps < check_every: never hit a check boundary
        policy = EQ_P(Q)
        V_pi = policy_evaluation(
            states, P, policy, env.action_space.n, gamma=gamma, horizon=horizon
        )
        final_subopt = normalized_suboptimality(V_star, V_pi, states)

    convergence = {
        "converged": converged_at is not None,
        "steps": converged_at if converged_at is not None else T,
        "final_subopt": final_subopt,
    }
    return Q, stats, convergence


def own_task_suboptimality(env, learned_EQ, gamma=1.0, horizon=None):
    """Exact normalized suboptimality of `learned_EQ`'s greedy policy,
    evaluated on the very env/task it was trained on -- no composition
    involved.

    This isolates how much of a composed policy's suboptimality is already
    present in its *constituent* EQs (i.e. imperfect Goal-Oriented Q-learning
    under `env.slip_prob`) as opposed to being introduced by the composition
    step (AND/OR/NOT or goal-set on/off) itself.
    """
    states = build_states(env)
    P = build_transition_model(env)
    V_star, _, _ = value_iteration(states, P, env.action_space.n, gamma=gamma, horizon=horizon)
    policy = EQ_P(learned_EQ)
    V_pi = policy_evaluation(
        states, P, policy, env.action_space.n, gamma=gamma, horizon=horizon
    )
    return normalized_suboptimality(V_star, V_pi, states)
