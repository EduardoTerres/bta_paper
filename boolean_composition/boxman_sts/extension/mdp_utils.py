"""Slip dynamics and exact finite-horizon MDP utilities for the Boxman env.

The Rooms sweep (`four_rooms/extension/exp_stochastic_sweep.py`) leans on
`GridWorld.slip_prob` plus the closed-form dynamics in
`four_rooms/extension/mdp_utils.py`. `CollectEnv` has no slip parameter, so
this module supplies the two pieces needed to run the same sweep here:

  1. `SlipAction`: a wrapper that perturbs the agent's action exactly the way
     `GridWorld.pertube_action` does -- with probability `slip_prob` the
     intended move is replaced by a uniformly random *other* direction, while
     the "stay" action (the one that picks an object up) is never perturbed.
     Everything downstream of the wrapper (DQN training, rollouts) is the
     unmodified Boxman env.

  2. A tabular rebuild of the resulting dynamics, so ground truth and the
     learned policies' values can be computed by dynamic programming instead
     of Monte Carlo rollouts.

The tabular rebuild is exact and cheap because a Boxman episode ends at the
*first* pick-up: before that, every object is still on the board, so the state
is fully described by the player's cell (58 of them). The two post-pick-up
states ("picked up something the task wanted" / "picked up something it did
not") plus a terminal state complete the model.

Dynamic programming here is *finite-horizon* with horizon `max_trajectory`,
matching the `MaxLength` wrapper every Boxman experiment evaluates under.
That is both the honest model of the evaluation protocol and the reason no
discounting is needed: the undiscounted infinite-horizon value of a policy
that gets stuck (e.g. one that keeps choosing "stay" on an empty cell, which
slip never perturbs) would diverge, and at high slip probabilities such
policies do occur.

Note what is and is not tabular: the *environment* is rebuilt in closed form,
but the policies being evaluated are still the greedy policies of the learned
DQNs, read off by rendering each state and running the network on it
(`render_observations` + `greedy_policy`). This is a lower-variance version of
"roll the composed policy out and measure the return", not a different
quantity -- and it keeps the function-approximation character of the
experiment intact.
"""
from collections import defaultdict

import gym
import numpy as np
import torch

from dqn import FloatTensor

# Mirrors CollectEnv._ACTIONS.
ACTIONS = {
    0: (-1, 0),  # North
    1: (0, 1),  # East
    2: (1, 0),  # South
    3: (0, -1),  # West
    4: (0, 0),  # Stay / pick up
}
DIRECTIONS = [0, 1, 2, 3]
STAY = 4

# Model states. Pre-pick-up states are ("pos", (row, col)).
PICKED_WANTED = ("picked", True)
PICKED_UNWANTED = ("picked", False)
TERMINAL = ("terminal",)

# Order of the goal images in `boxman_sts/goals.h5`, as (shape, colour) pairs:
# BC, BS, bS, PS, bC, PC. `goal_positions` checks this against the env.
GOAL_ORDER = [
    ("circle", "blue"),
    ("square", "blue"),
    ("square", "beige"),
    ("square", "purple"),
    ("circle", "beige"),
    ("circle", "purple"),
]


def slip_distribution(action, slip_prob, n_actions=5):
    """The {primitive_action: prob} distribution an action is resolved through.

    Identical in form to `four_rooms/extension/mdp_utils._slip_distribution`:
    the intended direction keeps `1 - slip_prob`, and `slip_prob` is spread
    evenly over the other `n_actions - 2` directions (all directions except
    the intended one and "stay"). "Stay" itself never slips -- it is the
    pick-up action, and perturbing it would make picking an object up
    impossible rather than merely noisy.
    """
    if action == STAY:
        return {STAY: 1.0}
    slip_share = slip_prob / (n_actions - 2)
    return {a: ((1 - slip_prob) if a == action else slip_share) for a in DIRECTIONS}


class SlipAction(gym.Wrapper):
    """Replaces the intended action by a random other direction w.p. `slip_prob`."""

    def __init__(self, env, slip_prob=0.0):
        gym.Wrapper.__init__(self, env)
        self.slip_prob = slip_prob

    def perturb_action(self, action):
        if self.slip_prob <= 0.0 or action == STAY:
            return action
        distribution = slip_distribution(action, self.slip_prob, self.action_space.n)
        actions = list(distribution)
        probs = [distribution[a] for a in actions]
        return int(np.random.choice(actions, p=probs))

    def step(self, action):
        return self.env.step(self.perturb_action(int(action)))


# ------------------------------------------------------------
# Reading the board layout off a (possibly wrapped) CollectEnv
# ------------------------------------------------------------
def free_positions(env):
    """Every non-wall cell, i.e. the support of the random start position."""
    return sorted(tuple(int(c) for c in pos) for pos in env.unwrapped.free_spaces)


def object_positions(env):
    """{(shape, colour): (row, col)} for the objects on the board.

    Requires a reset env: `CollectEnv.reset` is what places the collectibles.
    """
    collect_env = env.unwrapped
    return {
        (sprite.shape, sprite.colour): tuple(int(c) for c in sprite.position)
        for sprite in collect_env.collectibles
    }


def goal_positions(env, expected=None):
    """Object positions in `goals.h5` order, so goal indices line up with cells.

    Args:
        env: a reset (possibly wrapped) CollectEnv.
        expected: optional list of positions to assert against, to catch the
            goal-image order and the board layout drifting apart.
    """
    by_kind = object_positions(env)
    positions = [by_kind[kind] for kind in GOAL_ORDER]
    if expected is not None and positions != [tuple(p) for p in expected]:
        raise ValueError(
            f"Goal positions {positions} do not match the expected order {expected}; "
            "goals.h5 and the board layout have drifted apart."
        )
    return positions


# ------------------------------------------------------------
# Exact dynamics
# ------------------------------------------------------------
def build_transition_model(env, on_positions, slip_prob):
    """P[state][action] -> [(next_state, prob, expected_reward)] for one task.

    Replicates the deterministic core of `CollectEnv.step` (walls block moves;
    "stay" on top of an object picks it up and ends the episode one step
    later, with `rmax` if the task wanted that object and `rmin` otherwise)
    and folds in the slip distribution.

    Args:
        env: a reset (possibly wrapped) CollectEnv.
        on_positions: cells holding objects this task's goal condition
            accepts. Everything else on the board still *ends* the episode
            when picked up, just without the reward.
        slip_prob: probability the intended direction is replaced.
    """
    collect_env = env.unwrapped
    board = collect_env.board
    rmin, rmax = collect_env.rmin, collect_env.rmax
    n_actions = collect_env.action_space.n
    objects = set(object_positions(env).values())
    on_positions = set(on_positions)

    P = {}
    for pos in free_positions(env):
        state = ("pos", pos)
        P[state] = {}
        for action in range(n_actions):
            outcomes = defaultdict(lambda: [0.0, 0.0])  # next_state -> [prob, prob * reward]
            for primitive, prob in slip_distribution(action, slip_prob, n_actions).items():
                if primitive == STAY and pos in objects:
                    next_state = PICKED_WANTED if pos in on_positions else PICKED_UNWANTED
                else:
                    move = ACTIONS[primitive]
                    candidate = (pos[0] + move[0], pos[1] + move[1])
                    next_state = ("pos", pos if board[candidate] == "#" else candidate)
                outcomes[next_state][0] += prob
                outcomes[next_state][1] += prob * rmin  # every step costs rmin
            P[state][action] = [
                (next_state, prob, weighted_reward / prob)
                for next_state, (prob, weighted_reward) in outcomes.items()
                if prob > 0.0  # the non-intended directions vanish at slip_prob == 0
            ]

    # The step after a pick-up is the one that reveals the reward and sets
    # done (see the `if self.goal:` branch at the top of `CollectEnv.step`).
    # The action taken there is irrelevant, and slip does not apply.
    for state, reward in [(PICKED_WANTED, rmax), (PICKED_UNWANTED, rmin)]:
        P[state] = {action: [(TERMINAL, 1.0, reward)] for action in range(n_actions)}
    P[TERMINAL] = {action: [(TERMINAL, 1.0, 0.0)] for action in range(n_actions)}
    return P


def start_states(env):
    """The states an episode can begin in: every free cell, uniformly."""
    return [("pos", pos) for pos in free_positions(env)]


def value_iteration(P, horizon, optimize=max):
    """Finite-horizon backward induction. Returns V_horizon as a state dict.

    `horizon` is the `MaxLength` budget, so V(s) is exactly the best expected
    return obtainable from `s` within an episode.

    Passing `optimize=min` instead returns the *worst* achievable value, which
    `normalized_regret` uses as the bottom of the achievable range.
    """
    V = {state: 0.0 for state in P}
    for _ in range(horizon):
        V = {
            state: optimize(
                sum(prob * (reward + V[next_state]) for next_state, prob, reward in transitions)
                for transitions in actions.values()
            )
            for state, actions in P.items()
        }
    return V


def optimal_policy(P, horizon):
    """A stationary policy greedy w.r.t. the horizon-step optimal values.

    Used only to give the measured-return figure a reference line: `V_star`
    itself cannot be rolled out, since finite-horizon optimality is attained
    by a *non-stationary* policy (the right action depends on how many steps
    remain). This takes the first action of that policy and holds it fixed, so
    its expected return is at most `V_star` and in practice very close to it
    when the horizon comfortably exceeds the distance to an object.

    The reported `returns`/`optimal` line stays `expected_return(V_star, ...)`
    -- the true optimum -- so the DP figures are unaffected by this
    approximation.
    """
    V = {state: 0.0 for state in P}
    for _ in range(horizon):
        V = {
            state: max(
                sum(prob * (reward + V[next_state]) for next_state, prob, reward in transitions)
                for transitions in actions.values()
            )
            for state, actions in P.items()
        }
    policy = {}
    for state, actions in P.items():
        values = {
            action: sum(
                prob * (reward + V[next_state]) for next_state, prob, reward in transitions
            )
            for action, transitions in actions.items()
        }
        policy[state] = max(values, key=values.get)
    return policy


def policy_evaluation(P, policy, horizon):
    """Finite-horizon evaluation of a deterministic `policy` (state -> action).

    States missing from `policy` fall back to action 0; that only ever applies
    to the post-pick-up states, where every action behaves identically.
    """
    V = {state: 0.0 for state in P}
    for _ in range(horizon):
        V = {
            state: sum(
                prob * (reward + V[next_state])
                for next_state, prob, reward in actions[policy.get(state, 0)]
            )
            for state, actions in P.items()
        }
    return V


def expected_return(V, states):
    """Mean value over the (uniform) start-state distribution."""
    return float(np.mean([V[state] for state in states]))


def value_gap(V_star, V_pi, states, tol=1e-9):
    """Mean over start states of V*(s) - V^pi(s): expected return given up.

    The primary metric. Its units are the env's own -- one unit is one unit of
    episode return, and since a step costs `rmin` = -0.1, a gap of 0.4 reads as
    "about four wasted steps' worth of return". Nothing is normalized, so
    nothing can blow up or flip sign; comparing the two composition methods at
    a fixed slip probability compares like with like directly.

    No absolute value, deliberately. V* is the maximum over *all* policies,
    including the non-stationary ones finite-horizon optimality admits, so
    V*(s) >= V^pi(s) holds pointwise by construction and `abs` would be a
    no-op. That makes it worth far more as a check than as part of the
    definition: if the difference ever comes out negative, something upstream
    is wrong -- the policy was evaluated against a different horizon or
    transition model than V* was solved for, or `V_star` is not actually
    optimal -- and silently taking `abs` would bury exactly that bug.
    """
    gaps = np.array([V_star[state] - V_pi[state] for state in states])
    worst = gaps.min()
    if worst < -tol:
        state = states[int(gaps.argmin())]
        raise ValueError(
            f"V^pi exceeds V* by {-worst:g} at {state}: V* is not optimal for the "
            "model this policy was evaluated on (mismatched horizon, transition "
            "model, or state set)."
        )
    return float(gaps.mean())


def rollout_returns(env, policy, n_episodes, horizon, start_positions=None, seed=None):
    """Roll `policy` out in the real Boxman env and return the episode returns.

    The Monte Carlo counterpart of `policy_evaluation(P, policy, horizon)`:
    same policy, same slip, same cap, but sampled rather than solved. Reported
    alongside the DP numbers so the closed-form dynamics in this module can be
    checked against the environment they claim to model -- the two means agree
    up to Monte Carlo error if `build_transition_model` is faithful.

    No network call happens inside the loop, and none is needed: before the
    first pick-up the model state is exactly ("pos", player cell), so the
    per-cell decisions `greedy_policy` already read off the DQN *are* this
    policy. After a pick-up `CollectEnv.step` returns from its `if self.goal:`
    branch before looking at the action, so the lookup falling back to 0 there
    is exact rather than approximate.

    Start cells are drawn uniformly over the free cells, matching the
    distribution `expected_return` averages over. `CollectEnv` places the
    player at a fixed cell when `start_positions` was passed to it, so the
    player is teleported after reset -- the same move `render_observations`
    makes, and sound for the same reason: nothing has been collected yet.

    Args:
        env: a CollectEnv wrapped in `SlipAction` (the slip is what is being
            measured). No observation wrapper is needed.
        policy: model state -> action, e.g. the output of `greedy_policy`.
        n_episodes: how many episodes to run.
        horizon: per-episode step cap, matching the DP horizon.
        start_positions: cells to start from (default: every free cell).
        seed: optional seed. Both the start cells and `SlipAction` draw from
            the global numpy stream, so this seeds that stream and restores it
            afterwards rather than reshuffling training randomness.
    """
    collect_env = env.unwrapped
    cells = list(start_positions) if start_positions is not None else free_positions(env)

    state_backup = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(seed)
    try:
        returns = []
        for _ in range(n_episodes):
            env.reset()
            collect_env.player.reset(tuple(cells[np.random.randint(len(cells))]))
            total = 0.0
            for _ in range(horizon):
                position = tuple(int(c) for c in collect_env.player.position)
                action = policy.get(("pos", position), 0)
                _, reward, done, _ = env.step(int(action))
                total += reward
                if done:
                    break
            returns.append(total)
    finally:
        if state_backup is not None:
            np.random.set_state(state_backup)
    return returns


def normalized_suboptimality(V_star, V_pi, states, eps=1e-6):
    """Mean over start states of (V*(s) - V^pi(s)) / max(|V*(s)|, eps).

    Recorded for comparability with the Rooms sweep, not plotted. Unlike
    Rooms, V* here crosses zero at high slip probabilities -- a 20-step budget
    is often not enough to reach an object when 90% of moves are random, so
    the optimal return itself goes negative -- and the ratio becomes large and
    sign-flipped there. Use `value_gap`, or `normalized_regret` if a unit-free
    number is wanted.
    """
    return float(
        np.mean([
            (V_star[state] - V_pi[state]) / max(abs(V_star[state]), eps)
            for state in states
        ])
    )


def normalized_regret(V_star, V_worst, V_pi, states, eps=1e-6):
    """Mean over start states of (V*(s) - V^pi(s)) / (V*(s) - V_worst(s)).

    `value_gap` in units of the range actually achievable from that state at
    this slip probability, rather than of V* itself. It stays in [0, 1]
    (0 = optimal, 1 = worst possible) however far V* drifts towards zero.

    Worth having because the achievable range shrinks as slip rises -- the
    spread between the best and worst policy is about 3.9 return at p=0 but
    only about 1.2 at p=0.9 -- so the same raw gap means something different
    at the two ends of the sweep. Recorded alongside `value_gap` rather than
    instead of it, since the own-task baseline lines already control for
    "everything gets harder at high slip" more directly than any denominator
    can.
    """
    return float(
        np.mean([
            (V_star[state] - V_pi[state]) / max(V_star[state] - V_worst[state], eps)
            for state in states
        ])
    )


# ------------------------------------------------------------
# Reading a learned policy off a DQN
# ------------------------------------------------------------
def render_observations(env):
    """{position: observation} for every pre-pick-up state.

    `env` must be the observation-processing wrapper used at training time
    (i.e. a `WarpFrame`), so the frames here are pixel-identical to what the
    agent sees during a rollout. The board is rendered once and the player
    teleported across it, which is sound precisely because no object has been
    collected yet in any of these states.
    """
    collect_env = env.unwrapped
    collect_env.reset()
    observations = {}
    for pos in free_positions(env):
        collect_env.player.reset(pos)
        observations[pos] = env.observation(collect_env._draw_screen(collect_env._surface))
    return observations


def greedy_policy(dqn, observations, goal_tensor):
    """The policy a DQN induces: argmax_a max_g Q(s, g, a), as in `exp_returns`."""
    policy = {}
    with torch.inference_mode():
        for pos, obs in observations.items():
            obs_tensor = (
                torch.from_numpy(obs)
                .type(FloatTensor)
                .unsqueeze(0)
                .expand(goal_tensor.shape[0], -1, -1, -1)
            )
            values = dqn(torch.cat((obs_tensor, goal_tensor), dim=3))
            policy[("pos", pos)] = int(values.max(0)[0].argmax().item())
    return policy
