"""Tests for the exact Boxman MDP the stochastic sweep evaluates against.

`exp_stochastic_sweep.py` reports how far a composed policy falls short of
optimal, so the ground truth it measures against carries the whole result. The
tests below pin that ground truth to things we can derive independently: the
slip distribution matches `GridWorld.pertube_action`, the rebuilt dynamics are
a valid transition model, and at zero slip the optimal value of every cell is
exactly the shortest-path return.
"""
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mdp_utils
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

START_POSITIONS = {
    "crate_beige": (3, 4),
    "player": (6, 3),
    "circle_purple": (7, 7),
    "circle_beige": (1, 7),
    "crate_blue": (1, 1),
    "crate_purple": (8, 1),
    "circle_blue": (1, 8),
}
HORIZON = 20


@pytest.fixture(scope="module")
def env():
    env = WarpFrame(CollectEnv(start_positions=START_POSITIONS))
    env.reset()
    return env


def shortest_path_lengths(env, targets):
    """BFS over free cells: {cell: moves needed to stand on the nearest target}."""
    distances = {target: 0 for target in targets}
    queue = deque(targets)
    board = env.unwrapped.board
    while queue:
        cell = queue.popleft()
        for move in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            neighbour = (cell[0] + move[0], cell[1] + move[1])
            if board[neighbour] != "#" and neighbour not in distances:
                distances[neighbour] = distances[cell] + 1
                queue.append(neighbour)
    return distances


def test_slip_distribution_matches_gridworld():
    distribution = mdp_utils.slip_distribution(action=0, slip_prob=0.3, n_actions=5)

    assert distribution[0] == pytest.approx(0.7)
    for other in [1, 2, 3]:
        assert distribution[other] == pytest.approx(0.1)
    assert mdp_utils.STAY not in distribution
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_pickup_action_never_slips():
    assert mdp_utils.slip_distribution(mdp_utils.STAY, slip_prob=0.9) == {mdp_utils.STAY: 1.0}


def test_transition_probabilities_are_normalized(env):
    positions = mdp_utils.goal_positions(env)
    P = mdp_utils.build_transition_model(env, [positions[1]], slip_prob=0.4)

    for state, actions in P.items():
        for action, transitions in actions.items():
            total = sum(prob for _, prob, _ in transitions)
            assert total == pytest.approx(1.0), (state, action)


def test_goal_positions_are_in_goals_h5_order(env):
    # Raises if the board layout and the goal-image order drift apart.
    assert mdp_utils.goal_positions(env, expected=[(1, 8), (8, 1), (1, 1), (6, 3), (1, 7), (7, 7)])


def test_optimal_value_is_the_shortest_path_return_without_slip(env):
    """Without slip, V*(s) = rmax + rmin * (moves + 1): walk to the nearest
    wanted object, pick it up (one step), then collect the reward."""
    collect_env = env.unwrapped
    target = mdp_utils.goal_positions(env)[1]  # the blue square, the "B.S" task
    P = mdp_utils.build_transition_model(env, [target], slip_prob=0.0)
    V_star = mdp_utils.value_iteration(P, HORIZON)
    distances = shortest_path_lengths(env, [target])

    for pos, moves in distances.items():
        expected = collect_env.rmax + collect_env.rmin * (moves + 1)
        assert V_star[("pos", pos)] == pytest.approx(expected), pos


def test_worst_case_value_is_never_reaching_an_object(env):
    """The worst a policy can do is burn the whole budget at rmin per step:
    every alternative ends the episode early, which costs less."""
    collect_env = env.unwrapped
    positions = mdp_utils.goal_positions(env)
    P = mdp_utils.build_transition_model(env, positions, slip_prob=0.0)
    V_worst = mdp_utils.value_iteration(P, HORIZON, optimize=min)

    for state in mdp_utils.start_states(env):
        assert V_worst[state] == pytest.approx(collect_env.rmin * HORIZON)


def test_value_gap_is_zero_for_the_optimal_policy_and_positive_otherwise(env):
    positions = mdp_utils.goal_positions(env)
    P = mdp_utils.build_transition_model(env, [positions[0]], slip_prob=0.5)
    states = mdp_utils.start_states(env)
    V_star = mdp_utils.value_iteration(P, HORIZON)
    V_worst = mdp_utils.value_iteration(P, HORIZON, optimize=min)

    assert mdp_utils.value_gap(V_star, V_star, states) == pytest.approx(0.0)
    expected = np.mean([V_star[s] - V_worst[s] for s in states])
    assert mdp_utils.value_gap(V_star, V_worst, states) == pytest.approx(expected)


def test_value_gap_rejects_a_policy_that_beats_the_optimum(env):
    """The gap is signed on purpose: a negative entry means V* is wrong for the
    model the policy was evaluated on, and must surface rather than be absorbed
    by an abs()."""
    positions = mdp_utils.goal_positions(env)
    P = mdp_utils.build_transition_model(env, [positions[0]], slip_prob=0.0)
    states = mdp_utils.start_states(env)
    V_star = mdp_utils.value_iteration(P, HORIZON)
    impossible = {state: value + 1.0 for state, value in V_star.items()}

    with pytest.raises(ValueError, match="not optimal"):
        mdp_utils.value_gap(V_star, impossible, states)


def test_regret_is_one_for_the_worst_policy_and_zero_for_the_optimal_one(env):
    positions = mdp_utils.goal_positions(env)
    P = mdp_utils.build_transition_model(env, [positions[0]], slip_prob=0.5)
    states = mdp_utils.start_states(env)
    V_star = mdp_utils.value_iteration(P, HORIZON)
    V_worst = mdp_utils.value_iteration(P, HORIZON, optimize=min)

    assert mdp_utils.normalized_regret(V_star, V_worst, V_star, states) == pytest.approx(0.0)
    assert mdp_utils.normalized_regret(V_star, V_worst, V_worst, states) == pytest.approx(1.0)


def test_slip_lowers_the_optimal_value(env):
    positions = mdp_utils.goal_positions(env)
    states = mdp_utils.start_states(env)

    returns = []
    for slip_prob in [0.0, 0.3, 0.6, 0.9]:
        P = mdp_utils.build_transition_model(env, [positions[1]], slip_prob)
        returns.append(mdp_utils.expected_return(mdp_utils.value_iteration(P, HORIZON), states))

    assert returns == sorted(returns, reverse=True)


def test_render_observations_covers_every_free_cell(env):
    observations = mdp_utils.render_observations(env)

    assert set(observations) == set(mdp_utils.free_positions(env))
    assert all(obs.shape == (84, 84, 3) for obs in observations.values())
    # Distinct player positions must be distinguishable to the network.
    rendered = {obs.tobytes() for obs in observations.values()}
    assert len(rendered) == len(observations)
