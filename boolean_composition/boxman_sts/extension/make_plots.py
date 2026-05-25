import argparse
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import matplotlib.image as mpimg
import numpy as np

np.object = object

import deepdish as dd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

from dqn import ComposedDQN_onoff, FloatTensor
from exp_convergence import (
    START_POSITIONS,
    TASKS,
    TRAIN_TASKS,
    boolean_composition,
    close_env,
    completed_run_indices,
    load_checkpoint,
)
from gym_repoman.envs import CollectEnv
from plot_utils import (
    by_task_figure_path,
    fair_figure_path,
    plot_convergence,
    plot_convergence_by_task,
    plot_fair_convergence,
    plot_value_map,
    plot_value_progression,
)
from wrappers import WarpFrame


def make_plots(args):
    returns_per_steps = dd.io.load(str(args.output))
    plot_convergence(returns_per_steps, args.figure)
    by_task_figure = by_task_figure_path(args.figure)
    fair_figure = fair_figure_path(args.figure)
    plot_convergence_by_task(returns_per_steps, by_task_figure)
    plot_fair_convergence(returns_per_steps, fair_figure)

    steps = sorted(returns_per_steps)
    run_indices = completed_run_indices(steps)
    if run_indices:
        map_image = mpimg.imread(args.map)
        goals = dd.io.load(str(args.goals))
        value_dir = args.value_figure_dir
        onoff_maps, onoff_policies, boolean_maps, boolean_policies = value_progressions(
            run_indices,
            steps,
            goals,
        )
        plot_value_progression(
            boolean_maps,
            boolean_policies,
            map_image,
            value_dir / "blue_value_progression_base_tasks.png",
            title="Base Tasks",
        )
        plot_value_progression(
            onoff_maps,
            onoff_policies,
            map_image,
            value_dir / "blue_value_progression_univ_empty.png",
            title="Univ./Empty",
        )
        last_step = steps[-1]
        plot_value_map(
            boolean_maps[last_step],
            boolean_policies[last_step],
            map_image,
            value_dir / "blue_value_last_base_tasks.png",
        )
        plot_value_map(
            onoff_maps[last_step],
            onoff_policies[last_step],
            map_image,
            value_dir / "blue_value_last_univ_empty.png",
        )
    else:
        print("Skipping value-function maps: no complete convergence runs found.")

    print(f"Saved plots under {args.figure.parent}")


def value_progressions(run_indices, steps, goals):
    free_spaces, board_shape = board_spaces()
    observations = {
        position: observation_at(position)
        for position in tqdm(free_spaces, desc="Map states", leave=False)
    }
    onoff_maps = {}
    onoff_policies = {}
    boolean_maps = {}
    boolean_policies = {}
    for step in tqdm(steps, desc="Value-function maps"):
        onoff_values = []
        onoff_action_values = []
        boolean_values = []
        boolean_action_values = []
        for run_idx in run_indices:
            dqns = {
                task_name: load_checkpoint(run_idx, step, task_name, condition)
                for task_name, condition in TRAIN_TASKS.items()
            }
            blue_goals = [goals[i] for i in TASKS["B"]["goal_indices"]]
            onoff_dqn = ComposedDQN_onoff(dqns["on"], dqns["off"], on_goals=blue_goals)
            boolean_dqn = boolean_composition(dqns["blue"], dqns["square"], "B")
            onoff_value_map, onoff_action_value_map = value_map(
                onoff_dqn,
                blue_goals,
                observations,
                board_shape,
            )
            boolean_value_map, boolean_action_value_map = value_map(
                boolean_dqn,
                blue_goals,
                observations,
                board_shape,
            )
            onoff_values.append(onoff_value_map)
            onoff_action_values.append(onoff_action_value_map)
            boolean_values.append(boolean_value_map)
            boolean_action_values.append(boolean_action_value_map)
            del dqns, onoff_dqn, boolean_dqn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            onoff_maps[step] = np.nanmean(onoff_values, axis=0)
            onoff_mean_actions = np.nanmean(onoff_action_values, axis=0)
            boolean_maps[step] = np.nanmean(boolean_values, axis=0)
            boolean_mean_actions = np.nanmean(boolean_action_values, axis=0)
        onoff_policies[step] = policy_from_action_values(onoff_mean_actions)
        boolean_policies[step] = policy_from_action_values(boolean_mean_actions)
    return onoff_maps, onoff_policies, boolean_maps, boolean_policies


def value_map(dqn, goals, observations, board_shape):
    goal_tensor = torch.from_numpy(np.asarray(goals)).type(FloatTensor)
    grid = np.full(board_shape, np.nan, dtype=float)
    action_grid = np.full((*board_shape, 5), np.nan, dtype=float)
    dqn.eval()
    with torch.inference_mode():
        positions = list(observations)
        obs_tensor = torch.from_numpy(np.asarray([observations[pos] for pos in positions])).type(FloatTensor)
        obs_tensor = obs_tensor[:, None].expand(-1, goal_tensor.shape[0], -1, -1, -1)
        goals_tensor = goal_tensor[None].expand(len(positions), -1, -1, -1, -1)
        obs_goal = torch.cat((obs_tensor, goals_tensor), dim=4)
        obs_goal = obs_goal.reshape(-1, *obs_goal.shape[2:])
        values = dqn(obs_goal).reshape(len(positions), goal_tensor.shape[0], -1)
        action_values = values.amax(dim=1).detach().cpu().numpy()
        state_values = action_values.max(axis=1)
        for position, state_value, action_value in zip(positions, state_values, action_values):
            grid[position] = float(state_value)
            action_grid[position] = action_value
    return grid, action_grid


def policy_from_action_values(action_values):
    policy = np.full(action_values.shape[:2], np.nan)
    valid = np.isfinite(action_values).any(axis=2)
    policy[valid] = np.nanargmax(action_values[valid], axis=1)
    return policy


def observation_at(player_position):
    start_positions = dict(START_POSITIONS)
    start_positions["player"] = player_position
    env = WarpFrame(
        CollectEnv(
            start_positions=start_positions,
            changePlayerPos=False,
            goal_condition=TASKS["B"]["condition"],
        )
    )
    try:
        return env.reset()
    finally:
        close_env(env)


def board_spaces():
    env = CollectEnv(start_positions=START_POSITIONS, changePlayerPos=False)
    try:
        return env.free_spaces, env.board.shape
    finally:
        close_env(env)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "convergence_returns.h5",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "plots" / "convergence_returns.png",
    )
    parser.add_argument("--goals", type=Path, default=ROOT / "goals.h5")
    parser.add_argument("--map", type=Path, default=ROOT / "map.png")
    parser.add_argument(
        "--value-figure-dir",
        type=Path,
        default=ROOT / "plots" / "value_progression",
    )
    return parser.parse_args()


if __name__ == "__main__":
    make_plots(parse_args())
