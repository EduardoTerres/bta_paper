import argparse
from pathlib import Path
import sys

import deepdish as dd
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

from dqn import Agent, ComposedDQN, ComposedDQN_onoff, FloatTensor
from gym_repoman.envs import CollectEnv
from trainer import load, save
from wrappers import MaxLength, WarpFrame

np.object = object

ROOT = Path(__file__).resolve().parents[1]
START_POSITIONS = {
    "crate_beige": (3, 4),
    "player": (6, 3),
    "circle_purple": (7, 7),
    "circle_beige": (1, 7),
    "crate_blue": (1, 1),
    "crate_purple": (8, 1),
    "circle_blue": (1, 8),
}

# goals.h5 is expected to follow the original Boxman order:
# BC, BS, bS, PS, bC, PC.
TASKS = {
    "B": {
        "goal_indices": [0, 1],
        "condition": lambda x: x.colour == "blue",
    },
    "S": {
        "goal_indices": [1, 2, 3],
        "condition": lambda x: x.shape == "square",
    },
    "B+S": {
        "goal_indices": [0, 1, 2, 3],
        "condition": lambda x: x.colour == "blue" or x.shape == "square",
    },
    "B.S": {
        "goal_indices": [1],
        "condition": lambda x: x.colour == "blue" and x.shape == "square",
    },
    "BxorS": {
        "goal_indices": [0, 2, 3],
        "condition": lambda x: (x.colour == "blue" or x.shape == "square")
        and not (x.colour == "blue" and x.shape == "square"),
    },
}


TRAIN_TASKS = {
    "on": lambda x: True,
    "off": lambda x: False,
    "blue": lambda x: x.colour == "blue",
    "square": lambda x: x.shape == "square",
}


def make_env(condition, max_trajectory=None):
    env = WarpFrame(
        CollectEnv(start_positions=START_POSITIONS, goal_condition=condition)
    )
    if max_trajectory is not None:
        env = MaxLength(env, max_trajectory)
    return env


def train_or_load(
    task_name,
    condition,
    max_timesteps,
    force=False,
    train_log_callback=None,
):
    model_dir = ROOT / "models" / "convergence" / str(max_timesteps) / task_name
    model_path = model_dir / "model.dqn"
    env = make_env(condition)

    if model_path.exists() and not force:
        dqn = load(str(model_path), env, map_location="cpu")
        if torch.cuda.is_available():
            dqn.cuda()
        return dqn

    model_dir.mkdir(parents=True, exist_ok=True)
    agent = Agent(
        env,
        max_timesteps=max_timesteps,
        path=str(model_dir) + "/",
        train_log_callback=train_log_callback,
    )
    agent.train()
    save(str(model_path), agent)
    return agent.q_func


def boolean_composition(dqn_blue, dqn_square, task_name):
    dqn_not_blue = ComposedDQN([dqn_blue], compose="not")
    dqn_not_square = ComposedDQN([dqn_square], compose="not")
    dqn_or = ComposedDQN([dqn_blue, dqn_square], compose="or")
    dqn_and = ComposedDQN([dqn_blue, dqn_square], compose="and")
    dqn_not_and = ComposedDQN([dqn_and], compose="not")

    if task_name == "B":
        return dqn_blue
    if task_name == "S":
        return dqn_square
    if task_name == "B+S":
        return dqn_or
    if task_name == "B.S":
        return dqn_and
    if task_name == "BxorS":
        return ComposedDQN([dqn_or, dqn_not_and], compose="and")
    raise ValueError(f"Unknown task: {task_name}")


def evaluate(dqn, condition, goals, max_trajectory):
    env = make_env(condition, max_trajectory=max_trajectory)
    total_return = 0
    with torch.no_grad():
        obs = env.reset()
        for _ in range(max_trajectory):
            obs_tensor = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0)
            values = []
            for goal in goals:
                goal_tensor = (
                    torch.from_numpy(np.array(goal)).type(FloatTensor).unsqueeze(0)
                )
                values.append(dqn(torch.cat((obs_tensor, goal_tensor), dim=3)).squeeze(0))
            values = torch.stack(values, 1).t()
            action = values.data.max(0)[0].max(0)[1].item()
            obs, reward, done, _ = env.step(action)
            total_return += reward
            if done:
                break
    return total_return


def run(args):
    wandb_run = init_wandb(args)
    goals = dd.io.load(str(ROOT / "goals.h5"))
    returns_per_steps = {}

    for max_timesteps in tqdm(args.max_timesteps, desc="Convergence steps"):
        dqns = {
            name: train_or_load(
                name,
                condition,
                max_timesteps,
                args.force_train,
                make_train_log_callback(wandb_run, name, max_timesteps),
            )
            for name, condition in TRAIN_TASKS.items()
        }
        returns = {}
        for task_name, task in TASKS.items():
            on_goals = [goals[i] for i in task["goal_indices"]]
            dqn_onoff = ComposedDQN_onoff(dqns["on"], dqns["off"], on_goals=on_goals)
            dqn_boolean = boolean_composition(dqns["blue"], dqns["square"], task_name)
            returns[task_name] = {"onoff": [], "boolean": []}
            for _ in range(args.num_runs):
                returns[task_name]["onoff"].append(
                    evaluate(dqn_onoff, task["condition"], goals, args.max_trajectory)
                )
                returns[task_name]["boolean"].append(
                    evaluate(dqn_boolean, task["condition"], goals, args.max_trajectory)
                )
            log_evaluation_returns(wandb_run, max_timesteps, task_name, returns[task_name])
        returns_per_steps[max_timesteps] = returns

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dd.io.save(str(args.output), returns_per_steps)
    plot_convergence(returns_per_steps, args.figure)
    if wandb_run:
        wandb_run.save(str(args.output))
        wandb_run.finish()


def init_wandb(args):
    if not args.wandb:
        return None

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="eval/training_timesteps")
    return run


def make_train_log_callback(wandb_run, task_name, max_timesteps):
    if not wandb_run:
        return None

    metric_prefix = f"train/{max_timesteps}/{task_name}"

    def log_training_step(step, metrics):
        prefixed_metrics = {
            f"{metric_prefix}/{metric_name}": value
            for metric_name, value in metrics.items()
        }
        wandb_run.log({
            "train/step": step,
            "train/max_timesteps": max_timesteps,
            "train/task": task_name,
            **prefixed_metrics,
        })

    return log_training_step


def log_evaluation_returns(wandb_run, max_timesteps, task_name, task_returns):
    if not wandb_run:
        return

    log_data = {
        "eval/training_timesteps": max_timesteps,
        "eval/task": task_name,
    }
    for method, values in task_returns.items():
        prefix = f"eval/{task_name}/{method}"
        log_data[f"{prefix}/mean_return"] = float(np.mean(values))
        log_data[f"{prefix}/stderr_return"] = float(
            np.std(values) / np.sqrt(max(1, len(values)))
        )
    wandb_run.log(log_data)


def plot_convergence(returns_per_steps, figure_path):
    steps = sorted(returns_per_steps)
    methods = ["onoff", "boolean"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        means = []
        stderr = []
        for max_timesteps in steps:
            values = []
            for task_returns in returns_per_steps[max_timesteps].values():
                values.extend(task_returns[method])
            means.append(np.mean(values))
            stderr.append(np.std(values) / np.sqrt(max(1, len(values))))
        ax.errorbar(steps, means, yerr=stderr, marker="o", label=method)
    ax.set_xscale("log")
    ax.set_xlabel("DQN training timesteps")
    ax.set_ylabel("Average return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_path), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-timesteps",
        nargs="+",
        type=int,
        default=[10_000, 25_000, 50_000, 100_000],
    )
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--max-trajectory", type=int, default=20)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="boxman-sts-convergence")
    parser.add_argument("--wandb-name", default=None)
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
