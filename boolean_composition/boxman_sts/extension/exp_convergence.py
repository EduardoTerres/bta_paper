import argparse
import random
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

FULL_RUNS_DIR = ROOT / "models" / "convergence" / "full_runs"


def make_env(condition, max_trajectory=None):
    env = WarpFrame(
        CollectEnv(start_positions=START_POSITIONS, goal_condition=condition)
    )
    if max_trajectory is not None:
        env = MaxLength(env, max_trajectory)
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(run_idx, step, task_name):
    return (
        FULL_RUNS_DIR
        / f"run_{run_idx:03d}"
        / str(step)
        / task_name
        / "model.dqn"
    )


def run_dir(run_idx):
    return FULL_RUNS_DIR / f"run_{run_idx:03d}"


def run_returns_path(run_idx):
    return run_dir(run_idx) / "convergence_returns.h5"


def existing_run_indices():
    if not FULL_RUNS_DIR.exists():
        return []

    run_indices = []
    for path in FULL_RUNS_DIR.glob("run_*"):
        if not path.is_dir():
            continue
        try:
            run_indices.append(int(path.name.removeprefix("run_")))
        except ValueError:
            continue
    return sorted(run_indices)


def next_run_index():
    run_indices = existing_run_indices()
    if not run_indices:
        return 0
    return run_indices[-1] + 1


def reserve_run_indices(num_runs):
    FULL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    reserved = []
    run_idx = next_run_index()
    while len(reserved) < num_runs:
        try:
            run_dir(run_idx).mkdir()
        except FileExistsError:
            run_idx += 1
            continue
        reserved.append(run_idx)
        run_idx += 1
    return reserved


def is_complete_run(run_idx, checkpoint_steps):
    return all(
        checkpoint_path(run_idx, step, task_name).exists()
        for step in checkpoint_steps
        for task_name in TRAIN_TASKS
    )


def completed_run_indices(checkpoint_steps):
    return [
        run_idx
        for run_idx in existing_run_indices()
        if is_complete_run(run_idx, checkpoint_steps)
    ]


def load_checkpoint(run_idx, step, task_name, condition):
    model_path = checkpoint_path(run_idx, step, task_name)
    dqn = load(str(model_path), make_env(condition), map_location="cpu")
    if torch.cuda.is_available():
        dqn.cuda()
    return dqn


def save_returns(path, returns):
    path.parent.mkdir(parents=True, exist_ok=True)
    dd.io.save(str(path), returns)


def train_or_load_full_run(
    run_idx,
    task_name,
    task_idx,
    condition,
    checkpoint_steps,
    seed=0,
    eps_timesteps=None,
    train_log_callback=None,
):
    checkpoint_steps = sorted(set(checkpoint_steps))
    existing = [
        checkpoint_path(run_idx, step, task_name).exists()
        for step in checkpoint_steps
    ]
    if all(existing):
        return

    set_seed(seed + run_idx * len(TRAIN_TASKS) + task_idx)
    max_timesteps = max(checkpoint_steps)
    pending_steps = set(checkpoint_steps)
    model_dir = (
        run_dir(run_idx)
        / "full"
        / task_name
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_callback(agent, step):
        if step not in pending_steps:
            return
        model_path = checkpoint_path(run_idx, step, task_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save(str(model_path), agent)
        pending_steps.remove(step)

    agent_kwargs = {
        "max_timesteps": max_timesteps,
        "path": str(model_dir) + "/",
        "train_log_callback": train_log_callback,
        "checkpoint_callback": checkpoint_callback,
    }
    if eps_timesteps is not None:
        agent_kwargs["eps_timesteps"] = eps_timesteps

    agent = Agent(make_env(condition), **agent_kwargs)
    agent.train()

    if pending_steps:
        missing = ", ".join(str(step) for step in sorted(pending_steps))
        raise RuntimeError(f"Missing checkpoints for {task_name}: {missing}")


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


def evaluate_single_run(run_idx, checkpoint_steps, goals, max_trajectory):
    returns_per_steps = {}
    for step in tqdm(checkpoint_steps, desc=f"Evaluating run_{run_idx:03d}", leave=False):
        dqns = {
            name: load_checkpoint(run_idx, step, name, condition)
            for name, condition in TRAIN_TASKS.items()
        }
        returns = {}
        for task_name, task in TASKS.items():
            on_goals = [goals[i] for i in task["goal_indices"]]
            dqn_onoff = ComposedDQN_onoff(dqns["on"], dqns["off"], on_goals=on_goals)
            dqn_boolean = boolean_composition(dqns["blue"], dqns["square"], task_name)
            returns[task_name] = {
                "onoff": [
                    evaluate(dqn_onoff, task["condition"], goals, max_trajectory)
                ],
                "boolean": [
                    evaluate(dqn_boolean, task["condition"], goals, max_trajectory)
                ],
            }
        returns_per_steps[step] = returns
    return returns_per_steps


def load_or_evaluate_single_run(run_idx, checkpoint_steps, goals, max_trajectory):
    returns_path = run_returns_path(run_idx)
    if returns_path.exists():
        returns = dd.io.load(str(returns_path))
        if has_requested_returns(returns, checkpoint_steps):
            return returns

    returns = evaluate_single_run(run_idx, checkpoint_steps, goals, max_trajectory)
    save_returns(returns_path, returns)
    return returns


def has_requested_returns(returns, checkpoint_steps):
    return all(
        step in returns
        and task_name in returns[step]
        and all(method in returns[step][task_name] for method in ["onoff", "boolean"])
        for step in checkpoint_steps
        for task_name in TASKS
    )


def aggregate_returns(run_indices, checkpoint_steps, goals, max_trajectory):
    returns_per_steps = {
        step: {
            task_name: {"onoff": [], "boolean": []}
            for task_name in TASKS
        }
        for step in checkpoint_steps
    }

    for run_idx in tqdm(run_indices, desc="Loading run returns"):
        run_returns = load_or_evaluate_single_run(
            run_idx,
            checkpoint_steps,
            goals,
            max_trajectory,
        )
        for step in checkpoint_steps:
            for task_name in TASKS:
                for method in ["onoff", "boolean"]:
                    returns_per_steps[step][task_name][method].extend(
                        run_returns[step][task_name][method]
                    )

    return returns_per_steps


def run(args):
    wandb_run = init_wandb(args)
    checkpoint_steps = sorted(set(args.max_timesteps))
    goals = dd.io.load(str(ROOT / "goals.h5"))
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    if not args.eval_only:
        train_run_indices = reserve_run_indices(args.num_runs)  # for parallel training
        print("Reserved runs:", ", ".join(f"run_{idx:03d}" for idx in train_run_indices))
        for run_idx in tqdm(train_run_indices, desc="Full training runs"):
            for task_idx, (name, condition) in enumerate(TRAIN_TASKS.items()):
                train_or_load_full_run(
                    run_idx,
                    name,
                    task_idx,
                    condition,
                    checkpoint_steps,
                    args.seed,
                    args.eps_timesteps,
                    make_train_log_callback(
                        wandb_run,
                        name,
                        run_idx,
                        max(checkpoint_steps),
                    ),
                )
        for run_idx in train_run_indices:
            run_returns = evaluate_single_run(
                run_idx,
                checkpoint_steps,
                goals,
                args.max_trajectory,
            )
            save_returns(run_returns_path(run_idx), run_returns)
        if wandb_run:
            wandb_run.finish()
        return

    eval_run_indices = completed_run_indices(checkpoint_steps)
    if not eval_run_indices:
        raise RuntimeError("No complete convergence runs found to evaluate.")

    returns_per_steps = aggregate_returns(
        eval_run_indices,
        checkpoint_steps,
        goals,
        args.max_trajectory,
    )
    for step in checkpoint_steps:
        for task_name in TASKS:
            log_evaluation_returns(
                wandb_run,
                step,
                task_name,
                returns_per_steps[step][task_name],
            )

    save_returns(args.output, returns_per_steps)
    plot_convergence(returns_per_steps, args.figure)
    if wandb_run:
        wandb_run.save(str(args.output))
        wandb_run.save(str(args.figure))
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


def make_train_log_callback(wandb_run, task_name, run_idx, max_timesteps):
    if not wandb_run:
        return None

    metric_prefix = f"train/run_{run_idx:03d}/{task_name}"

    def log_training_step(step, metrics):
        prefixed_metrics = {
            f"{metric_prefix}/{metric_name}": value
            for metric_name, value in metrics.items()
        }
        wandb_run.log({
            "train/step": step,
            "train/full_run": run_idx,
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
        stds = []
        for step in steps:
            values = []
            for task_returns in returns_per_steps[step].values():
                values.extend(task_returns[method])
            means.append(np.mean(values))
            stds.append(np.std(values))
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(steps, means, marker="o", label=method)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
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
        help="Checkpoint/evaluation grid. Each training run goes to the largest value.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of full training replicates to average over.",
    )
    parser.add_argument("--max-trajectory", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-timesteps", type=int, default=100_000)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and average all complete saved runs for the requested grid.",
    )
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
