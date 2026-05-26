import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EXTENSION_ROOT = SCRIPT_DIR.parent
ROOT = EXTENSION_ROOT.parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = EXTENSION_ROOT / "exp_data_extension_safety"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

import envs  # noqa: F401
from sb3_utils import DQNAgent, TD3Agent
from sm import SkillMachine, TaskPrimitive, evaluate

PRIMITIVE_ENV_ID = "Safety-v0"
TASK_ENVS = {
    "task1": "Safety-Task-1-v0",
    "task2": "Safety-Task-2-v0",
    "task3": "Safety-Task-3-v0",
    "task4": "Safety-Task-4-v0",
    "task5": "Safety-Task-5-v0",
    "task6": "Safety-Task-6-v0",
}
TASK_LABELS = {
    "task1": "Task 1",
    "task2": "Task 2",
    "task3": "Task 3",
    "task4": "Task 4",
    "task5": "Task 5",
    "task6": "Task 6",
}
DEFAULT_STEPS = "100000,200000,400000,600000,800000,1000000"
DEFAULT_OUTPUT = DATA_DIR / "sm_convergence.pkl"
DEFAULT_RUNS_DIR = DATA_DIR / "runs"
DEFAULT_LOG_DIR = DATA_DIR / "logs"


class CheckpointCallback(BaseCallback):
    def __init__(self, primitive_env, run_dir, primitive, checkpoints, wandb_run=None, run_idx=None):
        super().__init__()
        self.primitive_env = primitive_env
        self.run_dir = Path(run_dir)
        self.primitive = primitive
        self.pending = sorted(set(checkpoints))
        self.wandb_run = wandb_run
        self.run_idx = run_idx

    def _on_step(self):
        while self.pending and self.num_timesteps >= self.pending[0]:
            step = self.pending.pop(0)
            out = checkpoint_dir(self.run_dir, step)
            out.mkdir(parents=True, exist_ok=True)
            self.model.save(out / f"wvf_{self.primitive}")
            torch.save(self.primitive_env.goals, out / "goals")
            print(f"[checkpoint] {out}/wvf_{self.primitive}")
            if self.wandb_run:
                self.wandb_run.log(
                    {
                        "train/step": step,
                        "train/run": self.run_idx,
                        "train/primitive": self.primitive,
                        "train/goals": len(self.primitive_env.goals),
                        f"train/wvf_{self.primitive}/checkpoint": 1,
                    },
                    step=step,
                )
        return True


def parse_steps(raw):
    return [int(value) for value in raw.split(",") if value]


def apply_debug_defaults(args):
    if not args.debug:
        return args
    args.maxiters = args.debug_maxiters
    args.eval_episodes = min(args.eval_episodes, args.debug_eval_episodes)
    args.eval_steps = min(args.eval_steps, args.debug_eval_steps)
    args.runs = min(args.runs, args.debug_runs)
    return args


def default_wandb_name(args):
    if args.wandb_name:
        return args.wandb_name
    if args.plot_only:
        return "safety-convergence-plot"
    if args.eval_only:
        return "safety-convergence-eval"
    return f"safety-convergence-run_{args.run:03d}"


def init_wandb(args):
    if not args.wandb:
        return None

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=default_wandb_name(args),
        config=vars(args),
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/training_steps")
    wandb.define_metric("eval/*", step_metric="eval/training_steps")
    return run


def log_eval_results(wandb_run, results):
    if not wandb_run:
        return
    for step in sorted(key for key in results if isinstance(key, int)):
        payload = {"eval/training_steps": step}
        for task_name, metrics in results[step].items():
            for metric_name, value in metrics.items():
                if np.isscalar(value) and np.isfinite(value):
                    payload[f"eval/{task_name}/{metric_name}"] = float(value)
        for metric in ("return", "success_rate"):
            values = np.asarray(
                [results[step][task][metric] for task in results[step]],
                dtype=float,
            )
            payload[f"eval/average/{metric}"] = float(np.mean(values))
            payload[f"eval/average/{metric}_std"] = float(np.std(values))
        wandb_run.log(payload, step=step)


def selected_tasks(raw):
    wanted = set(raw.split(",")) if raw else set(TASK_ENVS)
    tasks = [(name, env_id) for name, env_id in TASK_ENVS.items() if name in wanted]
    if not tasks:
        raise ValueError("No tasks selected.")
    return tasks


def checkpoint_dir(run_dir, step):
    return Path(run_dir) / str(step)


def run_dir(args, run_idx):
    return Path(args.runs_dir) / f"run_{run_idx:03d}"


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_primitive_env():
    return TaskPrimitive(gym.make(PRIMITIVE_ENV_ID), sb3=True)


def make_agent(algo, name, env, save_dir, log_dir, load=False):
    if algo == "dqn":
        return DQNAgent(name, env, str(save_dir) + "/", str(log_dir) + "/", load)
    if algo == "td3":
        return TD3Agent(name, env, str(save_dir) + "/", str(log_dir) + "/", load)
    raise ValueError(f"Unknown algo: {algo}")


def train_run(args):
    steps = parse_steps(args.maxiters)
    set_seed(None if args.seed is None else args.seed + args.run)
    this_run_dir = run_dir(args, args.run)
    this_log_dir = Path(args.log_dir) / f"run_{args.run:03d}"
    this_run_dir.mkdir(parents=True, exist_ok=True)
    this_log_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb(args)
    primitive_env = make_primitive_env()
    try:
        for primitive in ("0", "1"):
            primitive_env.primitive = primitive
            agent = make_agent(args.algo, f"wvf_{primitive}", primitive_env, this_run_dir, this_log_dir)
            existing = [
                (checkpoint_dir(this_run_dir, step) / f"wvf_{primitive}.zip").exists()
                for step in steps
            ]
            if all(existing) and not args.overwrite:
                print(f"[skip] run_{args.run:03d}/wvf_{primitive}")
                continue
            print(f"[train] run_{args.run:03d}/wvf_{primitive} to {max(steps)} steps")
            agent.model.learn(
                max(steps),
                callback=CheckpointCallback(
                    primitive_env,
                    this_run_dir,
                    primitive,
                    steps,
                    wandb_run=wandb_run,
                    run_idx=args.run,
                ),
                reset_num_timesteps=True,
            )
        print(f"Finished run_{args.run:03d}")
    finally:
        primitive_env.close()
        if wandb_run:
            wandb_run.finish()


def complete_run(args, run_idx):
    this_run_dir = run_dir(args, run_idx)
    for step in parse_steps(args.maxiters):
        step_dir = checkpoint_dir(this_run_dir, step)
        for filename in ("goals", "wvf_0.zip", "wvf_1.zip"):
            if not (step_dir / filename).exists():
                return False
    return True


def completed_runs(args):
    return [idx for idx in range(args.runs) if complete_run(args, idx)]


def load_sm(args, run_idx, step):
    step_dir = checkpoint_dir(run_dir(args, run_idx), step)
    primitive_env = make_primitive_env()
    primitive_env.goals.update(torch.load(step_dir / "goals", map_location="cpu"))
    agents = {
        primitive: make_agent(args.algo, f"wvf_{primitive}", primitive_env, step_dir, step_dir, load=True)
        for primitive in ("0", "1")
    }
    return primitive_env, SkillMachine(primitive_env, agents, vectorised=True)


def eval_single(args, run_idx):
    run_results = {}
    for step in tqdm(parse_steps(args.maxiters), desc=f"run_{run_idx:03d}"):
        primitive_env, sm = load_sm(args, run_idx, step)
        run_results[step] = {}
        for task_name, env_id in selected_tasks(args.tasks):
            task_env = gym.make(env_id, test=True)
            reward, success, avg_steps = evaluate(
                task_env,
                SM=sm,
                gamma=args.eval_gamma,
                episodes=args.eval_episodes,
                eval_steps=args.eval_steps,
                seed=None if args.seed is None else args.seed + run_idx * 1000,
            )
            run_results[step][task_name] = {
                "return": float(reward / args.eval_episodes),
                "success_rate": float(success),
                "avg_steps": float(avg_steps),
            }
            task_env.close()
        primitive_env.close()
    return run_results


def aggregate(run_results):
    out = {}
    steps = sorted({step for result in run_results for step in result})
    for step in steps:
        out[step] = {}
        task_names = sorted({task for result in run_results for task in result.get(step, {})})
        for task in task_names:
            out[step][task] = {}
            for metric in ("return", "success_rate", "avg_steps"):
                values = np.asarray(
                    [result[step][task][metric] for result in run_results if task in result.get(step, {})],
                    dtype=float,
                )
                out[step][task][metric] = float(np.mean(values))
                out[step][task][f"{metric}_std"] = float(np.std(values))
    return out


def eval_all(args):
    runs = completed_runs(args)
    if not runs:
        raise RuntimeError("No complete Safety-Gym convergence runs found.")
    print("Evaluating runs:", ", ".join(f"run_{idx:03d}" for idx in runs))
    wandb_run = init_wandb(args)
    results = aggregate([eval_single(args, idx) for idx in runs])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {args.output}")
    log_eval_results(wandb_run, results)
    if wandb_run:
        wandb_run.save(str(args.output))
        wandb_run.finish()
    if args.plot:
        plot_results(results, args.output)


def plot_results(results, output):
    out_dir = Path(output).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(key for key in results if isinstance(key, int))
    task_names = sorted({task for step in steps for task in results[step]})

    plot_metric_by_task(results, steps, task_names, "return", "Episode return", out_dir / "returns.png")
    plot_metric_by_task(results, steps, task_names, "success_rate", "Success rate", out_dir / "successes.png")
    plot_metric_average(results, steps, task_names, "return", "Episode return", out_dir / "average_returns.png")
    plot_metric_average(results, steps, task_names, "success_rate", "Success rate", out_dir / "average_successes.png")
    print(f"Plots saved to {out_dir}")


def plot_metric_by_task(results, steps, task_names, metric, ylabel, path):
    ncols = min(3, len(task_names))
    nrows = int(np.ceil(len(task_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False, sharex=True)
    axes = axes.reshape(-1)
    for ax, task in zip(axes, task_names):
        values = np.asarray([results[step][task][metric] for step in steps], dtype=float)
        stds = np.asarray([results[step][task].get(f"{metric}_std", 0.0) for step in steps], dtype=float)
        lower, upper = values - stds, values + stds
        if metric == "success_rate":
            lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        ax.plot(steps, values, "o-", label="Skill Machine")
        ax.fill_between(steps, lower, upper, alpha=0.2)
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    for ax in axes[len(task_names):]:
        ax.axis("off")
    fig.supxlabel("Training iterations")
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_average(results, steps, task_names, metric, ylabel, path):
    means, spreads = [], []
    for step in steps:
        values = np.asarray([results[step][task][metric] for task in task_names], dtype=float)
        means.append(float(np.mean(values)))
        spreads.append(float(np.std(values)))
    means = np.asarray(means)
    spreads = np.asarray(spreads)
    lower, upper = means - spreads, means + spreads
    if metric == "success_rate":
        lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, means, "o-", label="Task average")
    ax.fill_between(steps, lower, upper, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Training iterations")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_only(args):
    with open(args.output, "rb") as f:
        results = pickle.load(f)
    plot_results(results, args.output)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiters", default=DEFAULT_STEPS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--algo", choices=("td3", "dqn"), default="td3")
    parser.add_argument("--tasks", default="task1,task2,task3,task4,task5,task6")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval-only", action="store_true", dest="eval_only")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--plot-only", action="store_true", dest="plot_only")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_maxiters", default="100,200")
    parser.add_argument("--debug_eval_episodes", type=int, default=2)
    parser.add_argument("--debug_eval_steps", type=int, default=20)
    parser.add_argument("--debug_runs", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="skill-machines-safety-convergence")
    parser.add_argument("--wandb-name", default=None)
    return parser


def main(args):
    args = apply_debug_defaults(args)
    gym.logger.set_level(gym.logger.ERROR)
    if args.plot_only and args.eval_only:
        raise ValueError("--plot_only cannot be used with --eval_only")
    if args.plot_only:
        return plot_only(args)
    if args.eval_only:
        return eval_all(args)
    if args.run is None:
        raise ValueError("Training requires --run, usually 0, 1, or 2.")
    return train_run(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
