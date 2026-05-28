import argparse
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EXTENSION_ROOT = SCRIPT_DIR.parent
ROOT = EXTENSION_ROOT.parent
PROJECT_ROOT = ROOT.parent


def default_scratch_data_dir():
    username = os.environ.get("USER") or os.environ.get("LOGNAME") or Path.home().name
    return Path("/scratch-shared") / username / "bta_paper" / "safety_gym" / "exps_data_extension"


DATA_DIR = Path(os.environ.get("SAFETY_GYM_DATA_DIR", default_scratch_data_dir()))
FIGURES_DIR = SCRIPT_DIR / "exps_data_extension" / "figures"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

import envs  # noqa: F401
from sb3_utils import DQNAgent, TD3Agent
from sm import MinMaxSkillMachine, SkillMachine, TaskPrimitive

NO_LATEX_RC = {
    "text.usetex": False,
    "text.latex.preamble": "",
}
matplotlib.rcParams.update(NO_LATEX_RC)
plt.rcParams.update(NO_LATEX_RC)

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
METHODS = ("original", "minmax", "boolean")
METHOD_LABELS = {
    "original": "Original",
    "minmax": "Univ./empty",
    "boolean": "Base tasks",
}
COMPOSITION_TIME_METHODS = ("original", "minmax")
COLORS = {
    "original": "#1A5276",
    "minmax": "#C0560A",
    "boolean": "#2E7D32",
}
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 16
DEFAULT_STEPS = "100000,200000,400000,600000,800000,1000000"
DEFAULT_OUTPUT = DATA_DIR / "sm_convergence.pkl"
DEFAULT_RUNS_DIR = DATA_DIR / "runs"
DEFAULT_LOG_DIR = DATA_DIR / "logs"
DEFAULT_FIGURES_DIR = FIGURES_DIR


def disable_latex_rendering():
    matplotlib.rcParams.update(NO_LATEX_RC)
    plt.rcParams.update(NO_LATEX_RC)


class CheckpointCallback(BaseCallback):
    def __init__(self, primitive_env, run_dir, primitive, checkpoints):
        super().__init__()
        self.primitive_env = primitive_env
        self.run_dir = Path(run_dir)
        self.primitive = primitive
        self.pending = sorted(set(checkpoints))

    def _on_step(self):
        while self.pending and self.num_timesteps >= self.pending[0]:
            step = self.pending.pop(0)
            out = checkpoint_dir(self.run_dir, step)
            out.mkdir(parents=True, exist_ok=True)
            self.model.save(out / f"wvf_{self.primitive}")
            torch.save(self.primitive_env.goals, out / f"goals_{self.primitive}")
            print(f"[checkpoint] {out}/wvf_{self.primitive}")
        return True


class WandbSB3Callback(BaseCallback):
    def __init__(self, wandb_run, primitive, primitive_env=None):
        super().__init__()
        self.wandb_run = wandb_run
        self.primitive = primitive
        self.primitive_env = primitive_env
        self.last_n_updates = None

    def _log_pending_scalars(self):
        if not self.wandb_run:
            return

        values = self.logger.name_to_value
        n_updates = values.get("train/n_updates")
        if n_updates is None:
            return

        n_updates = int(n_updates)
        if n_updates == self.last_n_updates:
            return
        self.last_n_updates = n_updates

        prefix = f"train/wvf_{self.primitive}"
        payload = {
            f"{prefix}/n_updates": n_updates,
            f"{prefix}/env_steps": self.num_timesteps,
        }
        if self.primitive_env is not None:
            payload[f"{prefix}/goals"] = len(self.primitive_env.goals)
        for key in ("train/loss", "train/actor_loss", "train/critic_loss", "train/learning_rate"):
            value = values.get(key)
            if np.isscalar(value) and np.isfinite(value):
                payload[f"{prefix}/{key.split('/')[-1]}"] = float(value)

        self.wandb_run.log(payload)

    def _on_step(self):
        self._log_pending_scalars()
        return True

    def _on_training_end(self):
        self._log_pending_scalars()


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
    if args.train_primitive:
        return f"safety-convergence-run_{args.run:03d}-{args.train_primitive}"
    return f"safety-convergence-run_{args.run:03d}"


def init_wandb(args, train_primitives=None):
    if not args.wandb:
        return None

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=default_wandb_name(args),
        config=vars(args),
    )
    for primitive in train_primitives or ():
        prefix = f"train/wvf_{primitive}"
        wandb.define_metric(f"{prefix}/n_updates")
        wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/n_updates")
    wandb.define_metric("eval/training_steps")
    wandb.define_metric("eval/*", step_metric="eval/training_steps")
    return run


def log_eval_results(wandb_run, results):
    if not wandb_run:
        return
    for step in sorted(key for key in results if isinstance(key, int)):
        payload = {"eval/training_steps": step}
        for task_name, task_results in results[step].items():
            for method, metrics in task_results.items():
                prefix = f"eval/{task_name}/{method}"
                for metric_name, value in metrics.items():
                    if np.isscalar(value) and np.isfinite(value):
                        payload[f"{prefix}/{metric_name}"] = float(value)
        for method in METHODS:
            for metric in ("return", "success_rate"):
                values = np.asarray(
                    [
                        results[step][task][method][metric]
                        for task in results[step]
                        if method in results[step][task]
                    ],
                    dtype=float,
                )
                if len(values):
                    payload[f"eval/average/{method}/{metric}"] = float(np.mean(values))
                    payload[f"eval/average/{method}/{metric}_std"] = float(np.std(values))
        wandb_run.log(payload, step=step)


def mujoco_path():
    return Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", "~/.mujoco/mujoco210")).expanduser()


def configure_mujoco_py_backend():
    if "MUJOCO_PY_FORCE_CPU" in os.environ:
        return

    ld_paths = {
        os.path.abspath(path)
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if path
    }
    nvidia_paths = [
        "/usr/local/nvidia/lib64",
        "/usr/lib/nvidia",
        *sorted(str(path) for path in Path("/usr/lib").glob("nvidia-[0-9][0-9][0-9]")),
    ]
    if any(os.path.abspath(path) in ld_paths for path in nvidia_paths if Path(path).exists()):
        return

    os.environ["MUJOCO_PY_FORCE_CPU"] = "1"


def require_mujoco():
    path = mujoco_path()
    if path.exists():
        return
    raise RuntimeError(
        "MuJoCo 2.1 is required before Safety-Gym can run, and no install was found at "
        f"{path}.\n"
        "Install/extract mujoco210 there, or set MUJOCO_PY_MUJOCO_PATH to the existing "
        "mujoco210 directory. This check runs before W&B so failed jobs do not create runs."
    )


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


def base_primitives(primitive_env):
    return (
        ["0", "1"]
        + [f"p_{predicate}" for predicate in primitive_env.predicates]
        + [f"c_{constraint}" for constraint in primitive_env.constraints]
    )


def make_agent(algo, name, env, save_dir, log_dir, load=False):
    log_dir = None if log_dir is None else str(log_dir) + "/"
    if algo == "dqn":
        return DQNAgent(name, env, str(save_dir) + "/", log_dir, load)
    if algo == "td3":
        return TD3Agent(name, env, str(save_dir) + "/", log_dir, load)
    raise ValueError(f"Unknown algo: {algo}")


def train_run(args):
    steps = parse_steps(args.maxiters)
    set_seed(None if args.seed is None else args.seed + args.run)
    this_run_dir = run_dir(args, args.run)
    this_log_dir = Path(args.log_dir) / f"run_{args.run:03d}"
    this_run_dir.mkdir(parents=True, exist_ok=True)
    this_log_dir.mkdir(parents=True, exist_ok=True)

    primitive_env = make_primitive_env()
    primitives = base_primitives(primitive_env)
    if args.train_primitive:
        if args.train_primitive not in primitives:
            raise ValueError(f"Unknown --train-primitive {args.train_primitive}. Choose one of: {', '.join(primitives)}")
        primitives = [args.train_primitive]
    wandb_run = init_wandb(args, train_primitives=primitives)
    try:
        for primitive in primitives:
            primitive_env.primitive = primitive
            primitive_log_dir = this_log_dir / f"wvf_{primitive}"
            primitive_log_dir.mkdir(parents=True, exist_ok=True)
            existing = [
                (checkpoint_dir(this_run_dir, step) / f"wvf_{primitive}.zip").exists()
                for step in steps
            ]
            if any(existing):
                existing_steps = [step for step, exists in zip(steps, existing) if exists]
                print(
                    f"[warning] run_{args.run:03d}/wvf_{primitive} already has "
                    f"{len(existing_steps)} checkpoint(s); training will overwrite them."
                )
            agent = make_agent(args.algo, f"wvf_{primitive}", primitive_env, this_run_dir, primitive_log_dir)
            print(f"[train] run_{args.run:03d}/wvf_{primitive} to {max(steps)} steps")
            agent.model.learn(
                max(steps),
                callback=[
                    WandbSB3Callback(
                        wandb_run,
                        primitive,
                        primitive_env=primitive_env,
                    ),
                    CheckpointCallback(
                        primitive_env,
                        this_run_dir,
                        primitive,
                        steps,
                    ),
                ],
                reset_num_timesteps=True,
            )
        print(f"Finished run_{args.run:03d}")
    finally:
        primitive_env.close()
        if wandb_run:
            wandb_run.finish()


def complete_run(args, run_idx):
    this_run_dir = run_dir(args, run_idx)
    primitive_env = make_primitive_env()
    primitives = base_primitives(primitive_env)
    primitive_env.close()
    for step in parse_steps(args.maxiters):
        step_dir = checkpoint_dir(this_run_dir, step)
        has_split_goals = all((step_dir / f"goals_{primitive}").exists() for primitive in primitives)
        if not has_split_goals and not (step_dir / "goals").exists():
            return False
        for filename in [f"wvf_{primitive}.zip" for primitive in primitives]:
            if not (step_dir / filename).exists():
                return False
    return True


def completed_runs(args):
    return [idx for idx in range(args.runs) if complete_run(args, idx)]


def load_primitives(args, run_idx, step):
    step_dir = checkpoint_dir(run_dir(args, run_idx), step)
    primitive_env = make_primitive_env()
    primitives = base_primitives(primitive_env)
    goals_files = [step_dir / f"goals_{primitive}" for primitive in primitives]
    if any(path.exists() for path in goals_files):
        for path in goals_files:
            if path.exists():
                primitive_env.goals.update(torch.load(path, map_location="cpu"))
    else:
        primitive_env.goals.update(torch.load(step_dir / "goals", map_location="cpu"))
    agents = {
        primitive: make_agent(args.algo, f"wvf_{primitive}", primitive_env, step_dir, None, load=True)
        for primitive in primitives
    }
    return primitive_env, agents


def learned_sm(primitive_env, agents, method):
    if method == "minmax":
        method_agents = {primitive: agents[primitive] for primitive in ("0", "1")}
        sm_cls = MinMaxSkillMachine
    elif method == "original":
        method_agents = {primitive: agents[primitive] for primitive in ("0", "1")}
        sm_cls = SkillMachine
    else:
        method_agents = agents
        sm_cls = SkillMachine
    return sm_cls(primitive_env, method_agents, vectorised=True, goal_directed=True)


def evaluate_once(task_env, sm, method, args, seed):
    episode_returns = []
    episode_successes = []
    episode_steps = []
    composition_times = []

    for episode in range(args.eval_episodes):
        episode_return = 0.0
        episode_success = 0.0
        episode_seed = None if seed is None else seed + episode
        state, info = task_env.reset(seed=episode_seed)
        sm.reset(task_env.rm, info["true_propositions"])

        for step in range(args.eval_steps):
            states = {key: np.expand_dims(value, 0) for key, value in state.items()}
            composition_start = time.perf_counter()
            action = sm.get_action_value(states)[0][0]
            if method in COMPOSITION_TIME_METHODS:
                composition_times.append(time.perf_counter() - composition_start)
            state, reward, done, truncated, info = task_env.step(action)
            sm.step(task_env.rm, info["true_propositions"])

            episode_return += (args.eval_gamma**step) * reward
            episode_success = max(episode_success, float(reward >= task_env.rm.rmax))
            if done or truncated:
                episode_steps.append(step + 1)
                break
        else:
            episode_steps.append(args.eval_steps)

        episode_returns.append(episode_return)
        episode_successes.append(episode_success)

    metrics = {
        "return": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)),
        "success_rate_std": float(np.std(episode_successes)),
        "avg_steps": float(np.mean(episode_steps)),
        "avg_steps_std": float(np.std(episode_steps)),
    }
    if composition_times:
        metrics["composition_time"] = float(np.mean(composition_times))
        metrics["composition_time_std"] = float(np.std(composition_times))
        metrics["composition_times"] = composition_times
    return metrics


def eval_single(args, run_idx):
    run_results = {}
    for step in tqdm(parse_steps(args.maxiters), desc=f"run_{run_idx:03d}"):
        primitive_env, agents = load_primitives(args, run_idx, step)
        run_results[step] = {}
        for task_name, env_id in selected_tasks(args.tasks):
            task_env = gym.make(env_id, test=True)
            run_results[step][task_name] = {}
            for method in METHODS:
                sm = learned_sm(primitive_env, agents, method)
                metrics = evaluate_once(
                    task_env,
                    sm,
                    method,
                    args,
                    seed=None if args.seed is None else args.seed + run_idx * 1000,
                )
                run_results[step][task_name][method] = metrics
            task_env.close()
        primitive_env.close()
    return run_results


def aggregate_metric_dict(metric_dicts):
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    aggregate_metrics = {}
    for key in keys:
        if key.endswith("_times"):
            samples = []
            for metrics in metric_dicts:
                samples.extend(metrics.get(key, []))
            if samples:
                aggregate_metrics[key] = samples
            continue
        if key.endswith("_std") and key[:-4] in keys:
            continue
        values = np.asarray([metrics[key] for metrics in metric_dicts if key in metrics], dtype=float)
        if not len(values):
            continue
        aggregate_metrics[key] = float(np.mean(values))
        std_key = f"{key}_std"
        if std_key in keys:
            aggregate_metrics[std_key] = float(np.std(values))
    return aggregate_metrics


def aggregate(run_results):
    out = {}
    steps = sorted({step for result in run_results for step in result})
    for step in steps:
        out[step] = {}
        task_names = sorted({task for result in run_results for task in result.get(step, {})})
        for task in task_names:
            out[step][task] = {}
            for method in METHODS:
                metrics = [
                    result[step][task][method]
                    for result in run_results
                    if task in result.get(step, {}) and method in result[step][task]
                ]
                if metrics:
                    out[step][task][method] = aggregate_metric_dict(metrics)
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
        plot_results(results, args.output, args.figures_dir)


def plot_results(results, output, figures_dir=None):
    disable_latex_rendering()
    out_dir = Path(figures_dir) if figures_dir else Path(output).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(key for key in results if isinstance(key, int))
    task_names = sorted({task for step in steps for task in results[step]})

    plot_metric_by_task(results, steps, task_names, "return", "Episode return", out_dir / "returns.png")
    plot_metric_by_task(results, steps, task_names, "success_rate", "Success rate", out_dir / "successes.png")
    plot_metric_average(results, steps, task_names, "return", "Episode return", out_dir / "average_returns.png")
    plot_metric_average(results, steps, task_names, "success_rate", "Success rate", out_dir / "average_successes.png")
    plot_composition_time_violins(results, steps, task_names, out_dir)
    print(f"Plots saved to {out_dir}")


def plot_metric_by_task(results, steps, task_names, metric, ylabel, path):
    disable_latex_rendering()
    ncols = min(3, len(task_names))
    nrows = math.ceil(len(task_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False, sharex=True)
    axes = axes.reshape(-1)
    for ax, task in zip(axes, task_names):
        for method in METHODS:
            values = np.asarray(
                [results[step].get(task, {}).get(method, {}).get(metric, np.nan) for step in steps],
                dtype=float,
            )
            stds = np.asarray(
                [results[step].get(task, {}).get(method, {}).get(f"{metric}_std", 0.0) for step in steps],
                dtype=float,
            )
            if not np.isfinite(values).any():
                continue
            lower, upper = values - stds, values + stds
            if metric == "success_rate":
                lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
            ax.plot(
                steps,
                values,
                "o-",
                color=COLORS[method],
                linewidth=2,
                markersize=6,
                label=METHOD_LABELS[method],
            )
            ax.fill_between(steps, lower, upper, color=COLORS[method], alpha=0.2, linewidth=0)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=TITLE_FONTSIZE)
        ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.25)
    for ax in axes[len(task_names):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=LEGEND_FONTSIZE)
    fig.supxlabel("Training iterations per extended value function", fontsize=LABEL_FONTSIZE)
    fig.supylabel(ylabel, fontsize=LABEL_FONTSIZE)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.92))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_average(results, steps, task_names, metric, ylabel, path):
    disable_latex_rendering()
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHODS:
        means, spreads = [], []
        for step in steps:
            values = np.asarray(
                [
                    results[step].get(task, {}).get(method, {}).get(metric, np.nan)
                    for task in task_names
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if len(values):
                means.append(float(np.mean(values)))
                spreads.append(float(np.std(values)))
            else:
                means.append(np.nan)
                spreads.append(0.0)
        means = np.asarray(means)
        spreads = np.asarray(spreads)
        lower, upper = means - spreads, means + spreads
        if metric == "success_rate":
            lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        ax.plot(
            steps,
            means,
            "o-",
            color=COLORS[method],
            linewidth=2,
            markersize=6,
            label=METHOD_LABELS[method],
        )
        ax.fill_between(steps, lower, upper, color=COLORS[method], alpha=0.2, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Training iterations per extended value function", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_composition_time_stats(rows):
    if not rows:
        return

    headers = ("Task", "Step", "Method", "N", "Mean", "Std", "Median", "Q1", "Q3", "Min", "Max")
    table = [headers]
    for task, step, method, values in rows:
        table.append(
            (
                TASK_LABELS.get(task, task),
                str(step),
                METHOD_LABELS[method],
                str(len(values)),
                f"{np.mean(values):.3f}",
                f"{np.std(values):.3f}",
                f"{np.median(values):.3f}",
                f"{np.percentile(values, 25):.3f}",
                f"{np.percentile(values, 75):.3f}",
                f"{np.min(values):.3f}",
                f"{np.max(values):.3f}",
            )
        )

    widths = [max(len(row[idx]) for row in table) for idx in range(len(headers))]
    print("\nComposition time violin statistics (ms)")
    print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(table[0])))
    print("-+-".join("-" * width for width in widths))
    for row in table[1:]:
        print(" | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row)))


def plot_composition_time_violins(results, steps, task_names, out_dir):
    has_values = any(
        results[step][task][method].get("composition_times")
        for step in steps
        for task in task_names
        for method in COMPOSITION_TIME_METHODS
        if task in results[step] and method in results[step][task]
    )
    if not has_values:
        return

    ncols = min(3, len(task_names))
    nrows = math.ceil(len(task_names) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.8 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    offsets = np.linspace(-0.18, 0.18, len(COMPOSITION_TIME_METHODS))
    base_positions = np.arange(len(steps))
    stats_rows = []

    for ax, task in zip(axes, task_names):
        for offset, method in zip(offsets, COMPOSITION_TIME_METHODS):
            samples = []
            positions = []
            for idx, step in enumerate(steps):
                if task not in results[step] or method not in results[step][task]:
                    continue
                values = np.asarray(
                    results[step][task][method].get("composition_times", []),
                    dtype=float,
                )
                values = values[np.isfinite(values)] * 1000.0
                if len(values):
                    samples.append(values)
                    positions.append(base_positions[idx] + offset)
                    stats_rows.append((task, step, method, values))
            if not samples:
                continue
            violins = ax.violinplot(
                samples,
                positions=positions,
                widths=0.28,
                showmeans=True,
                showextrema=False,
            )
            for body in violins["bodies"]:
                body.set_facecolor(COLORS[method])
                body.set_edgecolor(COLORS[method])
                body.set_alpha(0.35)
            violins["cmeans"].set_color(COLORS[method])
            violins["cmeans"].set_linewidth(2)

        ax.set_title(TASK_LABELS.get(task, task), fontsize=TITLE_FONTSIZE)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(step) for step in steps], rotation=30, ha="right")
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.25)

    for ax in axes[len(task_names):]:
        ax.axis("off")

    print_composition_time_stats(stats_rows)

    handles = [
        Patch(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=0.35, label=METHOD_LABELS[method])
        for method in COMPOSITION_TIME_METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=LEGEND_FONTSIZE)
    fig.supxlabel("Training iterations per extended value function", fontsize=LABEL_FONTSIZE)
    fig.supylabel("Composition time per action (ms)", fontsize=LABEL_FONTSIZE)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.92))
    fig.savefig(out_dir / "composition_time_violins.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir / "composition_time_violins.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_only(args):
    with open(args.output, "rb") as f:
        results = pickle.load(f)
    plot_results(results, args.output, args.figures_dir)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiters", default=DEFAULT_STEPS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--train-primitive", default=None)
    parser.add_argument("--algo", choices=("td3", "dqn"), default="td3")
    parser.add_argument("--tasks", default="task1,task2,task3,task4,task5,task6")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval-only", action="store_true", dest="eval_only")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--plot-only", action="store_true", dest="plot_only")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_maxiters", default="100,200,400,600,800,1000")
    parser.add_argument("--debug_eval_episodes", type=int, default=1)
    parser.add_argument("--debug_eval_steps", type=int, default=1)
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
    require_mujoco()
    configure_mujoco_py_backend()
    if args.eval_only:
        return eval_all(args)
    if args.run is None:
        raise ValueError("Training requires --run, usually 0, 1, or 2.")
    return train_run(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
