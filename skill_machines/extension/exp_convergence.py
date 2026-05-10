import argparse
import os
import pickle
import random
import sys

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import envs  # noqa: F401
from sm import TaskPrimitive
from sm_ql import QLAgent, learn

PLOT_RC = {
    "text.usetex": False,
    "text.latex.preamble": "",
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(PLOT_RC)

TASK_ENVS = {
    "coffee": "Office-Coffee-Task-v0",
    "patrol": "Office-Patrol-Task-v0",
    "coffee_mail": "Office-CoffeeMail-Task-v0",
    "long": "Office-Long-Task-v0",
}
PRIMITIVE_ENV_ID = "Office-v0"
METHODS = ("boolean", "minmax")
COLORS = {"boolean": "steelblue", "minmax": "tomato"}
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "exps_data_extension", "sm_convergence.pkl")


def parse_maxiters(raw):
    return [int(value) for value in raw.split(",") if value]


def selected_tasks(raw):
    items = list(TASK_ENVS.items())
    if raw:
        wanted = set(raw.split(","))
        items = [(name, env_id) for name, env_id in items if name in wanted]
    if not items:
        raise ValueError("No tasks selected. Use one or more of: " + ", ".join(TASK_ENVS))
    return items


def apply_debug_defaults(args):
    if not args.debug:
        return args
    args.maxiters = args.debug_maxiters
    args.optimal_iters = args.debug_optimal_iters
    args.eval_steps = args.debug_eval_steps
    return args


def make_primitive_env(env_id=PRIMITIVE_ENV_ID):
    return TaskPrimitive(gym.make(env_id))


def save_primitives(primitive_env, SP, sp_dir, args):
    checkpoint_dirs = [sp_dir]
    if args.checkpoint_metric == "both":
        checkpoint_dirs = [os.path.join(sp_dir, "return"), os.path.join(sp_dir, "success")]

    for checkpoint_dir in checkpoint_dirs:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(primitive_env.goals, os.path.join(checkpoint_dir, "goals"))
        for primitive, agent in SP.items():
            torch.save(agent.values, os.path.join(checkpoint_dir, "wvf_" + primitive))


def restore_best_primitives(primitive_env, SP, sp_dir):
    """Load saved primitive WVFs into the in-memory agents."""
    goals_path = os.path.join(sp_dir, "goals")
    if not os.path.exists(goals_path):
        return SP

    primitive_env.goals.clear()
    primitive_env.goals.update(torch.load(goals_path))
    for primitive, agent in SP.items():
        wvf_path = os.path.join(sp_dir, "wvf_" + primitive)
        if os.path.exists(wvf_path):
            agent.values = torch.load(wvf_path)
    return SP


def metric_checkpoint_dir(sp_dir, metric, args):
    if args.checkpoint_metric == "both":
        return os.path.join(sp_dir, metric)
    return sp_dir


def checkpoint_env(args):
    return args.checkpoint_env or args.primitive_env


def primitive_run_name(total_steps, env_id, label="boolean"):
    return f"{env_id}/{label}/{total_steps}"


def primitive_run_dir(total_steps, args, label="boolean", env_id=None):
    return os.path.join(args.sp_dir, primitive_run_name(total_steps, env_id or args.primitive_env, label)) + "/"


def require_checkpoint(sp_dir, args):
    checkpoint_dirs = [sp_dir]
    if args.checkpoint_metric == "both":
        checkpoint_dirs = [os.path.join(sp_dir, "return"), os.path.join(sp_dir, "success")]

    missing = []
    for checkpoint_dir in checkpoint_dirs:
        for filename in ("goals", "wvf_0", "wvf_1"):
            path = os.path.join(checkpoint_dir, filename)
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing primitive checkpoint files for evaluation-only mode:\n"
            + "\n".join(f"  {path}" for path in missing)
        )


def init_primitives_once(total_steps, args, label="boolean"):
    primitive_env = make_primitive_env(args.primitive_env)
    sp_dir = primitive_run_dir(total_steps, args, label, env_id=checkpoint_env(args))
    require_checkpoint(sp_dir, args)
    SP = {
        primitive: QLAgent(
            primitive,
            primitive_env,
            lr=args.lr,
            gamma=args.gamma,
            qinit=args.qinit,
        )
        for primitive in ("0", "1")
    }
    return primitive_env, SP, sp_dir


def train_primitives_once(total_steps, args, label="boolean"):
    primitive_env = make_primitive_env(args.primitive_env)
    run_name = primitive_run_name(total_steps, args.primitive_env, label)
    log_dir = os.path.join(args.log_dir, run_name) + "/"
    sp_dir = primitive_run_dir(total_steps, args, label)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sp_dir, exist_ok=True)

    print(f"\n[train] {run_name}")
    SP = learn(
        primitive_env,
        None,
        total_steps,
        sp_dir=sp_dir,
        log_dir=log_dir,
        gamma=args.gamma,
        lr=args.lr,
        epsilon=args.epsilon,
        qinit=args.qinit,
        eval_episodes=args.num_runs,
        print_freq=args.print_freq,
        seed=args.seed,
        minmax=False,
        checkpoint_metric=args.checkpoint_metric,
    )
    save_primitives(primitive_env, SP, sp_dir, args)
    return primitive_env, SP, sp_dir


def evaluate_once(primitive_env, task_env, SP, env_id, method, total_steps, args):
    run_name = f"{env_id}/{method}/{total_steps}"
    print(f"[eval] {run_name}")
    sm = learned_sm(primitive_env, SP, method)
    episode_returns = []
    episode_successes = []
    episode_steps = []

    for episode in range(args.num_runs):
        episode_return = 0.0
        episode_success = 0.0
        episode_seed = None if args.seed is None else args.seed + episode
        state, info = task_env.reset(seed=episode_seed)
        sm.reset(task_env.rm, info["true_propositions"])

        for step in range(args.eval_steps):
            states = {k: np.expand_dims(v, 0) for (k, v) in state.items()}
            action = sm.get_action_value(states)[0][0]
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

    return {
        "return": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)),
        "success_rate_std": float(np.std(episode_successes)),
        "avg_steps": float(np.mean(episode_steps)),
        "avg_steps_std": float(np.std(episode_steps)),
    }


def learned_sm(primitive_env, SP, method):
    from sm import SkillMachine, MinMaxSkillMachine

    sm_cls = MinMaxSkillMachine if method == "minmax" else SkillMachine
    return sm_cls(primitive_env, SP, goal_directed=True)


def scalar_value(value):
    return float(np.asarray(value).reshape(-1)[0])


def value_error_once(primitive_env, SP, optimal_primitive_env, optimal_SP, env_id, method, args):
    task_env = gym.make(env_id)
    current_sm = learned_sm(primitive_env, SP, method)
    optimal_sm = learned_sm(optimal_primitive_env, optimal_SP, method)
    errors = []

    for episode in range(args.value_episodes):
        episode_seed = None if args.seed is None else args.seed + episode
        state, info = task_env.reset(seed=episode_seed)
        current_sm.reset(task_env.rm, info["true_propositions"])
        optimal_sm.reset(task_env.rm, info["true_propositions"])

        for _ in range(args.eval_steps):
            states = {k: np.expand_dims(v, 0) for (k, v) in state.items()}
            action, current_value = current_sm.get_action_value(states)
            _, optimal_value = optimal_sm.get_action_value(states)
            errors.append(abs(scalar_value(current_value) - scalar_value(optimal_value)))

            state, _, done, truncated, info = task_env.step(action[0])
            current_sm.step(task_env.rm, info["true_propositions"])
            optimal_sm.step(task_env.rm, info["true_propositions"])
            if done or truncated:
                break

    if not errors:
        return float("nan"), float("nan")
    return float(np.mean(errors)), float(np.std(errors))


def apply_max_observed_optimal(results):
    """Use the best observed task metric as the plotted/saved optimal reference."""
    maxiters = sorted(key for key in results if isinstance(key, int))
    optimal_results = results.setdefault("optimal", {})
    task_names = sorted(
        {
            task
            for maxiter in maxiters
            for task in results[maxiter]
        }
    )

    for task_name in task_names:
        optimal_results.setdefault(task_name, {})
        for method in METHODS:
            candidates = [
                results[maxiter][task_name][method]
                for maxiter in maxiters
                if task_name in results[maxiter] and method in results[maxiter][task_name]
            ]
            if not candidates:
                continue
            optimal_metrics = optimal_results[task_name].setdefault(method, {})

            best_return = max(candidates, key=lambda item: item.get("return", float("-inf")))
            optimal_metrics["return"] = best_return.get("return", optimal_metrics.get("return", float("nan")))
            optimal_metrics["return_std"] = best_return.get("return_std", optimal_metrics.get("return_std", 0.0))
            optimal_metrics["avg_steps"] = best_return.get("avg_steps", optimal_metrics.get("avg_steps", float("nan")))
            optimal_metrics["avg_steps_std"] = best_return.get(
                "avg_steps_std",
                optimal_metrics.get("avg_steps_std", 0.0),
            )

            best_success = max(candidates, key=lambda item: item.get("success_rate", float("-inf")))
            optimal_metrics["success_rate"] = best_success.get(
                "success_rate",
                optimal_metrics.get("success_rate", float("nan")),
            )
            optimal_metrics["success_rate_std"] = best_success.get(
                "success_rate_std",
                optimal_metrics.get("success_rate_std", 0.0),
            )
            optimal_metrics["success_avg_steps"] = best_success.get(
                "success_avg_steps",
                best_success.get("avg_steps", optimal_metrics.get("success_avg_steps", float("nan"))),
            )
            optimal_metrics["success_avg_steps_std"] = best_success.get(
                "success_avg_steps_std",
                best_success.get("avg_steps_std", optimal_metrics.get("success_avg_steps_std", 0.0)),
            )


def plot_results(results, output):
    with plt.rc_context(PLOT_RC):
        output_dir = os.path.dirname(output) or "."
        figure_dir = os.path.join(output_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)

        optimal_results = results.get("optimal", {})
        maxiters = sorted(key for key in results if isinstance(key, int))
        task_names = sorted(
            {
                task
                for key, tasks in results.items()
                if isinstance(key, int)
                for task in tasks
            }
        )
        metrics = (
            ("returns", "return", "Reward"),
            ("successes", "success_rate", "Success rate"),
            ("value_error", "value_error", "Mean absolute value error"),
        )

        for filename, key, ylabel in metrics:
            has_values = any(
                np.isfinite(results[maxiter][task_name][method].get(key, np.nan))
                for maxiter in maxiters
                for task_name in task_names
                for method in METHODS
                if task_name in results[maxiter] and method in results[maxiter][task_name]
            )
            if not has_values:
                continue
            fig, axes = plt.subplots(1, len(task_names), figsize=(5 * len(task_names), 4), squeeze=False)
            for ax, task_name in zip(axes[0], task_names):
                for method in METHODS:
                    values = [results[maxiter][task_name][method].get(key, np.nan) for maxiter in maxiters]
                    std_key = f"{key}_std"
                    stds = [results[maxiter][task_name][method].get(std_key, 0.0) for maxiter in maxiters]
                    values = np.asarray(values, dtype=float)
                    stds = np.asarray(stds, dtype=float)
                    lower = values - stds
                    upper = values + stds
                    if key == "success_rate":
                        lower = np.clip(lower, 0.0, 1.0)
                        upper = np.clip(upper, 0.0, 1.0)
                    ax.plot(maxiters, values, marker="o", label=method, color=COLORS[method])
                    ax.fill_between(maxiters, lower, upper, color=COLORS[method], alpha=0.18, linewidth=0)
                    if key != "value_error" and task_name in optimal_results:
                        optimal_value = optimal_results[task_name][method][key]
                        optimal_std = optimal_results[task_name][method].get(std_key, 0.0)
                        ax.axhline(
                            optimal_value,
                            linestyle="--",
                            linewidth=1,
                            color=COLORS[method],
                            alpha=0.8,
                            label=f"{method} optimal",
                        )
                        if optimal_std:
                            opt_lower = optimal_value - optimal_std
                            opt_upper = optimal_value + optimal_std
                            if key == "success_rate":
                                opt_lower = max(0.0, opt_lower)
                                opt_upper = min(1.0, opt_upper)
                            ax.axhspan(opt_lower, opt_upper, color=COLORS[method], alpha=0.08)
                ax.set_title(task_name)
                ax.set_xlabel("Training steps")
                ax.set_ylabel(ylabel)
                if len(maxiters) > 1:
                    ax.set_xscale("log")
                ax.legend()
            for text in fig.findobj(match=matplotlib.text.Text):
                text.set_usetex(False)
            fig.tight_layout()
            fig.savefig(os.path.join(figure_dir, f"{filename}.png"), dpi=200)
            fig.savefig(os.path.join(figure_dir, f"{filename}.pdf"))
            plt.close(fig)
        print(f"Plots saved to {figure_dir}")


def run(args):
    args = apply_debug_defaults(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    gym.logger.set_level(gym.logger.ERROR)

    maxiters = parse_maxiters(args.maxiters)
    task_items = selected_tasks(args.tasks)
    results = {}
    get_primitives = init_primitives_once if args.eval_only else train_primitives_once
    use_long_run_optimal = args.optimal_reference == "long_run"
    optimal_primitive_env = None
    optimal_SP = None
    optimal_sp_dir = None

    if use_long_run_optimal:
        results["optimal"] = {}
        optimal_primitive_env, optimal_SP, optimal_sp_dir = get_primitives(args.optimal_iters, args, label="optimal")
        for task_name, env_id in tqdm(task_items, desc="Optimal references"):
            optimal_task_env = gym.make(env_id)
            results["optimal"][task_name] = {}
            for method in METHODS:
                restore_best_primitives(
                    optimal_primitive_env,
                    optimal_SP,
                    metric_checkpoint_dir(optimal_sp_dir, "return", args),
                )
                metrics = evaluate_once(
                    optimal_primitive_env,
                    optimal_task_env,
                    optimal_SP,
                    env_id,
                    method,
                    args.optimal_iters,
                    args,
                )
                restore_best_primitives(
                    optimal_primitive_env,
                    optimal_SP,
                    metric_checkpoint_dir(optimal_sp_dir, "success", args),
                )
                success_metrics = evaluate_once(
                    optimal_primitive_env,
                    optimal_task_env,
                    optimal_SP,
                    env_id,
                    method,
                    args.optimal_iters,
                    args,
                )
                metrics["success_rate"] = success_metrics["success_rate"]
                metrics["success_rate_std"] = success_metrics["success_rate_std"]
                metrics["success_avg_steps"] = success_metrics["avg_steps"]
                metrics["success_avg_steps_std"] = success_metrics["avg_steps_std"]
                metrics["value_error"] = 0.0
                metrics["value_error_std"] = 0.0
                results["optimal"][task_name][method] = metrics
                print(
                    f"{task_name} {method} optimal @ {args.optimal_iters}: "
                    f"return={metrics['return']:.3f} "
                    f"success={metrics['success_rate']:.3f} "
                    f"avg_steps={metrics['avg_steps']:.1f}"
                )
    elif args.optimal_reference == "max_observed":
        print("Skipping long-run optimal training; using max observed return/success for plot references.")
    else:
        print("Skipping optimal reference training and plotting.")

    for total_steps in tqdm(maxiters, desc="Training budgets"):
        results[total_steps] = {}
        primitive_env, SP, sp_dir = get_primitives(total_steps, args)
        for task_name, env_id in tqdm(task_items, desc=f"Tasks @ {total_steps}", leave=False):
            results[total_steps][task_name] = {}
            task_env = gym.make(env_id)
            for method in METHODS:
                restore_best_primitives(
                    primitive_env,
                    SP,
                    metric_checkpoint_dir(sp_dir, "return", args),
                )
                metrics = evaluate_once(primitive_env, task_env, SP, env_id, method, total_steps, args)
                if use_long_run_optimal:
                    restore_best_primitives(
                        optimal_primitive_env,
                        optimal_SP,
                        metric_checkpoint_dir(optimal_sp_dir, "return", args),
                    )
                    metrics["value_error"], metrics["value_error_std"] = value_error_once(
                        primitive_env,
                        SP,
                        optimal_primitive_env,
                        optimal_SP,
                        env_id,
                        method,
                        args,
                    )
                else:
                    metrics["value_error"] = float("nan")
                    metrics["value_error_std"] = float("nan")
                restore_best_primitives(
                    primitive_env,
                    SP,
                    metric_checkpoint_dir(sp_dir, "success", args),
                )
                success_metrics = evaluate_once(primitive_env, task_env, SP, env_id, method, total_steps, args)
                metrics["success_rate"] = success_metrics["success_rate"]
                metrics["success_rate_std"] = success_metrics["success_rate_std"]
                metrics["success_avg_steps"] = success_metrics["avg_steps"]
                metrics["success_avg_steps_std"] = success_metrics["avg_steps_std"]
                results[total_steps][task_name][method] = metrics
                print(
                    f"{task_name} {method} @ {total_steps}: "
                    f"return={metrics['return']:.3f} "
                    f"success={metrics['success_rate']:.3f} "
                    f"avg_steps={metrics['avg_steps']:.1f} "
                    f"value_error={metrics['value_error']:.4f}"
                )

    if args.optimal_reference == "max_observed":
        apply_max_observed_optimal(results)
        print("Using max observed return/success as optimal plotting reference.")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {args.output}")
    if args.plot:
        plot_results(results, args.output)
    return results


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiters", default="100000,200000,400000,600000,800000,1000000", help="Comma-separated training steps for each experiment")
    parser.add_argument("--optimal_iters", type=int, default=5000000, help="Training steps for the long-run optimal value-function reference")
    parser.add_argument(
        "--optimal_reference",
        choices=("long_run", "max_observed", "none"),
        default="long_run",
        help=(
            "Optimal reference handling: train/evaluate a long-run checkpoint, "
            "use the best observed return/success without long-run training, or omit optimal lines."
        ),
    )
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--tasks", default="coffee_mail", help="Comma-separated subset: coffee,patrol,coffee_mail,long")
    parser.add_argument("--primitive_env", default=PRIMITIVE_ENV_ID, help="Base environment used to train shared primitive WVFs")
    parser.add_argument("--checkpoint_env", default=None, help="Checkpoint path environment prefix used by --eval_only; defaults to --primitive_env")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eval_gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--qinit", type=float, default=0.0)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--value_episodes", type=int, default=10, help="Episodes sampled to estimate convergence to the optimal value function")
    parser.add_argument("--print_freq", type=int, default=10000)
    parser.add_argument("--checkpoint_metric", choices=("success", "return", "both"), default="both", help="Metric used to keep the best primitive checkpoint during training")
    parser.add_argument("--sp_dir", default=os.path.join(SCRIPT_DIR, "exps_data_extension", "sp_ql"))
    parser.add_argument("--log_dir", default=os.path.join(SCRIPT_DIR, "exps_data_extension", "logs"))
    parser.add_argument("--eval_only", action="store_true", help="Skip primitive training and evaluate existing shared checkpoints")
    parser.add_argument("--plot", action="store_true", default=True, help="Save convergence plots")
    parser.add_argument("--no_plot", action="store_false", dest="plot", help="Disable plot generation")
    parser.add_argument("--debug", action="store_true", help="Use low training/evaluation steps with the same experiment setup")
    parser.add_argument("--debug_maxiters", default="100000", help="Training steps used when --debug is set")
    parser.add_argument("--debug_optimal_iters", type=int, default=200000, help="Optimal-reference training steps used when --debug is set")
    parser.add_argument("--debug_eval_steps", type=int, default=50, help="Evaluation horizon used when --debug is set")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
