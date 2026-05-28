import argparse
import copy
import math
import os
import pickle
import random
import sys
import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import rc
from matplotlib.patches import Patch
from stable_baselines3.common.logger import configure
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSION_ROOT = os.path.dirname(SCRIPT_DIR)
SKILL_MACHINES_ROOT = os.path.dirname(EXTENSION_ROOT)
PROJECT_ROOT = os.path.dirname(SKILL_MACHINES_ROOT)
for path in (SKILL_MACHINES_ROOT, EXTENSION_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import envs  # noqa: F401
from sm import TaskPrimitive
from sm_ql import QLAgent

PLOT_RC = {
    "text.usetex": True,
    "text.latex.preamble": "",
}
plt.rcParams.update(PLOT_RC)

TASK_ENVS = {
    "coffee": "Office-Coffee-Task-v0",
    "patrol": "Office-Patrol-Task-v0",
    "coffee_mail": "Office-CoffeeMail-Task-v0",
    "long": "Office-Long-Task-v0",
}
PRIMITIVE_ENV_ID = "Office-v0"
METHODS = ("original", "minmax", "boolean")
METHOD_LABELS = {
    "original": r"Original",
    "minmax": r"Univ./empty",
    "boolean": r"Base tasks",
}
COMPOSITION_TIME_METHODS = METHODS
COLORS = {
    "original": "#1A5276",
    "minmax": "#C0560A",
    "boolean": "#2E7D32",
}
CONVERGENCE_MARKER = "^-"
CONVERGENCE_MARKER_SIZE = 8
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 16
TASK_LABELS = {
    "coffee": "Coffee",
    "patrol": "Patrol",
    "coffee_mail": "Coffee and Mail",
    "long": "Long",
}
RMIN_ENV_VARS = ("RMIN", "SM_RMIN")
RUN_LABELS = ("boolean", "optimal")
DEFAULT_RUNS = 3
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "exps_data_extension", "sm_convergence_new.pkl")
DEFAULT_RUNS_DIR = os.path.join(SCRIPT_DIR, "exps_data_extension", "runs_new")
DEFAULT_SP_DIR = os.path.join(SCRIPT_DIR, "exps_data_extension", "sp_ql_new")
DEFAULT_LOG_DIR = os.path.join(SCRIPT_DIR, "exps_data_extension", "logs_new")


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
    args.num_runs = min(args.num_runs, args.debug_num_runs)
    args.value_episodes = min(args.value_episodes, args.debug_value_episodes)
    args.print_freq = min(args.print_freq, args.debug_print_freq)
    return args


def value_slug(value):
    return str(value).replace("-", "neg").replace(".", "p")


def env_float(names):
    for name in names:
        raw = os.environ.get(name)
        if raw not in (None, ""):
            return float(raw), name
    return None, None


def resolve_rmin(args):
    if args.rmin is not None:
        args.rmin_source = "--rmin"
        return args

    rmin, source = env_float(RMIN_ENV_VARS)
    if rmin is None:
        args.rmin = 0.0
        args.rmin_source = "default"
    else:
        args.rmin = rmin
        args.rmin_source = f"${source}"
    return args


def configure_output_paths(args):
    run_id = args.run_id
    using_default_output = args.output == DEFAULT_OUTPUT
    if run_id is None and args.rmin_source != "default":
        run_id = f"rmin_{value_slug(args.rmin)}"
    args.run_id = run_id

    if run_id:
        args.sp_dir = os.path.join(args.sp_dir, run_id)
        args.log_dir = os.path.join(args.log_dir, run_id)
        if using_default_output:
            args.output = os.path.join(
                SCRIPT_DIR,
                "exps_data_extension",
                run_id,
                "sm_convergence_new.pkl",
            )
    return args


def ensure_no_existing_file(path, args, description):
    if os.path.exists(path) and not args.overwrite:
        raise FileExistsError(
            f"{description} already exists: {path}\n"
            "Choose a different --output/--run_id."
        )


def ensure_writable_run_dir(path, args, description):
    if os.path.isdir(path) and os.listdir(path) and not args.overwrite:
        raise FileExistsError(
            f"{description} already has files: {path}\n"
            "Choose a different --run_id/--sp_dir/--log_dir."
        )
    os.makedirs(path, exist_ok=True)


def make_primitive_env(args, env_id=PRIMITIVE_ENV_ID):
    return TaskPrimitive(gym.make(env_id), rmin=args.rmin)


def base_primitives(primitive_env):
    return (
        ["0", "1"]
        + [f"p_{predicate}" for predicate in primitive_env.predicates]
        + [f"c_{constraint}" for constraint in primitive_env.constraints]
    )


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
    primitive_env = make_primitive_env(args, checkpoint_env(args))
    filenames = ["goals"] + [f"wvf_{primitive}" for primitive in base_primitives(primitive_env)]
    primitive_env.close()

    for checkpoint_dir in checkpoint_dirs:
        for filename in filenames:
            path = os.path.join(checkpoint_dir, filename)
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing primitive checkpoint files for evaluation-only mode:\n"
            + "\n".join(f"  {path}" for path in missing)
        )


def init_primitives_once(total_steps, args, label="boolean"):
    primitive_env = make_primitive_env(args, args.primitive_env)
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
        for primitive in base_primitives(primitive_env)
    }
    return primitive_env, SP, sp_dir


def learn_base_primitives(
    primitive_env,
    total_steps,
    log_dir,
    args,
    checkpoint_steps=None,
    checkpoint_callback=None,
    wandb_run=None,
):
    SP = {
        primitive: QLAgent(
            primitive,
            primitive_env,
            lr=args.lr,
            gamma=args.gamma,
            qinit=args.qinit,
        )
        for primitive in base_primitives(primitive_env)
    }

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    pending_checkpoints = sorted(set(checkpoint_steps or []))
    step, reward_total, successes, num_episodes, start_time = 0, 0, 0, 1, time.time()
    while step < total_steps:
        episode_seed = None if args.seed is None else args.seed + num_episodes - 1
        state, _ = primitive_env.reset(seed=episode_seed)

        while True:
            if random.random() < args.epsilon:
                action = primitive_env.environment.action_space.sample()
            else:
                values = SP["1"].get_values(state)
                action = random.choice([idx for idx, value in enumerate(values) if value == values.max()])

            state_, reward, done, truncated, info = primitive_env.step(action)
            tp_state, tp_state_ = state.copy(), state_.copy()
            for primitive, agent in SP.items():
                primitive_env.primitive = primitive
                for desired_goal in primitive_env.goals.values():
                    tp_state["desired_goal"], tp_state_["desired_goal"] = desired_goal, desired_goal
                    tp_reward = primitive_env.compute_reward(
                        info["achieved_goal"].reshape(1, -1),
                        desired_goal.reshape(1, -1),
                        info["env_reward"],
                        done,
                    )[0]
                    agent.update_values(tp_state, action, tp_reward, tp_state_, done)

            step += 1
            while pending_checkpoints and step >= pending_checkpoints[0]:
                checkpoint_step = pending_checkpoints.pop(0)
                if checkpoint_callback is not None:
                    checkpoint_callback(checkpoint_step, SP)
            state = state_
            if (step - 1) % args.print_freq == 0:
                success_rate = successes / num_episodes
                elapsed = time.time() - start_time
                logger.record("steps", step)
                logger.record("episodes", num_episodes)
                logger.record("goals", len(primitive_env.goals))
                logger.record("primitives", len(SP))
                logger.record("total reward", reward_total)
                logger.record("successes", success_rate)
                logger.record("time elapsed", elapsed)
                logger.dump(step)
                if wandb_run:
                    wandb_run.log(
                        {
                            "train/n_updates": step,
                            "train/episodes": num_episodes,
                            "train/goals": len(primitive_env.goals),
                            "train/primitives": len(SP),
                            "train/total_reward": reward_total,
                            "train/success_rate": success_rate,
                            "train/time_elapsed": elapsed,
                        }
                    )
                reward_total, successes, num_episodes, start_time = 0, 0, 1, time.time()
            if done or truncated:
                num_episodes += 1
                reward_total += reward
                successes += reward >= primitive_env.rmax
                break

    return SP


def train_primitives_once(total_steps, args, label="boolean"):
    primitive_env = make_primitive_env(args, args.primitive_env)
    run_name = primitive_run_name(total_steps, args.primitive_env, label)
    log_dir = os.path.join(args.log_dir, run_name) + "/"
    sp_dir = primitive_run_dir(total_steps, args, label)
    ensure_writable_run_dir(log_dir, args, "Primitive log directory")
    ensure_writable_run_dir(sp_dir, args, "Primitive checkpoint directory")

    print(f"\n[train] {run_name}")
    SP = learn_base_primitives(
        primitive_env,
        total_steps,
        log_dir,
        args,
        checkpoint_steps=[total_steps],
        checkpoint_callback=lambda _step, sp: save_primitives(primitive_env, sp, sp_dir, args),
    )
    return primitive_env, SP, sp_dir


def train_primitives_schedule(total_steps_list, args, label="boolean", wandb_run=None):
    checkpoint_steps = sorted(set(total_steps_list))
    total_steps = max(checkpoint_steps)
    primitive_env = make_primitive_env(args, args.primitive_env)
    run_name = primitive_run_name(total_steps, args.primitive_env, label)
    log_dir = os.path.join(args.log_dir, run_name) + "/"
    ensure_writable_run_dir(log_dir, args, "Primitive log directory")
    for checkpoint_step in checkpoint_steps:
        ensure_writable_run_dir(
            primitive_run_dir(checkpoint_step, args, label),
            args,
            "Primitive checkpoint directory",
        )

    print(f"\n[train] {run_name} with checkpoints {checkpoint_steps}")

    def save_checkpoint(checkpoint_step, SP):
        sp_dir = primitive_run_dir(checkpoint_step, args, label)
        save_primitives(primitive_env, SP, sp_dir, args)
        print(f"[checkpoint] {primitive_run_name(checkpoint_step, args.primitive_env, label)}")
        log_training_checkpoint(wandb_run, args, label, checkpoint_step)

    SP = learn_base_primitives(
        primitive_env,
        total_steps,
        log_dir,
        args,
        checkpoint_steps=checkpoint_steps,
        checkpoint_callback=save_checkpoint,
        wandb_run=wandb_run,
    )
    return primitive_env, SP, primitive_run_dir(total_steps, args, label)


def default_wandb_name(args, suffix=None):
    if args.wandb_name:
        base = args.wandb_name
    elif args.eval_only:
        base = "eval"
    elif args.run is not None and args.train_label:
        base = f"run_{args.run:03d}-{args.train_label}"
    else:
        base = "convergence"
    return f"{base}-{suffix}" if suffix else base


def init_wandb(args, suffix=None):
    if not args.wandb:
        return None

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=default_wandb_name(args, suffix),
        config=vars(args),
    )
    wandb.define_metric("train/n_updates")
    wandb.define_metric("train/*", step_metric="train/n_updates")
    wandb.define_metric("eval/training_steps")
    wandb.define_metric("eval/*", step_metric="eval/training_steps")
    return run


def log_training_checkpoint(wandb_run, args, label, total_steps):
    if not wandb_run:
        return
    wandb_run.log(
        {
            "train/n_updates": total_steps,
            "train/run": args.run,
            "train/rmin": args.rmin,
            "train/complete": 1,
        },
    )


def log_eval_results(wandb_run, results):
    if not wandb_run:
        return
    for total_steps in sorted(key for key in results if isinstance(key, int)):
        payload = {"eval/training_steps": total_steps}
        for task_name, task_results in results[total_steps].items():
            for method, metrics in task_results.items():
                prefix = f"eval/{task_name}/{method}"
                for metric_name, value in metrics.items():
                    if np.isscalar(value) and np.isfinite(value):
                        payload[f"{prefix}/{metric_name}"] = float(value)
        wandb_run.log(payload, step=total_steps)


def learned_sm(primitive_env, SP, method):
    from sm import SkillMachine, MinMaxSkillMachine

    if method == "minmax":
        SP = {primitive: SP[primitive] for primitive in ("0", "1")}
        sm_cls = MinMaxSkillMachine
    elif method == "original":
        SP = {primitive: SP[primitive] for primitive in ("0", "1")}
        sm_cls = SkillMachine
    else:
        sm_cls = SkillMachine
    return sm_cls(primitive_env, SP, goal_directed=True)


def evaluate_once(primitive_env, task_env, SP, env_id, method, total_steps, args):
    run_name = f"{env_id}/{method}/{total_steps}"
    print(f"[eval] {run_name}")
    sm = learned_sm(primitive_env, SP, method)
    episode_returns = []
    episode_successes = []
    episode_steps = []
    composition_times = []

    for episode in range(args.num_runs):
        episode_return = 0.0
        episode_success = 0.0
        episode_seed = None if args.seed is None else args.seed + episode
        state, info = task_env.reset(seed=episode_seed)
        sm.reset(task_env.rm, info["true_propositions"])

        for step in range(args.eval_steps):
            states = {k: np.expand_dims(v, 0) for (k, v) in state.items()}
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


def run_once(args):
    args = apply_debug_defaults(args)
    args = resolve_rmin(args)
    args = configure_output_paths(args)
    ensure_no_existing_file(args.output, args, "Results output")
    print(f"Using rmin={args.rmin:g} ({args.rmin_source})")
    if args.run_id:
        print(f"Using run_id={args.run_id}")
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
            optimal_task_env = gym.make(env_id, rmin=args.rmin)
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
            task_env = gym.make(env_id, rmin=args.rmin)
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


def run_path(base_dir, run_idx):
    return os.path.join(base_dir, f"run_{run_idx:03d}")


def child_args(args, run_idx, configure=True):
    child = copy.copy(args)
    child.run_id = None
    child.sp_dir = run_path(args.sp_dir, run_idx)
    child.log_dir = run_path(args.log_dir, run_idx)
    child.output = os.path.join(run_path(args.runs_dir, run_idx), "sm_convergence.pkl")
    child.plot = False
    if args.seed is not None:
        child.seed = args.seed + run_idx
    if configure:
        return configure_output_paths(child)
    return child


def required_labels(args):
    labels = ["boolean"]
    if args.optimal_reference == "long_run":
        labels.append("optimal")
    return labels


def label_steps(label, args):
    if label == "optimal":
        return [args.optimal_iters]
    return parse_maxiters(args.maxiters)


def train_label(args):
    args = apply_debug_defaults(args)
    args = resolve_rmin(args)
    if args.run is None:
        raise ValueError("Training requires --run. Use Slurm arrays over --run and --train-label.")
    if args.train_label is None:
        raise ValueError("Training requires --train-label boolean or optimal.")

    train_args = child_args(args, args.run)
    wandb_run = init_wandb(args)
    print(f"Training run_{args.run:03d}/{args.train_label}")
    try:
        steps = label_steps(args.train_label, train_args)
        train_primitives_schedule(steps, train_args, label=args.train_label, wandb_run=wandb_run)
    finally:
        if wandb_run:
            wandb_run.finish()


def complete_run(run_idx, args):
    eval_args = child_args(args, run_idx)
    try:
        for label in required_labels(args):
            for total_steps in label_steps(label, eval_args):
                sp_dir = primitive_run_dir(
                    total_steps,
                    eval_args,
                    label=label,
                    env_id=checkpoint_env(eval_args),
                )
                require_checkpoint(sp_dir, eval_args)
    except FileNotFoundError:
        return False
    return True


def completed_runs(args):
    return [run_idx for run_idx in range(args.runs) if complete_run(run_idx, args)]


def eval_single_run(args, run_idx):
    eval_args = child_args(args, run_idx, configure=False)
    eval_args.eval_only = True
    eval_args.overwrite = True
    eval_args.plot = False
    os.makedirs(os.path.dirname(eval_args.output), exist_ok=True)
    print(f"Evaluating run_{run_idx:03d}")
    return run_once(eval_args)


def aggregate_metric_dict(metric_dicts):
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    aggregate = {}
    for key in keys:
        if key.endswith("_times"):
            samples = []
            for metrics in metric_dicts:
                samples.extend(metrics.get(key, []))
            if samples:
                aggregate[key] = samples
            continue
        if key.endswith("_std") and key[:-4] in keys:
            continue
        values = np.asarray(
            [metrics[key] for metrics in metric_dicts if key in metrics],
            dtype=float,
        )
        if not len(values):
            continue
        aggregate[key] = float(np.mean(values))
        std_key = f"{key}_std"
        if std_key in keys:
            aggregate[std_key] = float(np.std(values))
    return aggregate


def aggregate_results(run_results):
    aggregate = {}
    top_keys = sorted(
        {key for result in run_results for key in result},
        key=lambda item: (not isinstance(item, str), item),
    )
    for key in top_keys:
        if key == "optimal":
            aggregate[key] = {}
            task_names = sorted({task for result in run_results for task in result.get(key, {})})
            for task_name in task_names:
                aggregate[key][task_name] = {}
                for method in METHODS:
                    metrics = [
                        result[key][task_name][method]
                        for result in run_results
                        if task_name in result.get(key, {}) and method in result[key][task_name]
                    ]
                    if metrics:
                        aggregate[key][task_name][method] = aggregate_metric_dict(metrics)
            continue

        if not isinstance(key, int):
            continue
        aggregate[key] = {}
        task_names = sorted({task for result in run_results for task in result.get(key, {})})
        for task_name in task_names:
            aggregate[key][task_name] = {}
            for method in METHODS:
                metrics = [
                    result[key][task_name][method]
                    for result in run_results
                    if task_name in result.get(key, {}) and method in result[key][task_name]
                ]
                if metrics:
                    aggregate[key][task_name][method] = aggregate_metric_dict(metrics)
    return aggregate


def plot_results(results, output):
    with plt.rc_context(PLOT_RC):
        _set_style()
        output_dir = os.path.dirname(output) or "."
        figure_dir = os.path.join(output_dir, "figures_new")
        os.makedirs(figure_dir, exist_ok=True)

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
            ("returns", "return", "Episode return"),
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
            for ax, task_name in zip(axes, task_names):
                for method in METHODS:
                    values = [
                        results[maxiter].get(task_name, {}).get(method, {}).get(key, np.nan)
                        for maxiter in maxiters
                    ]
                    std_key = f"{key}_std"
                    stds = [
                        results[maxiter].get(task_name, {}).get(method, {}).get(std_key, 0.0)
                        for maxiter in maxiters
                    ]
                    values = np.asarray(values, dtype=float)
                    stds = np.asarray(stds, dtype=float)
                    if not np.isfinite(values).any():
                        continue
                    lower = values - stds
                    upper = values + stds
                    if key == "success_rate":
                        lower = np.clip(lower, 0.0, 1.0)
                        upper = np.clip(upper, 0.0, 1.0)
                    ax.plot(
                        maxiters,
                        values,
                        CONVERGENCE_MARKER,
                        color=COLORS[method],
                        linewidth=2,
                        markersize=CONVERGENCE_MARKER_SIZE,
                        label=METHOD_LABELS[method],
                    )
                    ax.fill_between(maxiters, lower, upper, color=COLORS[method], alpha=0.2, linewidth=0)
                ax.set_title(TASK_LABELS.get(task_name, task_name), fontsize=TITLE_FONTSIZE)
                _format_axis(ax)
            for ax in axes[len(task_names):]:
                ax.axis("off")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=LEGEND_FONTSIZE)
            fig.supxlabel("Training iterations per extended value function", fontsize=LABEL_FONTSIZE)
            fig.supylabel(ylabel, fontsize=LABEL_FONTSIZE)
            fig.tight_layout(rect=(0.02, 0.02, 1, 0.92))
            fig.savefig(os.path.join(figure_dir, f"{filename}.png"), bbox_inches="tight", dpi=200)
            fig.savefig(os.path.join(figure_dir, f"{filename}.pdf"), bbox_inches="tight")
            plt.close(fig)

            if key in ("return", "success_rate"):
                fig, ax = plt.subplots(figsize=(6, 4.5))
                for method in METHODS:
                    means = []
                    spreads = []
                    for maxiter in maxiters:
                        task_values = [
                            results[maxiter].get(task_name, {}).get(method, {}).get(key, np.nan)
                            for task_name in task_names
                        ]
                        task_values = np.asarray(task_values, dtype=float)
                        task_values = task_values[np.isfinite(task_values)]
                        if len(task_values) == 0:
                            means.append(np.nan)
                            spreads.append(0.0)
                        else:
                            means.append(float(np.mean(task_values)))
                            spreads.append(float(np.std(task_values)))

                    means = np.asarray(means, dtype=float)
                    spreads = np.asarray(spreads, dtype=float)
                    lower = means - spreads
                    upper = means + spreads
                    if key == "success_rate":
                        lower = np.clip(lower, 0.0, 1.0)
                        upper = np.clip(upper, 0.0, 1.0)
                    ax.plot(
                        maxiters,
                        means,
                        CONVERGENCE_MARKER,
                        color=COLORS[method],
                        linewidth=2,
                        markersize=CONVERGENCE_MARKER_SIZE,
                        label=METHOD_LABELS[method],
                    )
                    ax.fill_between(maxiters, lower, upper, color=COLORS[method], alpha=0.2, linewidth=0)

                _format_axis(ax, with_xlabel=True, ylabel=ylabel)
                ax.legend(fontsize=LEGEND_FONTSIZE)
                fig.tight_layout()
                fig.savefig(os.path.join(figure_dir, f"average_{filename}.png"), bbox_inches="tight", dpi=200)
                fig.savefig(os.path.join(figure_dir, f"average_{filename}.pdf"), bbox_inches="tight")
                plt.close(fig)
        plot_average_returns_successes(results, maxiters, task_names, figure_dir)
        plot_composition_time_violins(results, maxiters, task_names, figure_dir)
        print(f"Plots saved to {figure_dir}")


def average_metric_by_iter(results, maxiters, task_names, method, key):
    means = []
    for maxiter in maxiters:
        values = [
            results[maxiter].get(task_name, {}).get(method, {}).get(key, np.nan)
            for task_name in task_names
        ]
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        means.append(np.nan if len(values) == 0 else float(np.mean(values)))
    return np.asarray(means, dtype=float)


def plot_average_returns_successes(results, maxiters, task_names, figure_dir):
    has_returns = any(
        np.isfinite(results[maxiter][task_name][method].get("return", np.nan))
        for maxiter in maxiters
        for task_name in task_names
        for method in METHODS
        if task_name in results[maxiter] and method in results[maxiter][task_name]
    )
    has_successes = any(
        np.isfinite(results[maxiter][task_name][method].get("success_rate", np.nan))
        for maxiter in maxiters
        for task_name in task_names
        for method in METHODS
        if task_name in results[maxiter] and method in results[maxiter][task_name]
    )
    if not has_returns or not has_successes:
        return

    fig, ax_return = plt.subplots(figsize=(6, 4.5))
    ax_success = ax_return.twinx()
    return_handles = []
    success_handles = []
    for method in METHODS:
        returns = average_metric_by_iter(results, maxiters, task_names, method, "return")
        successes = average_metric_by_iter(results, maxiters, task_names, method, "success_rate")
        if np.isfinite(returns).any():
            (line,) = ax_return.plot(
                maxiters,
                returns,
                marker="^",
                linestyle="-",
                color=COLORS[method],
                linewidth=2,
                markersize=CONVERGENCE_MARKER_SIZE,
                label=f"{METHOD_LABELS[method]} return",
            )
            return_handles.append(line)
        if np.isfinite(successes).any():
            (line,) = ax_success.plot(
                maxiters,
                successes,
                marker="^",
                linestyle="--",
                color=COLORS[method],
                linewidth=2,
                markersize=CONVERGENCE_MARKER_SIZE,
                label=f"{METHOD_LABELS[method]} success",
            )
            success_handles.append(line)

    _format_axis(ax_return, with_xlabel=True, ylabel="Episode return")
    ax_success.set_ylabel("Success rate", fontsize=LABEL_FONTSIZE)
    ax_success.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_success.set_ylim(0.0, 1.0)
    handles = return_handles + success_handles
    ax_return.legend(handles=handles, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "average_returns_successes.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(figure_dir, "average_returns_successes.pdf"), bbox_inches="tight")
    plt.close(fig)


def print_composition_time_stats(rows):
    if not rows:
        return

    headers = ("Task", "Method", "N", "Mean", "Std", "Median", "Q1", "Q3", "Min", "Max")
    table = [headers]
    for task_name, method, values in rows:
        table.append(
            (
                TASK_LABELS.get(task_name, task_name),
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


def plot_composition_time_violins(results, maxiters, task_names, figure_dir):
    has_values = any(
        results[maxiter][task_name][method].get("composition_times")
        for maxiter in maxiters
        for task_name in task_names
        for method in COMPOSITION_TIME_METHODS
        if task_name in results[maxiter] and method in results[maxiter][task_name]
    )
    if not has_values:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    base_positions = np.arange(len(task_names))
    offsets = np.linspace(-0.25, 0.25, len(COMPOSITION_TIME_METHODS))
    stats_rows = []

    for offset, method in zip(offsets, COMPOSITION_TIME_METHODS):
        samples = []
        positions = []
        for idx, task_name in enumerate(task_names):
            values_by_task = []
            for maxiter in maxiters:
                if task_name not in results[maxiter] or method not in results[maxiter][task_name]:
                    continue
                values = np.asarray(
                    results[maxiter][task_name][method].get("composition_times", []),
                    dtype=float,
                )
                values = values[np.isfinite(values)] * 1000.0
                if len(values):
                    values_by_task.append(values)
            if values_by_task:
                task_values = np.concatenate(values_by_task)
                samples.append(task_values)
                positions.append(base_positions[idx] + offset)
                stats_rows.append((task_name, method, task_values))
        if not samples:
            continue
        violins = ax.violinplot(
            samples,
            positions=positions,
            widths=0.22,
            showmeans=True,
            showextrema=False,
        )
        for body in violins["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor(COLORS[method])
            body.set_alpha(0.35)
        violins["cmeans"].set_color(COLORS[method])
        violins["cmeans"].set_linewidth(2)

    print_composition_time_stats(stats_rows)

    handles = [
        Patch(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=0.35, label=METHOD_LABELS[method])
        for method in COMPOSITION_TIME_METHODS
    ]
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([TASK_LABELS.get(task_name, task_name) for task_name in task_names], rotation=20, ha="right")
    ax.set_ylabel("Composition time per action (ms)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "composition_time_violins.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(figure_dir, "composition_time_violins.pdf"), bbox_inches="tight")
    plt.close(fig)


def filter_results_maxiters(results, requested_maxiters):
    maxiters = [maxiter for maxiter in requested_maxiters if maxiter in results]
    missing = [maxiter for maxiter in requested_maxiters if maxiter not in results]
    if missing:
        print("Skipping maxiters not found in results: " + ", ".join(str(maxiter) for maxiter in missing))
    if not maxiters:
        raise ValueError(
            "None of the requested --maxiters are present in results. "
            "Available maxiters: "
            + ", ".join(str(maxiter) for maxiter in sorted(key for key in results if isinstance(key, int)))
        )

    filtered = {key: value for key, value in results.items() if not isinstance(key, int)}
    filtered.update({maxiter: results[maxiter] for maxiter in maxiters})
    return filtered


def load_available_run_results(args):
    run_results = []
    run_paths = []
    for run_idx in range(args.runs):
        run_output = os.path.join(run_path(args.runs_dir, run_idx), "sm_convergence.pkl")
        if os.path.exists(run_output):
            with open(run_output, "rb") as f:
                run_results.append(pickle.load(f))
            run_paths.append(run_output)
    return run_results, run_paths


def _set_style():
    rc("text", usetex=True)
    sns.set_context("notebook", font_scale=0.8)


def _format_axis(ax, with_xlabel=False, ylabel=None):
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if with_xlabel:
        ax.set_xlabel("Training iterations per extended value function", fontsize=LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xscale("log")


def eval_all_runs(args):
    args = apply_debug_defaults(args)
    args = resolve_rmin(args)
    if args.run is not None:
        if not complete_run(args.run, args):
            raise RuntimeError(f"run_{args.run:03d} is not complete and cannot be evaluated.")
        return eval_single_run(args, args.run)

    done = completed_runs(args)
    if not done:
        raise RuntimeError("No complete new convergence runs found to evaluate.")
    print("Evaluating runs:", ", ".join(f"run_{idx:03d}" for idx in done))
    wandb_run = init_wandb(args)
    run_results = [eval_single_run(args, run_idx) for run_idx in done]
    results = aggregate_results(run_results)
    if args.optimal_reference == "max_observed":
        apply_max_observed_optimal(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {args.output}")
    log_eval_results(wandb_run, results)
    if wandb_run:
        wandb_run.save(args.output)
        wandb_run.finish()
    if args.plot:
        plot_results(results, args.output)
    return results


def make_plots(args):
    requested_maxiters = parse_maxiters(args.maxiters)
    run_results, run_paths = load_available_run_results(args)
    if run_results:
        results = aggregate_results(run_results)
        print(f"Plotting aggregate from {len(run_results)} run files:")
        for run_output in run_paths:
            print(f"  {run_output}")
        results = filter_results_maxiters(results, requested_maxiters)
        plot_results(results, args.output)
        return

    if not os.path.exists(args.output):
        raise FileNotFoundError(args.output)

    with open(args.output, "rb") as f:
        results = pickle.load(f)
    print(f"No per-run files found under {args.runs_dir}; plotting {args.output}")
    results = filter_results_maxiters(results, requested_maxiters)
    plot_results(results, args.output)


def run(args):
    if args.plot_only and args.eval_only:
        raise ValueError("--plot_only cannot be used with --eval_only")
    if args.make_plots or args.plot_only:
        return make_plots(args)
    if args.eval_only:
        return eval_all_runs(args)
    return train_label(args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.set_defaults(overwrite=True)
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
    parser.add_argument("--rmin", type=float, default=None)
    parser.add_argument("--run_id", default=None)
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
    parser.add_argument("--sp_dir", default=DEFAULT_SP_DIR)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--eval_only", action="store_true", help="Skip primitive training and evaluate existing shared checkpoints")
    parser.add_argument("--plot", action="store_true", default=True, help="Save convergence plots")
    parser.add_argument("--no_plot", action="store_false", dest="plot", help="Disable plot generation")
    parser.add_argument("--debug", action="store_true", help="Use low training/evaluation steps with the same experiment setup")
    parser.add_argument("--debug_maxiters", default="1000", help="Training steps used when --debug is set")
    parser.add_argument("--debug_optimal_iters", type=int, default=2000, help="Optimal-reference training steps used when --debug is set")
    parser.add_argument("--debug_eval_steps", type=int, default=20, help="Evaluation horizon used when --debug is set")
    parser.add_argument("--debug_num_runs", type=int, default=2, help="Evaluation episodes used when --debug is set")
    parser.add_argument("--debug_value_episodes", type=int, default=2, help="Value-error episodes used when --debug is set")
    parser.add_argument("--debug_print_freq", type=int, default=100, help="Training print frequency used when --debug is set")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Maximum number of independent value-function runs")
    parser.add_argument("--run", type=int, default=None, help="Single run index to train, usually from a Slurm array")
    parser.add_argument("--runs_dir", default=DEFAULT_RUNS_DIR, help="Per-run evaluation cache directory")
    parser.add_argument("--train-label", choices=RUN_LABELS, default=None, help="Checkpoint label to train for this run")
    parser.add_argument("--eval-only", action="store_true", dest="eval_only", help="Alias for --eval_only")
    parser.add_argument("--make-plots", action="store_true", help="Only load --output and write figures_new plots")
    parser.add_argument("--plot_only", action="store_true", help="Only load --output and write figures_new plots")
    parser.add_argument("--plot-only", action="store_true", dest="plot_only", help="Alias for --plot_only")
    parser.add_argument("--only_plot", action="store_true", dest="plot_only", help="Alias for --plot_only")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="skill-machines-convergence")
    parser.add_argument("--wandb-name", default=None)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.plot_only and args.eval_only:
        parser.error("--plot_only cannot be used with --eval_only")
    run(args)
