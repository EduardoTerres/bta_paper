"""Convergence experiment for original and minmax Skill Machine composition."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse, pickle, numpy as np, gymnasium as gym
import matplotlib.pyplot as plt
import envs  # registers Office-v0 and task variants
from sm import TaskPrimitive, SkillMachine, MinMaxSkillMachine, evaluate
from rm import Task
import sm_ql


# ─── Experiment configuration ─────────────────────────────────────────────────

ENV_NAME = "Office-v0"
TASKS = {
    "Coffee":
        "(F (c & X (F o))) & (G~d)",
    "Patrol":
        "(F (A & X (F (B & X (F (C & X (F D))))))) & (G~d)",
    "CoffeeMail":
        "((F (m & X (F (c & X (F o))))) | (F (c & X (F (m & X (F o)))))) & (G~d)",
    "Long":
        "(F (m & X (F (o & X (~ m U (~ tm & m & X (F (c & X (~ o U (~ to & o & X "
        "(F (A & X (F (B & X (F (C & X (F (D & X (F A))))))))))))))))))) & (G~d)",
}
MAX_ITERS     = [50_000, 100_000, 200_000]
OPTIMAL_ITERS = 300_000
N_SEEDS       = 1
EVAL_EPISODES = 50
OUT_DIR = os.path.join(os.path.dirname(__file__), "exps_data_extension")
METHODS = {
    "original": SkillMachine,
    "minmax": MinMaxSkillMachine,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _train(total_steps, seed):
    """Fresh primitive training run; returns (primitive_env, SP)."""
    primitive_env = TaskPrimitive(gym.make(ENV_NAME))
    log_dir = f"/tmp/sm_conv_logs/seed{seed}_iter{total_steps}/"
    SP = sm_ql.learn(primitive_env, None, total_steps,
                     log_dir=log_dir, print_freq=10**9, seed=seed)
    return primitive_env, SP


def _eval(primitive_env, SP, ltl, method, seed=None):
    """Evaluate one SM variant on a task; returns success rate in [0, 1]."""
    task_env = Task(gym.make(ENV_NAME), ltl)
    SM = METHODS[method](primitive_env, SP, goal_directed=True)
    _, success_rate, _ = evaluate(task_env, SM, epsilon=0, gamma=1,
                                  episodes=EVAL_EPISODES, seed=seed)
    return success_rate


# ─── Main experiment ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Composition methods to evaluate.",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_iters = MAX_ITERS + [OPTIMAL_ITERS]
    results = {
        method: {t: {it: [] for it in all_iters} for t in TASKS}
        for method in args.methods
    }

    for seed in range(N_SEEDS):
        print(f"\n=== Seed {seed + 1}/{N_SEEDS} ===")
        for iters in all_iters:
            print(f"  Training {iters:,} steps...")
            primitive_env, SP = _train(iters, seed)
            for task_name, ltl in TASKS.items():
                parts = []
                for method in args.methods:
                    score = _eval(primitive_env, SP, ltl, method, seed)
                    results[method][task_name][iters].append(score)
                    parts.append(f"{method}={score:.3f}")
                print(f"    {task_name}: " + "  ".join(parts))

    # Save results
    with open(os.path.join(OUT_DIR, "sm_convergence.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("Results saved to", OUT_DIR)

    # ─── Plotting ─────────────────────────────────────────────────────────────
    COLORS = {"original": "steelblue", "minmax": "tomato"}

    def _plot_curve(ax, task, data, method, color):
        vals  = [data[task][it] for it in MAX_ITERS]
        means = [np.mean(v) for v in vals]
        stds  = [np.std(v)  for v in vals]
        opt   = data[task][OPTIMAL_ITERS]
        ax.plot(MAX_ITERS, means, color=color, label=method)
        ax.fill_between(MAX_ITERS, np.subtract(means, stds), np.add(means, stds),
                        color=color, alpha=0.2)
        for fn, ls in [(np.mean, "--"), (np.min, ":"), (np.max, ":")]:
            ax.axhline(fn(opt), color=color, linestyle=ls, linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Training steps"); ax.set_ylabel("Success rate")
        ax.set_title(task)

    three_tasks = ["Coffee", "Patrol", "CoffeeMail"]

    # Two combined 3-subplot figures (one per method)
    for method, data in results.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, task in zip(axes, three_tasks):
            _plot_curve(ax, task, data, method, COLORS[method])
        fig.suptitle(f"Convergence - {method}")
        plt.tight_layout()
        fname = f"fig_{method}_3tasks.pdf"
        fig.savefig(os.path.join(OUT_DIR, fname)); plt.close(fig)

    # Individual task figures with both methods overlaid
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, data in results.items():
            _plot_curve(ax, task, data, method, COLORS[method])
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"fig_{task}.pdf")); plt.close(fig)

    print("Figures saved to", OUT_DIR)
