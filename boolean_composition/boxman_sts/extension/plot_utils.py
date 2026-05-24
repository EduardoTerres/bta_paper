import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


METHODS = ["onoff", "boolean"]


def plot_convergence(returns_per_steps, figure_path):
    steps = sorted(returns_per_steps)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        means, stds = _mean_std_by_step(returns_per_steps, steps, method)
        ax.plot(steps, means, marker="o", label=method)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
    _format_axis(ax)
    ax.legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_path), bbox_inches="tight")
    plt.close(fig)


def plot_convergence_by_task(returns_per_steps, figure_path):
    steps = sorted(returns_per_steps)
    task_names = list(next(iter(returns_per_steps.values())).keys())
    ncols = 3
    nrows = math.ceil(len(task_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, task_name in zip(axes, task_names):
        for method in METHODS:
            means, stds = _mean_std_by_step(
                returns_per_steps,
                steps,
                method,
                task_name=task_name,
            )
            ax.plot(steps, means, marker="o", label=method)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
        ax.set_title(task_name)
        _format_axis(ax)

    for ax in axes[len(task_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_path), bbox_inches="tight")
    plt.close(fig)


def by_task_figure_path(figure_path):
    return figure_path.with_name(f"{figure_path.stem}_by_task{figure_path.suffix}")


def _mean_std_by_step(returns_per_steps, steps, method, task_name=None):
    means = []
    stds = []
    for step in steps:
        values = []
        if task_name is None:
            for task_returns in returns_per_steps[step].values():
                values.extend(task_returns[method])
        else:
            values.extend(returns_per_steps[step][task_name][method])
        means.append(np.mean(values))
        stds.append(np.std(values))
    return np.array(means), np.array(stds)


def _format_axis(ax):
    ax.set_xscale("log")
    ax.set_xlabel("DQN training timesteps")
    ax.set_ylabel("Average return")
    ax.grid(True, alpha=0.3)
