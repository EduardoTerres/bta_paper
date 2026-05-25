import math

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rc


METHODS = ["onoff", "boolean"]
METHOD_LABELS = [r"Univ./Empty (Ours)", r"Base Tasks"]
COLORS = plt.cm.tab10.colors[:2]


def plot_convergence(returns_per_steps, figure_path):
    _set_style()
    steps = sorted(returns_per_steps)
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, method in enumerate(METHODS):
        means, stds = _mean_std_by_step(returns_per_steps, steps, method)
        ax.plot(steps, means, "o-", color=COLORS[idx], linewidth=2, markersize=6, label=method)
        ax.fill_between(steps, means - stds, means + stds, color=COLORS[idx], alpha=0.2)
    _format_axis(ax, with_xlabel=True, with_ylabel=True)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=METHOD_LABELS, fontsize=20)
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_path), bbox_inches="tight")
    plt.close(fig)


def plot_convergence_by_task(returns_per_steps, figure_path):
    _set_style()
    steps = sorted(returns_per_steps)
    task_names = list(next(iter(returns_per_steps.values())).keys())
    ncols = 3
    nrows = math.ceil(len(task_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, task_name in zip(axes, task_names):
        for idx, method in enumerate(METHODS):
            means, stds = _mean_std_by_step(
                returns_per_steps,
                steps,
                method,
                task_name=task_name,
            )
            ax.plot(steps, means, "o-", color=COLORS[idx], linewidth=2, markersize=6, label=method)
            ax.fill_between(steps, means - stds, means + stds, color=COLORS[idx], alpha=0.2)
        ax.set_title(task_name, fontsize=20)
        _format_axis(ax)

    for ax in axes[len(task_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, METHOD_LABELS, loc="upper center", ncol=len(METHODS), fontsize=20)
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


def _set_style():
    rc("text", usetex=True)
    sns.set_context("notebook", font_scale=0.8)


def _format_axis(ax, with_xlabel=False, with_ylabel=False):
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=18)
    if with_xlabel:
        ax.set_xlabel("DQN training timesteps", fontsize=20)
    if with_ylabel:
        ax.set_ylabel("Returns", fontsize=20)
