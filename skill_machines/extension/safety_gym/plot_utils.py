import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
from matplotlib.patches import Patch

PLOT_RC = {
    "text.usetex": True,
    "text.latex.preamble": "",
}
plt.rcParams.update(PLOT_RC)

METHODS = ("original", "minmax", "boolean")
BASE_TASK_METHODS = ("0", "1", "p_buttons", "p_goal", "p_hazards", "c_buttons", "c_goal", "c_hazards")
METHOD_LABELS = {
    "original": r"Original SM",
    "minmax": r"Univ./empty (Ours)",
    "boolean": r"Base tasks BTA",
    "0": r"Base task $0$",
    "1": r"Base task $1$",
    "p_buttons": r"Base task buttons",
    "p_goal": r"Base task goal",
    "p_hazards": r"Base task hazards",
    "c_buttons": r"Base constraint buttons",
    "c_goal": r"Base constraint goal",
    "c_hazards": r"Base constraint hazards",
}
COMPOSITION_TIME_METHODS = METHODS + BASE_TASK_METHODS
COLORS = {
    "original": "#C0560A",
    "minmax": "#1A5276",
    "boolean": "#2E7D32",
    "0": "#7F8C8D",
    "1": "#34495E",
    "p_buttons": "#8E44AD",
    "p_goal": "#D35400",
    "p_hazards": "#C0392B",
    "c_buttons": "#9B59B6",
    "c_goal": "#E67E22",
    "c_hazards": "#E74C3C",
}
TASK_LABELS = {
    "task1": "Task 1",
    "task2": "Task 2",
    "task3": "Task 3",
    "task4": "Task 4",
    "task5": "Task 5",
    "task6": "Task 6",
}
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 16
CONVERGENCE_MARKER = "^-"
CONVERGENCE_MARKER_SIZE = 8


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


def plot_results(results, output, figures_dir=None):
    with plt.rc_context(PLOT_RC):
        _set_style()
        out_dir = Path(figures_dir) if figures_dir else Path(output).parent / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = sorted(key for key in results if isinstance(key, int))
        task_names = sorted({task for step in steps for task in results[step]})

        plot_metric_by_task(results, steps, task_names, "return", "Episode return", out_dir / "returns.png")
        plot_metric_by_task(results, steps, task_names, "success_rate", "Success rate", out_dir / "successes.png")
        plot_metric_average(results, steps, task_names, "return", "Episode return", out_dir / "average_returns.png")
        plot_metric_average(results, steps, task_names, "success_rate", "Success rate", out_dir / "average_successes.png")

        plot_metric_by_task(results, steps, task_names, "return", "Episode return", out_dir / "returns.pdf")
        plot_metric_by_task(results, steps, task_names, "success_rate", "Success rate", out_dir / "successes.pdf")
        plot_metric_average(results, steps, task_names, "return", "Episode return", out_dir / "average_returns.pdf")
        plot_metric_average(results, steps, task_names, "success_rate", "Success rate", out_dir / "average_successes.pdf")
        plot_composition_time_violins(results, steps, task_names, out_dir)
        print(f"Plots saved to {out_dir}")


def plot_metric_by_task(results, steps, task_names, metric, ylabel, path):
    ncols = min(3, len(task_names))
    nrows = math.ceil(len(task_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False, sharex=True)
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
                CONVERGENCE_MARKER,
                color=COLORS[method],
                linewidth=2,
                markersize=CONVERGENCE_MARKER_SIZE,
                label=METHOD_LABELS[method],
            )
            ax.fill_between(steps, lower, upper, color=COLORS[method], alpha=0.2, linewidth=0)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=TITLE_FONTSIZE)
        _format_axis(ax)
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
    fig, ax = plt.subplots(figsize=(6, 4.5))
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
            CONVERGENCE_MARKER,
            color=COLORS[method],
            linewidth=2,
            markersize=CONVERGENCE_MARKER_SIZE,
            label=METHOD_LABELS[method],
        )
        ax.fill_between(steps, lower, upper, color=COLORS[method], alpha=0.2, linewidth=0)
    _format_axis(ax, with_xlabel=True, ylabel=ylabel)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_composition_time_stats(rows):
    if not rows:
        return

    headers = ("Task", "Method", "N", "Mean", "Std", "Median", "Q1", "Q3", "Min", "Max")
    table = [headers]
    for task, method, values in rows:
        table.append(
            (
                TASK_LABELS.get(task, task),
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

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    base_positions = np.arange(len(task_names))
    offsets = np.linspace(-0.25, 0.25, len(COMPOSITION_TIME_METHODS))
    stats_rows = []
    plotted_methods = []

    for offset, method in zip(offsets, COMPOSITION_TIME_METHODS):
        samples = []
        positions = []
        for idx, task in enumerate(task_names):
            values_by_task = []
            for step in steps:
                if task not in results[step] or method not in results[step][task]:
                    continue
                values = np.asarray(
                    results[step][task][method].get("composition_times", []),
                    dtype=float,
                )
                values = values[np.isfinite(values)] * 1000.0
                values = values[values > 0]
                if len(values):
                    values_by_task.append(values)
            if values_by_task:
                task_values = np.concatenate(values_by_task)
                samples.append(task_values)
                positions.append(base_positions[idx] + offset)
                stats_rows.append((task, method, task_values))
        if not samples:
            continue
        plotted_methods.append(method)
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
        for method in plotted_methods
    ]
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([TASK_LABELS.get(task, task) for task in task_names], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Composition time per action (ms)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "composition_time_violins.png", bbox_inches="tight", dpi=200)
    fig.savefig(out_dir / "composition_time_violins.pdf", bbox_inches="tight")
    plt.close(fig)
