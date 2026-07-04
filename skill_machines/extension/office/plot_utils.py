import math
import os

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
METHOD_LABELS = {
    "original": r"Original SM",
    "minmax": r"Univ./empty (Ours)",
    "boolean": r"Base tasks",
}
COMPOSITION_TIME_METHODS = METHODS
COLORS = {
    "original": "#C0560A",
    "minmax": "#1A5276",
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
        plot_composition_time_boxplots(results, maxiters, task_names, figure_dir)
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

    fig, ax_return = plt.subplots(figsize=(10.8, 4.5))
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
                marker="o",
                linestyle="--",
                color=COLORS[method],
                linewidth=2,
                markersize=CONVERGENCE_MARKER_SIZE,
                label=f"{METHOD_LABELS[method]} success",
            )
            success_handles.append(line)

    _format_axis(ax_return, with_xlabel=False, ylabel="Episode return")
    ax_success.set_ylabel("Success rate", fontsize=LABEL_FONTSIZE)
    ax_success.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_success.set_ylim(0.0, 1.0)
    handles = return_handles + success_handles
    ax_return.legend(
        handles=handles,
        fontsize=LEGEND_FONTSIZE,
        loc="center left",
        bbox_to_anchor=(1.18, 0.5),
        borderaxespad=0.0,
    )
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
                values = values[values > 0]
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
    ax.set_yscale("log")
    ax.set_ylabel("Composition time per action (ms)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "composition_time_violins.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(figure_dir, "composition_time_violins.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_composition_time_boxplots(results, maxiters, task_names, figure_dir):
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
                values = values[values > 0]
                if len(values):
                    values_by_task.append(values)
            if values_by_task:
                samples.append(np.concatenate(values_by_task))
                positions.append(base_positions[idx] + offset)
        if not samples:
            continue
        box = ax.boxplot(
            samples,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_edgecolor(COLORS[method])
            patch.set_alpha(0.35)
        for key in ("whiskers", "caps", "medians"):
            for line in box[key]:
                line.set_color(COLORS[method])
                line.set_linewidth(2 if key == "medians" else 1.5)

    handles = [
        Patch(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=0.35, label=METHOD_LABELS[method])
        for method in COMPOSITION_TIME_METHODS
    ]
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([TASK_LABELS.get(task_name, task_name) for task_name in task_names], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Composition time per action (ms)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "composition_time_boxplots.png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(figure_dir, "composition_time_boxplots.pdf"), bbox_inches="tight")
    plt.close(fig)


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
