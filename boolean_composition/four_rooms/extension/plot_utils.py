import os
from collections import defaultdict
from four_rooms.GridWorld import GridWorld
from four_rooms.library import (
    EQ_P,
    EQ_V,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import rc
from scipy.optimize import curve_fit

def plot_composed_EQs(composed_EQs, goals, terminal_states, num_rooms):
    """Plot composed EQs for all tasks individually."""
    tasks = list(composed_EQs.keys())
    
    for row_idx, task in enumerate(tasks):
        env = GridWorld(
            MAP="MAP_" + str(num_rooms),
            goals=goals,
            T_states=terminal_states,
        )
        
        # Render onoff EQ and move axes to main figure
        on_off_fig = env.render(P=EQ_P(composed_EQs[task]["onoff"]), V=EQ_V(composed_EQs[task]["onoff"]))
        boolean_fig = env.render(P=EQ_P(composed_EQs[task]["boolean"]), V=EQ_V(composed_EQs[task]["boolean"]))

        # Save the figs to pngs
        on_off_fig.savefig(f"four_rooms/extension/figures_comparison/on_off_task_{row_idx + 1}_rooms_{num_rooms}.png")
        boolean_fig.savefig(f"four_rooms/extension/figures_comparison/boolean_task_{row_idx + 1}_rooms_{num_rooms}.png")

        # Close the figs
        plt.close(on_off_fig)
        plt.close(boolean_fig)


def plot_returns(returns: dict[tuple[int, int], dict[str, list[float]]], save_name: str = None):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side."""
    tasks = ["\n".join(str(g) for g in task) for task in returns.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, returns.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(16, 6))
    sns.set_context("notebook", font_scale=0.8)
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)
    ax.set_xlabel("Task", fontsize=20)
    ax.set_ylabel("Returns", fontsize=20)
    ax.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)


def plot_returns_all_num_goals(
    returns: dict[int, dict[tuple[int, int], dict[str, list[float]]]],
    save_name: str = None,
):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side.
    
    Args:
        returns: Dictionary mapping num_rooms to tasks to their returns
        save_name: Path to save the figure
    """
    rc("text", usetex=True)
    # Number of tasks to show for each number of goals
    # The keys of the outer dictionary are the number of rooms
    # The keys of the inner dictionary are the number of goals
    # The values are the number of tasks to show for each number of goals
    shown_tasks = {
        4: [1, 2, 3],
        8: [1, 3, 5, 7],
        16: [1, 5, 9, 12, 15],
    }

    flattened_returns = {}
    # Proportional sample of the tasks for each number of rooms
    for num_rooms, task_returns_dict in returns.items():
        tasks_by_length = defaultdict(list)
        for task in task_returns_dict.keys():
            tasks_by_length[len(task)].append(task)
        
        for length in shown_tasks[num_rooms]:
            tasks_of_length_returns = {
                "onoff": [],
                "boolean": [],
                "optimal": [],
            }
            for task in tasks_by_length[length]:
                tasks_of_length_returns["onoff"].extend(returns[num_rooms][task]["onoff"])
                tasks_of_length_returns["boolean"].extend(returns[num_rooms][task]["boolean"])
                tasks_of_length_returns["optimal"].extend(returns[num_rooms][task]["optimal"])

            flattened_returns[(num_rooms, length)] = tasks_of_length_returns

    print(
        "Sampled tasks:",
        ", ".join(
            f"{len(returns[num_rooms].keys())} tasks for {num_rooms} rooms"
            for num_rooms in [4, 8, 16]
        ),
    )
    
    tasks = [f"{task[1]}-{task[0]}" for task in flattened_returns.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, flattened_returns.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(20, 10))
    sns.set_context("notebook", font_scale=0.8)
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)

    # Add vertical lines to separate rooms
    tasks_per_room = [len(shown_tasks[r]) for r in [4, 8, 16]]
    pos = 0
    for n in tasks_per_room[:-1]:
        pos += n
        ax.axvline(pos - 0.5, color='black', linestyle='-', linewidth=1)
    
    # Add text labels below each room block
    pos = 0
    for r, n in zip([4, 8, 16], tasks_per_room):
        ax.text(pos + (n - 1) / 2, 1.04, f"{r} total goals", ha='center', transform=ax.get_xaxis_transform(), fontsize=28)
        pos += n
    
    ax.set_xlabel("Number of goals in task", fontsize=28)
    ax.set_ylabel("Returns", fontsize=28)
    # Restore legend colors by explicitly passing the handles
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[r'Univ./Empty (Ours)', r'Base Tasks', r'Optimal'], fontsize=28)

    # Change labels
    ax.set_xticklabels([tick.get_text().split('-')[0] for tick in ax.get_xticklabels()], fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(save_name)


def plot_time_taken(time_taken: dict[str, list[float]], num_rooms: int, save_name: str):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side."""
    tasks = ["\n".join(str(g) for g in task) for task in time_taken.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, time_taken.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(12, 12))  # Make the figure bigger
    sns.set_context("notebook", font_scale=0.8)  # Make the words smaller
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)
    # Do not plot the x ticks labels
    ax.set_ylabel("Time taken for composition", fontsize=20)
    ax.set_xlabel("All tasks", fontsize=20)
    ax.legend(fontsize=20)
    ax.set_xticklabels([])  # Remove x tick labels
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def plot_time_taken_all_num_goals(time_taken: dict[int, dict[str, list[float]]], save_name: str):
    """ Plot time taken for all number of goals (log scale on y-axis)."""
    rc("text", usetex=True)
    num_rooms = sorted(time_taken.keys())
    fig, ax = plt.subplots(figsize=(8, 6))
    positions = {r: i for i, r in enumerate(num_rooms)}
    
    colors = plt.cm.tab10.colors[:2]
    for idx, method in enumerate(["onoff", "boolean"]):
        data = [[1000 * t for t in time_taken[r][method]] for r in num_rooms]
        bp = ax.boxplot(data, positions=[positions[r] for r in num_rooms], widths=0.3, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.7)
        # Set the color of the median lines
        for median in bp['medians']:
            median.set_color(colors[idx])
        means = [sum(vals) / len(vals) for vals in data]
        ax.plot([positions[r] for r in num_rooms], means, 'o-', label=method, linewidth=2, color=colors[idx])

    ax.set_xticks(range(len(num_rooms)))
    ax.set_xticklabels(num_rooms, fontsize=20)
    ax.set_xlabel("Number of goals", fontsize=20)
    ax.set_ylabel("Time (ms)", fontsize=20)
    ax.set_yticklabels(ax.get_yticks(), fontsize=20)
    ax.set_yscale('log')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[r'Univ./Empty (Ours)', r'Base Tasks'], fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def plot_num_value_functions_learned(save_name: str):
    rc("text", usetex=True)
    x = 2 ** np.linspace(1, 20, 20)
    plt.figure(figsize=(8, 6))
    y1 = [2] * len(x)
    y2 = 2 + np.log2(x)
    
    # Special points: 2^2, 2^3, 2^4
    special_indices = [1, 2, 3]  # indices for 2^2, 2^3, 2^4
    
    # Plot full lines with transparency
    line1 = plt.plot(x, y1, alpha=0.5, label='Univ./Empty', linewidth=3)[0]
    line2 = plt.plot(x, y2, alpha=0.5, label='Base Tasks', linewidth=3)[0]
    
    # Overlay segments between special nodes with full opacity
    plt.plot(x[1:4], y1[1:4], color=line1.get_color(), alpha=1.0, linewidth=3)
    plt.plot(x[1:4], y2[1:4], color=line2.get_color(), alpha=1.0, linewidth=3)
    
    # Plot triangles
    plt.scatter(x, y1, marker='^', s=50, color=line1.get_color(), alpha=0.5)
    plt.scatter(x, y2, marker='^', s=50, color=line2.get_color(), alpha=0.5)

    plt.scatter(x[1:4], y1[1:4], marker='^', s=50, color=line1.get_color())
    plt.scatter(x[1:4], y2[1:4], marker='^', s=50, color=line2.get_color())
    
    # Add numbers next to special points
    for idx in special_indices:
        plt.annotate('$2$', (x[idx], y1[idx]), xytext=(-3, 9), textcoords='offset points', fontsize=18)
        plt.annotate(f'${{{idx+3}}}$', (x[idx], y2[idx]), xytext=(-5, 10), textcoords='offset points', fontsize=18)
    
    plt.xlabel("Number of goals", fontsize=20)
    plt.ylabel("Number of value functions learned", fontsize=20)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    tick_exponents = sorted([1, 3] + list(range(2, 21, 2)))
    tick_values = [2 ** e for e in tick_exponents]
    plt.xticks(tick_values, ['$2^' + r'{' + str(e) + r'}$' for e in tick_exponents], fontsize=20)
    plt.yticks(fontsize=20)
    # Generate legend with full opacity handles (ignore line alpha in legend)
    handles, labels = plt.gca().get_legend_handles_labels()
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=handles[0].get_color(), linewidth=3, marker='^', markersize=8, label=labels[0]),
        Line2D([0], [0], color=handles[1].get_color(), linewidth=3, marker='^', markersize=8, label=labels[1]),
    ]
    plt.legend(handles=legend_handles, labels=labels, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_learning_time(learning_time: dict[int, dict[str, float]], save_name: str):
    """Plot learning time for 4, 8, 16 goals with log(x) * C regression."""
    rc("text", usetex=True)
    x_vals = sorted(learning_time.keys())
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10.colors[:2]
    for idx, method in enumerate(["onoff", "boolean"]):
        y_vals = [learning_time[x][method] for x in x_vals]
        ax.scatter(x_vals, y_vals, color=colors[idx], s=100, label=method)

        # Fit to log(x) * x * C
        def fit_func_onoff(x, c):
            return np.log(x) * c
        
        def fit_func_boolean(x, c):
            return np.log(x) * c

        function = fit_func_onoff if method == "onoff" else fit_func_boolean

        popt, _ = curve_fit(function, x_vals, y_vals)
        x_fit = np.linspace(min(x_vals), max(x_vals), 100)
        y_fit = function(x_fit, popt[0])
        ax.plot(x_fit, y_fit, '--', color=colors[idx], linewidth=2)

    ax.set_xlabel("Number of goals", fontsize=20)
    ax.set_ylabel("Learning time", fontsize=20)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[r'Univ./Empty (Ours)', r'Base Tasks'], fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()


def plot_extended_q_value(learned_EQ, goals, terminal_states, num_rooms: int, save_name: str):
    all_goals = [str([goal, goal]) for goal in goals]

    for i, goal in enumerate(all_goals):
        env = GridWorld(
            MAP="MAP_" + str(num_rooms),
            goals=[goal],
            T_states=terminal_states,
        )
        fig = env.render(P=EQ_P(learned_EQ, goal), V=EQ_V(learned_EQ, goal), no_ticks=True)
        fig.savefig(save_name + f"_{i}.png", bbox_inches='tight', pad_inches=0.1)
    
    # Also save the whole learned_EQ
    fig = env.render(P=EQ_P(learned_EQ), V=EQ_V(learned_EQ), no_ticks=True)
    fig.tight_layout()
    fig.savefig(save_name + "_all.png", bbox_inches='tight', pad_inches=0.1)


# ------------------------------------------------------------
# Stochastic sweep: shared styling for the three sweep figures
# ------------------------------------------------------------
SWEEP_COMPOSED_METHODS = ["onoff", "boolean"]
SWEEP_OWN_METHODS = ["universal", "empty", "base_tasks"]
SWEEP_LABELS = {
    "onoff": "Goal-set composition (ours)",
    "boolean": "Original BTA composition",
    "universal": "Universal task",
    "empty": "Empty task",
    "base_tasks": "Base tasks",
    "optimal": "Optimal $V^*$",
}
SWEEP_LINESTYLES = {
    "onoff": "-", "boolean": "-",
    "universal": "--", "empty": "--", "base_tasks": "--",
    "optimal": ":",
}
SWEEP_MARKERS = {
    "onoff": "o", "boolean": "o",
    "universal": "s", "empty": "^", "base_tasks": "D",
    "optimal": "",
}
SWEEP_COLORS = dict(
    zip(SWEEP_COMPOSED_METHODS + SWEEP_OWN_METHODS, plt.cm.tab10.colors)
)
SWEEP_COLORS["optimal"] = "black"


def _save_sweep_figure(fig, save_name):
    """Write the figure as both .png and .pdf, as `boxman_sts` does.

    The PDF is the one to \\includegraphics in the paper -- vector, so the
    LaTeX-rendered text stays sharp at any size -- while the PNG stays handy
    for quick viewing. `save_name` may carry either extension.
    """
    base = os.path.splitext(save_name)[0]
    for suffix in (".png", ".pdf"):
        fig.savefig(base + suffix)


# Type sizes shared by the three sweep figures.
SWEEP_LABEL_SIZE = 24
SWEEP_TICK_SIZE = 20
SWEEP_LEGEND_SIZE = 15


def _set_sweep_style():
    """Render the sweep figures' text with LaTeX, as the Boxman plots do.

    Any literal '%' in a label must be written '\\%' once this is on -- bare
    '%' starts a comment in LaTeX and silently swallows the rest of the string.
    """
    rc("text", usetex=True)
    rc("font", family="serif")


def _plot_sweep_panel(ax, results, slip_probs, methods, ylabel, title, scale=1.0):
    """One slip-probability panel. `results` is slip_prob -> method -> values.

    Shading is +/- 1 SEM over whatever the per-slip lists hold -- per-seed
    scalars after aggregation, so the band reflects seed-to-seed variance.
    """
    for method in methods:
        if method not in results[slip_probs[0]]:
            continue
        values = [np.asarray(results[p][method], dtype=float) for p in slip_probs]
        means = np.array([scale * v.mean() for v in values])
        sems = np.array([scale * v.std() / max(len(v), 1) ** 0.5 for v in values])
        ax.plot(
            slip_probs, means,
            linestyle=SWEEP_LINESTYLES[method], marker=SWEEP_MARKERS[method],
            color=SWEEP_COLORS[method], linewidth=2, markersize=6,
            label=SWEEP_LABELS[method],
        )
        ax.fill_between(
            slip_probs, means - sems, means + sems,
            color=SWEEP_COLORS[method], alpha=0.15,
        )
    ax.set_xlabel("Slip probability $p$", fontsize=SWEEP_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=SWEEP_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=SWEEP_TICK_SIZE)
    ax.legend(fontsize=SWEEP_LEGEND_SIZE)


def plot_stochastic_sweep_gap(results: dict[float, dict[str, list[float]]], save_name: str):
    """Figure 2 of 3: the value gap V* - V^pi, in units of episode return.

    The same comparison as `plot_stochastic_sweep` without the per-state
    normalization. Both methods are measured against the same V* at each slip
    probability, so the lines can be read against each other directly, and
    since a step costs 0.1 a gap of 0.4 is "about four wasted steps' worth".
    Nothing here can blow up or flip sign, which is the point: the normalized
    view divides by |V*(s)|, and V*(s) passes through zero on tasks with few
    goals.
    """
    _set_sweep_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_sweep_panel(
        ax, results, sorted(results),
        SWEEP_COMPOSED_METHODS + SWEEP_OWN_METHODS,
        ylabel="Return",
        title=None,
    )
    plt.tight_layout()
    _save_sweep_figure(fig, save_name)
    plt.close(fig)


def plot_stochastic_sweep_returns(results: dict[float, dict[str, list[float]]], save_name: str):
    """Figure 3 of 3: mean return of actual rollout episodes.

    The Monte Carlo counterpart of the other two figures -- the composed
    policies are rolled out in the real env under slip and their returns
    averaged, rather than solved for. Agreement with the DP gap is a check
    that the closed-form dynamics match the environment.

    No optimal reference line: it cannot be `V*` (a value, not a policy that
    can be rolled out), and rolling out the stationary policy greedy w.r.t.
    `V_H` labels a slight underestimate of the optimum as "optimal". The gap
    figure already shows distance from the optimum exactly.
    """
    _set_sweep_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_sweep_panel(
        ax, results, sorted(results),
        SWEEP_COMPOSED_METHODS + SWEEP_OWN_METHODS,
        ylabel="Return",
        title=None,
    )
    plt.tight_layout()
    _save_sweep_figure(fig, save_name)
    plt.close(fig)


def plot_stochastic_sweep(results: dict[float, dict[str, list[float]]], save_name: str):
    """Plot normalized suboptimality (V* - V^method) / V* vs slip probability,
    shaded by +/- 1 SEM across tasks.

    Two families of lines are drawn, if present in `results`:
      - Composed-task methods ("onoff"/"boolean", solid): suboptimality of
        the zero-shot composed policy, averaged over held-out composed tasks.
      - Own-task baselines ("universal"/"empty"/"base_tasks", dashed): each
        constituent EQ evaluated on the very task it was trained on -- no
        composition involved. This isolates how much suboptimality is
        already present from imperfect Goal-Oriented Q-learning under slip,
        as opposed to being introduced by the composition step itself.

    Args:
        results: slip_prob -> method -> list of normalized suboptimalities.
        save_name: path to save the figure.
    """
    slip_probs = sorted(results.keys())
    composed_methods = [m for m in ["onoff", "boolean"] if m in results[slip_probs[0]]]
    own_methods = [m for m in ["universal", "empty", "base_tasks"] if m in results[slip_probs[0]]]
    methods = composed_methods + own_methods

    labels = {
        "onoff": "Goal-set composition (ours)",
        "boolean": "Original BTA composition",
        "universal": "Universal task",
        "empty": "Empty task",
        "base_tasks": "Base tasks",
    }
    linestyles = {"onoff": "-", "boolean": "-", "universal": "--", "empty": "--", "base_tasks": "--"}
    markers = {"onoff": "o", "boolean": "o", "universal": "s", "empty": "^", "base_tasks": "D"}
    colors = dict(zip(methods, plt.cm.tab10.colors))

    _set_sweep_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        values = [results[p][method] for p in slip_probs]
        means = np.array([100 * np.mean(v) for v in values])
        sems = np.array([100 * np.std(v) / max(len(v), 1) ** 0.5 for v in values])
        ax.plot(
            slip_probs, means,
            linestyle=linestyles[method], marker=markers[method],
            color=colors[method], linewidth=2, markersize=6, label=labels[method],
        )
        ax.fill_between(slip_probs, means - sems, means + sems, color=colors[method], alpha=0.15)

    ax.set_xlabel("Slip probability $p$", fontsize=SWEEP_LABEL_SIZE)
    ax.set_ylabel(r"Normalized suboptimality (\%)", fontsize=SWEEP_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=SWEEP_TICK_SIZE)
    ax.legend(fontsize=SWEEP_LEGEND_SIZE)
    plt.tight_layout()
    _save_sweep_figure(fig, save_name)
    plt.close(fig)


def plot_returns_optimality_all_num_goals(
    returns_per_maxiter: dict[int, dict[int, dict[tuple[int, int], dict[str, list[float]]]]],
    optimal_return: dict[int, float],
    save_name: str,
):
    """Plot returns statistics across max_iters for multiple num_goals."""
    rc("text", usetex=True)
    num_goals_list = returns_per_maxiter.keys()
    methods = ["onoff", "boolean"]
    colors = plt.cm.tab10.colors[:2]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for subplot_idx, num_goals in enumerate(num_goals_list):
        maxiters = sorted(returns_per_maxiter[num_goals].keys())
        ax = axes[subplot_idx]
        ax.set_xscale('log')
        # ax.set_xticks([10**k for k in range(len(maxiters) + 1)])
        # ax.set_xticklabels([rf"$10^{{{k}}}$" for k in range(len(maxiters) + 1)])
        ax.tick_params(axis='both', labelsize=18)
        
        for idx, method in enumerate(methods):
            means = []
            
            for maxiter in maxiters:
                all_returns = []
                for task_returns in returns_per_maxiter[num_goals][maxiter].values():
                    all_returns.extend(task_returns[method])
                
                all_returns = np.array(all_returns)
                mean = np.mean(all_returns)
                means.append(mean)
            
            ax.plot(maxiters, means, 'o-', color=colors[idx], linewidth=2, markersize=6, label=method)
        
        ax.axhline(optimal_return[num_goals], color='black', linestyle='--', linewidth=2, label='Optimal')

        if subplot_idx == 0:
            ax.set_ylabel("Returns", fontsize=20)

        if subplot_idx == 1:
            ax.legend(fontsize=18, labels=[r'Univ./Empty (Ours)', r'Base Tasks', r'Optimal'])
            ax.text(0.5, -0.17, "Max iterations", ha='center', transform=ax.transAxes, fontsize=20)
        
        yticks = list(ax.get_yticks())[1:-1]
        yticks.append(optimal_return[num_goals])
        ax.set_yticks(sorted(set(yticks)))

        ax.text(0.5, -0.25, f"{num_goals} total goals", ha='center', transform=ax.transAxes, fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()