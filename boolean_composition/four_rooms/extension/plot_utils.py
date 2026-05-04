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