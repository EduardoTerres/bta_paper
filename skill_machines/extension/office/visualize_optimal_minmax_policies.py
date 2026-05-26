import os
import sys

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import envs  # noqa: F401
from sm import TaskPrimitive

PLOT_RC = {
    "text.usetex": False,
    "text.latex.preamble": "",
    "font.family": "DejaVu Sans",
}
matplotlib.rcParams.update(PLOT_RC)
plt.rcParams.update(PLOT_RC)

CHECKPOINT_DIR = os.path.join(
    SCRIPT_DIR,
    "exps_data_extension",
    "sp_ql",
    "Office-CoffeeMail-Task-v0",
    "optimal",
    "5000000",
    "return",
)
OUTPUT = os.path.join(
    SCRIPT_DIR,
    "exps_data_extension",
    "figures",
    "optimal_min_max_policies.png",
)


def load_checkpoint():
    goals = list(torch.load(os.path.join(CHECKPOINT_DIR, "goals")).values())
    wvfs = {
        "min / wvf_0": torch.load(os.path.join(CHECKPOINT_DIR, "wvf_0")),
        "max / wvf_1": torch.load(os.path.join(CHECKPOINT_DIR, "wvf_1")),
    }
    return goals, wvfs


def state_key(primitive_env, position, goal, violated_constraints):
    state = {
        "env_state": np.array(position, dtype=np.uint8),
        "violated_constraints": violated_constraints.copy(),
        "desired_goal": goal.copy(),
    }
    return gym.spaces.flatten(primitive_env.observation_space, state).tobytes()


def task_goal_vectors(primitive_env, learned_goals):
    goal_specs = []
    for prop in ("m", "c", "o"):
        goal = np.zeros(2 * len(primitive_env.predicates), dtype=np.uint8)
        goal[primitive_env.predicates.index(prop)] = 1
        match = next((g for g in learned_goals if np.array_equal(g, goal)), None)
        if match is None:
            raise RuntimeError(f"Missing learned goal for proposition {prop}: {goal.tolist()}")
        goal_specs.append((prop, match))
    return goal_specs


def value_ranges(primitive_env, wvfs, goal_specs, violated_constraints):
    ranges = {}
    for label, table in wvfs.items():
        values = []
        for _, goal in goal_specs:
            for position in primitive_env.environment.possiblePositions:
                q = table.get(state_key(primitive_env, position, goal, violated_constraints))
                if q is not None:
                    values.append(float(np.max(q)))
        ranges[label] = (min(values), max(values)) if values else (0.0, 1.0)
    return ranges


def draw_object_labels(ax, env):
    object_labels = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "d": "dec",
        "c": "coffee",
        "m": "mail",
        "o": "office",
        "tm": "mail+",
        "to": "office+",
    }
    for position, objects in env.position_predicate.items():
        names = []
        for obj in objects:
            name = obj if isinstance(obj, str) else obj.object
            if name in object_labels:
                names.append(object_labels[name])
        if names:
            ax.text(
                position[1] + 0.5,
                position[0] + 0.52,
                "\n".join(names),
                ha="center",
                va="center",
                fontsize=5.5,
                color="black",
                weight="bold",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=8,
            )


def draw_policy(ax, primitive_env, table, goal, value_range, violated_constraints):
    env = primitive_env.environment
    cmap = plt.cm.viridis
    vmin, vmax = value_range
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
    action_delta = {
        0: (0.0, -0.34),
        1: (0.34, 0.0),
        2: (0.0, 0.34),
        3: (-0.34, 0.0),
    }

    ax.set_xlim(0, env.n)
    ax.set_ylim(env.m, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for row in range(env.m):
        for col in range(env.n):
            if env.grid[row][col] == 1:
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#171717", edgecolor="#333333", lw=0.35))
                continue

            q = table.get(state_key(primitive_env, (row, col), goal, violated_constraints))
            if q is None:
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#f4f4f4", edgecolor="#d8d8d8", lw=0.35))
                continue

            q = np.asarray(q)
            best_action = int(np.argmax(q))
            best_value = float(np.max(q))
            ax.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=cmap(norm(best_value)),
                    edgecolor="white",
                    lw=0.35,
                    alpha=0.92,
                )
            )

            env_action = best_action % env.action_space.n
            done_action = best_action >= env.action_space.n
            center_x, center_y = col + 0.5, row + 0.5
            if done_action:
                ax.add_patch(
                    Circle(
                        (center_x, center_y),
                        0.18,
                        facecolor="white",
                        edgecolor="#111111",
                        lw=0.8,
                        zorder=4,
                    )
                )
                ax.text(
                    center_x,
                    center_y + 0.01,
                    "T",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    weight="bold",
                    color="#111111",
                    zorder=5,
                )
            else:
                dx, dy = action_delta[env_action]
                ax.add_patch(
                    FancyArrowPatch(
                        (center_x - dx * 0.45, center_y - dy * 0.45),
                        (center_x + dx, center_y + dy),
                        arrowstyle="-|>",
                        mutation_scale=6.5,
                        lw=0.6,
                        color="white",
                        zorder=5,
                    )
                )

    draw_object_labels(ax, env)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def main():
    matplotlib.rcParams.update(PLOT_RC)
    plt.rcParams.update(PLOT_RC)
    task_env = gym.make("Office-CoffeeMail-Task-v0")
    primitive_env = TaskPrimitive(task_env.environment)
    matplotlib.rcParams.update(PLOT_RC)
    plt.rcParams.update(PLOT_RC)
    learned_goals, wvfs = load_checkpoint()
    goal_specs = task_goal_vectors(primitive_env, learned_goals)
    violated_constraints = np.zeros(len(primitive_env.predicates), dtype=np.uint8)
    ranges = value_ranges(primitive_env, wvfs, goal_specs, violated_constraints)

    prop_labels = {"m": "mail", "c": "coffee", "o": "office"}
    fig, axes = plt.subplots(len(goal_specs), len(wvfs), figsize=(11, 12), dpi=180)
    for row, (prop, goal) in enumerate(goal_specs):
        for col, (policy_label, table) in enumerate(wvfs.items()):
            ax = axes[row, col]
            ax.set_title(f"{policy_label} | goal: {prop_labels[prop]}", fontsize=11, pad=7)
            colorbar = draw_policy(ax, primitive_env, table, goal, ranges[policy_label], violated_constraints)
            cbar = fig.colorbar(colorbar, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7, length=2)

    fig.suptitle(
        "Optimal learned min and max primitive policies, return checkpoint, Office-CoffeeMail",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.017,
        "Arrows show greedy movement actions. T marks greedy terminal/done actions. Color shows max Q value for the goal-conditioned WVF.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
