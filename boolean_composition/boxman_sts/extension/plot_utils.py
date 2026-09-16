import math
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap


METHODS = ["onoff", "boolean"]
METHOD_LABELS = [r"Univ./Empty (Ours)", r"Base Tasks"]
COLORS = ["#1A5276", "#C0560A"]
CONVERGENCE_MARKER = "^-"
CONVERGENCE_MARKER_SIZE = 8
TASK_LABELS = {
    "B": "Blue",
    "S": "Square",
    "B+S": r"Blue $\vee$ Square",
    "B.S": r"Blue $\wedge$ Square",
    "BxorS": "Blue xor Square",
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "white_orange",
    ["#ffffff", "#fff2d7", "#f7b45d", "#d95f02"],
)
HEATMAP_CMAP.set_bad(alpha=0.0)


def plot_convergence(returns_per_steps, figure_path):
    _set_style()
    steps = sorted(returns_per_steps)
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, method in enumerate(METHODS):
        means, stds = _mean_std_by_step(returns_per_steps, steps, method)
        ax.plot(steps, means, CONVERGENCE_MARKER, color=COLORS[idx], linewidth=2, markersize=CONVERGENCE_MARKER_SIZE, label=method)
        ax.fill_between(steps, means - stds, means + stds, color=COLORS[idx], alpha=0.2)
    _format_axis(ax, with_ylabel=True)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=METHOD_LABELS, fontsize=20)
    plt.tight_layout()
    _save_figure(fig, figure_path)
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
            ax.plot(steps, means, CONVERGENCE_MARKER, color=COLORS[idx], linewidth=2, markersize=CONVERGENCE_MARKER_SIZE, label=method)
            ax.fill_between(steps, means - stds, means + stds, color=COLORS[idx], alpha=0.2)
        ax.set_title(TASK_LABELS.get(task_name, task_name), fontsize=20)
        _format_axis(ax)

    for ax in axes[len(task_names):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, METHOD_LABELS, loc="upper center", ncol=len(METHODS), fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, figure_path)
    plt.close(fig)


def plot_fair_convergence(returns_per_steps, figure_path):
    _set_style()
    steps, boolean_steps = fair_step_pairs(returns_per_steps)
    if not steps:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    onoff_means, onoff_stds = _mean_std_by_step(returns_per_steps, steps, "onoff")
    boolean_means, boolean_stds = _mean_std_by_step(
        returns_per_steps,
        boolean_steps,
        "boolean",
    )
    series = (
        (onoff_means, onoff_stds, COLORS[0], METHOD_LABELS[0]),
        (boolean_means, boolean_stds, COLORS[1], METHOD_LABELS[1]),
    )
    for means, stds, color, label in series:
        ax.plot(steps, means, CONVERGENCE_MARKER, color=color, linewidth=2, markersize=CONVERGENCE_MARKER_SIZE, label=label)
        ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)
    _format_axis(ax, with_ylabel=True)
    ax.legend(fontsize=20)
    plt.tight_layout()
    _save_figure(fig, figure_path)
    plt.close(fig)


def plot_value_progression(step_maps, policy_maps, map_image, figure_path, title=None):
    _set_style()
    steps = sorted(step_maps)
    if not steps:
        return

    ncols = min(5, len(steps))
    nrows = math.ceil(len(steps) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols + 0.45, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)
    image_extent = (0, map_image.shape[1], map_image.shape[0], 0)
    last_mesh = None

    for ax, step in zip(axes, steps):
        ax.imshow(map_image, extent=image_extent)
        heatmap = _resize_heatmap(step_maps[step], map_image.shape[:2])
        vmin, vmax = _value_limits({step: step_maps[step]})
        last_mesh = ax.imshow(
            heatmap,
            cmap=HEATMAP_CMAP,
            alpha=0.62,
            vmin=vmin,
            vmax=vmax,
            extent=image_extent,
        )
        _draw_policy_arrows(ax, policy_maps.get(step), map_image.shape[:2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(str(step), fontsize=16)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(steps):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0, 0.94, 0.96 if title else 1))
    if last_mesh is not None:
        bbox = _axes_bbox(fig, axes[:len(steps)])
        cax = fig.add_axes([0.955, bbox.y0, 0.018, bbox.height])
        fig.colorbar(last_mesh, cax=cax)
        cax.tick_params(labelsize=12)
    _save_figure(fig, figure_path, dpi=200)
    plt.close(fig)


def plot_value_map(value_map, policy_map, map_image, figure_path):
    _set_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    image_extent = (0, map_image.shape[1], map_image.shape[0], 0)
    ax.imshow(map_image, extent=image_extent)
    heatmap = _resize_heatmap(value_map, map_image.shape[:2])
    vmin, vmax = _value_limits({"last": value_map})
    mesh = ax.imshow(
        heatmap,
        cmap=HEATMAP_CMAP,
        alpha=0.62,
        vmin=vmin,
        vmax=vmax,
        extent=image_extent,
    )
    _draw_policy_arrows(ax, policy_map, map_image.shape[:2])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    bbox = ax.get_position()
    cax = fig.add_axes([0.94, bbox.y0, 0.035, bbox.height])
    fig.colorbar(mesh, cax=cax)
    cax.tick_params(labelsize=12)
    _save_figure(fig, figure_path, dpi=200)
    plt.close(fig)


def fair_figure_path(figure_path):
    return figure_path.with_name(f"{figure_path.stem}_fair_iterations{figure_path.suffix}")


def by_task_figure_path(figure_path):
    return figure_path.with_name(f"{figure_path.stem}_by_task{figure_path.suffix}")


def _save_figure(fig, figure_path, dpi=None):
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    for suffix in (".png", ".pdf"):
        fig.savefig(str(figure_path.with_suffix(suffix)), **save_kwargs)


def fair_step_pairs(returns_per_steps):
    available_steps = set(returns_per_steps)
    steps = [step for step in sorted(available_steps) if step // 2 in available_steps and step % 2 == 0]
    return steps, [step // 2 for step in steps]


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
        ax.set_xlabel("Total training iterations (for all UVFA summed)", fontsize=20)
    if with_ylabel:
        ax.set_ylabel("Episode return", fontsize=20)


def _resize_heatmap(value_map, image_shape):
    row_scale = image_shape[0] // value_map.shape[0]
    col_scale = image_shape[1] // value_map.shape[1]
    return np.kron(value_map, np.ones((row_scale, col_scale)))


def _value_limits(step_maps):
    values = np.concatenate([np.ravel(value_map) for value_map in step_maps.values()])
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def _axes_bbox(fig, axes):
    boxes = [ax.get_position() for ax in axes]
    return matplotlib.transforms.Bbox.union(boxes)


def _draw_policy_arrows(ax, policy_map, image_shape):
    if policy_map is None:
        return

    rows, cols = policy_map.shape
    cell_h = image_shape[0] / rows
    cell_w = image_shape[1] / cols
    centers_x = (np.arange(cols) + 0.5) * cell_w
    centers_y = (np.arange(rows) + 0.5) * cell_h
    directions = {
        0: (0.0, -1.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (-1.0, 0.0),
        4: (0.0, 0.0),
    }
    xs, ys, us, vs = [], [], [], []
    stay_xs, stay_ys = [], []
    length = min(cell_w, cell_h) * 0.28
    for row in range(rows):
        for col in range(cols):
            action = policy_map[row, col]
            if not np.isfinite(action):
                continue
            dx, dy = directions[int(action)]
            x = centers_x[col]
            y = centers_y[row]
            if dx == 0.0 and dy == 0.0:
                stay_xs.append(x)
                stay_ys.append(y)
                continue
            xs.append(x)
            ys.append(y)
            us.append(dx * length)
            vs.append(dy * length)

    if xs:
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.006,
            headwidth=4,
            headlength=5,
            headaxislength=4.5,
            pivot="middle",
            zorder=4,
        )
    if stay_xs:
        ax.scatter(stay_xs, stay_ys, s=8, c="black", marker="o", zorder=4)


# ------------------------------------------------------------
# Stochastic sweep (exp_stochastic_sweep.py)
# ------------------------------------------------------------
SWEEP_COMPOSED_METHODS = ["onoff", "boolean"]
SWEEP_OWN_METHODS = ["universal", "empty", "base_tasks"]
SWEEP_LABELS = {
    "onoff": "Univ./Empty (Ours)",
    "boolean": "Base Tasks",
    "universal": "Universal task",
    "empty": "Empty task",
    "base_tasks": "Base tasks",
    "optimal": "Optimal ($V^*$)",
}
SWEEP_COLORS = {
    "onoff": "#1A5276",
    "boolean": "#C0560A",
    "universal": "#7F8C8D",
    "empty": "#9B59B6",
    "base_tasks": "#27AE60",
    "optimal": "#000000",
}
SWEEP_MARKERS = {
    "onoff": "o",
    "boolean": "o",
    "universal": "s",
    "empty": "^",
    "base_tasks": "D",
    "optimal": "",
}
SWEEP_LINESTYLES = {
    "onoff": "-",
    "boolean": "-",
    "universal": "--",
    "empty": "--",
    "base_tasks": "--",
    "optimal": ":",
}


def plot_stochastic_sweep(aggregated, figure_path):
    """Two views of the slip sweep, sharing an x axis.

    Args:
        aggregated: metric -> slip_prob -> method -> list of per-seed scalars,
            as produced by `exp_stochastic_sweep.run_aggregate`. Shading is
            +/- 1 SEM across seeds.
        figure_path: where to write the figure (.png and .pdf are both saved).

    Panels:
      - The value gap V* - V^pi, in units of episode return. This is the
        comparison the experiment is actually making: at a fixed slip
        probability the two composition methods are measured against the same
        V*, in the same units, so their lines can be read against each other
        directly. The dashed own-task lines are the control for "everything
        gets harder as slip rises" -- if the composed gaps grow no faster than
        those, the composition is not what broke.
      - The raw expected return behind those gaps, with V* drawn in, so the
        shrinking headroom at high slip is visible rather than implied.

    `aggregated` also carries `regret` and `subopt`; neither is plotted (see
    the module docstring of exp_stochastic_sweep.py for why).
    """
    slip_probs = sorted(aggregated["gap"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    _plot_sweep_panel(
        axes[0], aggregated["gap"], slip_probs,
        SWEEP_COMPOSED_METHODS + SWEEP_OWN_METHODS, scale=1,
        ylabel="Value gap $V^* - V^\\pi$ (return)",
        title="Expected return given up",
    )
    _plot_sweep_panel(
        axes[1], aggregated["returns"], slip_probs,
        SWEEP_COMPOSED_METHODS + ["optimal"], scale=1,
        ylabel="Expected episode return",
        title="Raw expected return",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    optimal_handles, optimal_labels = axes[1].get_legend_handles_labels()
    for handle, label in zip(optimal_handles, optimal_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save_figure(fig, figure_path)
    plt.close(fig)


SWEEP_LABEL_SIZE = 24
SWEEP_TICK_SIZE = 20
SWEEP_LEGEND_SIZE = 15


def _plot_sweep_panel(ax, results, slip_probs, methods, scale, ylabel, title, zero_line=True,
                      label_size=16, tick_size=12):
    for method in methods:
        if method not in results[slip_probs[0]]:
            continue
        values = [np.asarray(results[p][method], dtype=float) for p in slip_probs]
        means = np.array([scale * v.mean() for v in values])
        sems = np.array([scale * v.std() / max(len(v), 1) ** 0.5 for v in values])
        ax.plot(
            slip_probs, means,
            linestyle=SWEEP_LINESTYLES[method],
            marker=SWEEP_MARKERS[method],
            color=SWEEP_COLORS[method],
            linewidth=2, markersize=6,
            label=SWEEP_LABELS[method],
        )
        ax.fill_between(slip_probs, means - sems, means + sems, color=SWEEP_COLORS[method], alpha=0.15)
    if zero_line:
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Slip probability $p$", fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    if title:
        ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=tick_size)


def _single_panel_sweep(aggregated_metric, figure_path, methods, ylabel, title):
    """One standalone slip-sweep figure drawn with the shared panel styling."""
    _set_style()  # LaTeX text, as the other Boxman figures use
    slip_probs = sorted(aggregated_metric)
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_sweep_panel(
        ax, aggregated_metric, slip_probs, methods, scale=1,
        ylabel=ylabel, title=title, zero_line=False,
        label_size=SWEEP_LABEL_SIZE, tick_size=SWEEP_TICK_SIZE,
    )
    ax.legend(fontsize=SWEEP_LEGEND_SIZE)
    fig.tight_layout()
    _save_figure(fig, figure_path)
    plt.close(fig)


def plot_stochastic_sweep_gap(aggregated, figure_path):
    """The value gap V* - V^pi on its own, in units of episode return.

    The same data as the left panel of `plot_stochastic_sweep`, drawn alone so
    it can be used as a standalone figure. Both composition methods are
    measured against the same V* at each slip probability, so the lines can be
    read against each other directly; the dashed own-task lines are the
    control for "everything gets harder as slip rises".
    """
    _single_panel_sweep(
        aggregated["gap"], figure_path,
        SWEEP_COMPOSED_METHODS + SWEEP_OWN_METHODS,
        ylabel="Return",
        title=None,
    )


def plot_stochastic_sweep_returns(aggregated, figure_path):
    """Mean return of actual rollout episodes.

    The Monte Carlo counterpart of the DP figures: the composed policies are
    rolled out in the real Boxman env under slip and their returns averaged,
    rather than solved for. Agreement with the `returns` panel of
    `plot_stochastic_sweep` is a check that the closed-form dynamics in
    `mdp_utils` match the environment.

    No optimal reference line here: `V*` is a value, not a policy that can be
    rolled out, and the stationary policy greedy w.r.t. `V_H` is a slight
    underestimate of it. The DP panels carry the exact optimum instead.
    """
    _single_panel_sweep(
        aggregated["rollout"], figure_path,
        SWEEP_COMPOSED_METHODS + SWEEP_OWN_METHODS,
        ylabel="Return",
        title=None,
    )


TRAINING_TABLE_COLUMNS = [
    ("slip_prob", 9),
    ("task", 8),
    ("seeds", 7),
    ("train_steps", 12),
    ("own_gap", 12),
    ("own_regret_%", 14),
    ("own_return", 12),
    ("optimal_return", 15),
]


def write_training_table(records, path):
    """Plain-text table of each constituent UVFA's own-task quality.

    The counterpart of the Rooms sweep's convergence table: a DQN is trained
    for a fixed budget rather than to convergence, so what is worth tabulating
    is not how long it took but how good it ended up -- on the very task it was
    trained on, with no composition involved -- at each slip probability.

    Args:
        records: dicts with "seed", "slip_prob", "task", "steps", "own_gap",
            "own_regret", "own_return" and "optimal_return", one per
            (seed, slip probability, constituent task).
        path: where to write the table.
    """
    groups = {}
    for record in records:
        groups.setdefault((float(record["slip_prob"]), str(record["task"])), []).append(record)

    header = " | ".join(name.rjust(width) for name, width in TRAINING_TABLE_COLUMNS)
    lines = [header, "-+-".join("-" * width for _, width in TRAINING_TABLE_COLUMNS)]
    for slip_prob, task in sorted(groups):
        rows = groups[(slip_prob, task)]
        values = [
            f"{slip_prob:.2f}",
            task,
            str(len(rows)),
            f"{np.mean([r['steps'] for r in rows]):.0f}",
            f"{np.mean([r['own_gap'] for r in rows]):.3f}",
            f"{100 * np.mean([r['own_regret'] for r in rows]):.2f}",
            f"{np.mean([r['own_return'] for r in rows]):.3f}",
            f"{np.mean([r['optimal_return'] for r in rows]):.3f}",
        ]
        lines.append(" | ".join(v.rjust(width) for v, (_, width) in zip(values, TRAINING_TABLE_COLUMNS)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"Training table saved to {path}")
