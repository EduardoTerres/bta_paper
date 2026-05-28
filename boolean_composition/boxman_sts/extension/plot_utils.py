import math

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
