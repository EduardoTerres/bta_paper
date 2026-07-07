"""Inspect saved agent checkpoints (actor/critic/target_critic .pt state dicts).

Also prints the "goal collapse" check for state-based four-rooms checkpoints: it
extracts the trained psi (state) encoder from the critic checkpoint and evaluates it
at the environment's actual goal (x, y) positions, showing the point their embeddings
collapse toward (their centroid) and how far apart they still are. Works with either
raw (x, y) or one-hot observation encodings -- the input dimensionality is inferred
from the checkpoint itself, and goal inputs are built to match.

Usage:
    python inspect_checkpoint.py
    python inspect_checkpoint.py snapshots/GCRB_four_rooms_8_36582_1783331985
    python inspect_checkpoint.py snapshots/GCRB_four_rooms_8_36582_1783331985/agents/actorbest_agent.pt
    python inspect_checkpoint.py --num-rooms 8 --checkpoint-name final_agent
"""

import argparse
import os
import sys

import torch

# Make this script runnable regardless of cwd/PYTHONPATH: it needs the GCB root on
# sys.path for `environments.four_rooms...` and this extension dir for `state_gcab...`.
EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))
GCB_ROOT = os.path.dirname(EXTENSION_DIR)
for _path in (GCB_ROOT, EXTENSION_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_SNAPSHOT = "snapshots/GCRB_four_rooms_state_4_77142_1783340862"


def find_pt_files(path):
    if os.path.isfile(path):
        return [path]

    agents_dir = os.path.join(path, "agents")
    search_dir = agents_dir if os.path.isdir(agents_dir) else path

    files = sorted(
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.endswith(".pt")
    )
    if not files:
        raise FileNotFoundError(f"No .pt files found under {search_dir}")
    return files


def describe_tensor(t):
    t = t.float()
    return (
        f"shape={tuple(t.shape)!s:<20} dtype={t.dtype!s:<12} "
        f"numel={t.numel():<10} mean={t.mean().item(): .4e} "
        f"std={t.std().item(): .4e} min={t.min().item(): .4e} max={t.max().item(): .4e}"
    )


def inspect_file(path):
    print(f"\n=== {path} ===")
    state_dict = torch.load(path, map_location="cpu")

    if not isinstance(state_dict, dict):
        print(f"Loaded object is not a state dict, got {type(state_dict)}: {state_dict}")
        return

    total_params = 0
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            total_params += value.numel()
            print(f"  {key:<40} {describe_tensor(value)}")
        else:
            print(f"  {key:<40} (non-tensor entry) {value!r}")

    print(f"  -- total parameters: {total_params:,}")


def load_vector_encoder(pt_path, feature_dim, num_layers, num_filters, prefix="encoder."):
    """Reconstruct the standalone psi VectorEncoder from a saved actor/critic
    checkpoint (its state dict only has the `encoder.*`-prefixed keys populated).

    The input dimensionality is read off the checkpoint's first layer rather than
    assumed, so this works for both the raw (x, y) (dim 2) and one-hot (dim
    num_positions) observation encodings without the caller needing to know which
    one was used to train it.
    """
    from state_gcab.encoders import VectorEncoder

    full_state = torch.load(pt_path, map_location="cpu")
    encoder_state = {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}
    if not encoder_state:
        raise ValueError(f"No keys with prefix '{prefix}' found in {pt_path}; not a psi checkpoint?")

    in_dim = encoder_state["net.0.weight"].shape[1]
    encoder = VectorEncoder((in_dim,), feature_dim, num_layers, num_filters)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    return encoder


def goal_positions(num_rooms):
    from environments.four_rooms.four_rooms_wrapper import CONFIGS
    return CONFIGS[num_rooms]["Goals"]


def all_state_positions(num_rooms):
    from state_gcab.environment import VectorFourRoomsGoalEnv
    env = VectorFourRoomsGoalEnv(num_rooms=num_rooms)
    return env, list(env.env.possiblePositions)


def state_inputs(positions, num_rooms, in_dim, env=None):
    """Build the model-input tensor for each (x, y) position, matching whichever
    observation encoding the checkpoint was actually trained with (raw or one-hot).
    """
    if in_dim == 2:
        return torch.tensor(positions, dtype=torch.float32)

    if env is None:
        from state_gcab.environment import VectorFourRoomsGoalEnv
        env = VectorFourRoomsGoalEnv(num_rooms=num_rooms)

    if env.observation_space.shape[0] != in_dim:
        raise ValueError(
            f"Checkpoint's encoder expects input dim {in_dim}, but the one-hot "
            f"observation space for num_rooms={num_rooms} has dim "
            f"{env.observation_space.shape[0]}; wrong --num-rooms?"
        )
    one_hots = [env._one_hot_position(pos) for pos in positions]
    return torch.tensor(one_hots, dtype=torch.float32)


def goal_inputs(num_rooms, in_dim):
    goals = goal_positions(num_rooms)
    return goals, state_inputs(goals, num_rooms, in_dim)


def count_collapsed_clusters(embeddings, threshold):
    """Single-linkage cluster the embeddings at `threshold`: states end up in the
    same cluster iff there's a chain of psi-distances <= threshold connecting them.
    This is the practical, thresholded analogue of a bisimulation equivalence class.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(embeddings) < 2:
        return len(embeddings), [1] * len(embeddings)

    condensed = pdist(embeddings.numpy())
    linkage_matrix = linkage(condensed, method="single")
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    return len(set(labels)), labels


def inspect_state_collapse(snapshot_dir, num_rooms, feature_dim, num_layers, num_filters, checkpoint_name,
                            collapse_fractions=(0.01, 0.05, 0.1, 0.25)):
    """Print how many distinct clusters the trained psi encoder collapses the full
    reachable state space into, at a few distance thresholds (states are "the same"
    under the abstraction if their psi embeddings end up in the same cluster).
    """
    pt_path = os.path.join(snapshot_dir, "agents", f"critic{checkpoint_name}.pt")
    print(f"\n=== State-space collapse in psi, from {pt_path} ===")

    encoder = load_vector_encoder(pt_path, feature_dim, num_layers, num_filters)
    in_dim = encoder.obs_shape[0]

    env, positions = all_state_positions(num_rooms)
    inputs = state_inputs(positions, num_rooms, in_dim, env=env)
    with torch.no_grad():
        embeddings = encoder(inputs)

    n = len(positions)
    pairwise = torch.cdist(embeddings, embeddings)
    off_diag = pairwise[~torch.eye(n, dtype=torch.bool)]
    max_dist = off_diag.max().item()

    print(f"  {n} reachable states for num_rooms={num_rooms} "
          f"(psi input dim={in_dim}, feature_dim={feature_dim})")
    print(f"  Pairwise psi distance across all states: "
          f"min={off_diag.min().item():.4e} median={off_diag.median().item():.4e} max={max_dist:.4e}")

    print(f"\n  Number of distinct states the abstraction collapses {n} raw states into, "
          f"by clustering threshold (fraction of max pairwise distance {max_dist:.4e}):")
    print("  (single-linkage: a==b if some chain of psi-distances <= threshold connects them -- "
          "a sudden jump to 1 cluster can be a chaining artifact, not true dense collapse; "
          "check the surrounding thresholds too)")
    for frac in collapse_fractions:
        threshold = frac * max_dist
        num_clusters, _ = count_collapsed_clusters(embeddings, threshold)
        print(f"    threshold={frac:>5.0%} of max ({threshold:.4e}) -> "
              f"{num_clusters:>4} distinct states (from {n})")


# Sequential single-hue (blue, light -> dark) ramp, per references/palette.md.
_SEQUENTIAL_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281", "#0d366b"]


def plot_goal_distance_matrix(goals, pairwise, out_path):
    """Save a heatmap of pairwise L2 distances between the psi(goal) embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("sequential_blue", _SEQUENTIAL_BLUE_RAMP)
    matrix = pairwise.numpy()
    n = len(goals)
    labels = [f"({gx}, {gy})" for gx, gy in goals]
    vmax = matrix.max() if matrix.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(1.4 * n + 2, 1.3 * n + 2))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("goal")
    ax.set_ylabel("goal")
    ax.set_title("Pairwise distance between psi(goal) embeddings")

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            text_color = "white" if value > 0.6 * vmax else "#1a1a1a"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("L2 distance")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved pairwise distance matrix plot -> {out_path}")


def inspect_goal_collapse(snapshot_dir, num_rooms, feature_dim, num_layers, num_filters, checkpoint_name,
                           plot_out=None):
    """Print the psi(x, y) embedding for every goal state in the environment, plus the
    centroid they collapse toward, and save a pairwise-distance heatmap between them.
    """
    pt_path = os.path.join(snapshot_dir, "agents", f"critic{checkpoint_name}.pt")
    print(f"\n=== Goal collapse in psi (state encoder), from {pt_path} ===")

    encoder = load_vector_encoder(pt_path, feature_dim, num_layers, num_filters)
    in_dim = encoder.obs_shape[0]
    encoding = "raw (x, y)" if in_dim == 2 else "one-hot"

    goals, goal_tensor = goal_inputs(num_rooms, in_dim)
    with torch.no_grad():
        embeddings = encoder(goal_tensor)

    print(f"  {len(goals)} goal states for num_rooms={num_rooms} ({encoding} input, dim={in_dim}): {goals}")
    for (gx, gy), emb in zip(goals, embeddings):
        print(f"  psi({gx:>2}, {gy:>2}) -> mean={emb.mean().item(): .4e} std={emb.std().item(): .4e} "
              f"first5={[f'{v:.4f}' for v in emb[:5].tolist()]}")

    centroid = embeddings.mean(dim=0)
    dists_to_centroid = torch.norm(embeddings - centroid, dim=1)
    pairwise = torch.cdist(embeddings, embeddings)

    print(f"\n  State the goals collapse to (centroid of psi(goal) over all goals):")
    print(f"    mean={centroid.mean().item(): .4e} std={centroid.std().item(): .4e} "
          f"first5={[f'{v:.4f}' for v in centroid[:5].tolist()]}")
    print(f"  Distance of each goal's embedding to that centroid: "
          f"{[f'{d.item():.4e}' for d in dists_to_centroid]}")
    print(f"  Max pairwise distance among goal embeddings: {pairwise.max().item():.4e} "
          f"(0 => goals fully collapse to a single point)")

    if plot_out is None:
        plot_out = os.path.join(snapshot_dir, f"goal_distance_matrix_{checkpoint_name}.png")
    plot_goal_distance_matrix(goals, pairwise, plot_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path", nargs="?", default=DEFAULT_SNAPSHOT,
        help=f"Path to a .pt file, or a snapshot directory containing an agents/ folder "
             f"(default: {DEFAULT_SNAPSHOT})",
    )
    parser.add_argument("--num-rooms", type=int, default=4, choices=[4, 8, 16],
                         help="Which four-rooms map's goal positions to probe")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--checkpoint-name", type=str, default="best_agent",
                         help="'best_agent' or 'final_agent'")
    parser.add_argument("--skip-goal-collapse", action="store_true",
                         help="Only dump raw tensor stats, skip the psi(goal) collapse check")
    parser.add_argument("--plot-out", type=str, default=None,
                         help="Where to save the pairwise-distance heatmap PNG "
                              "(default: <snapshot_dir>/goal_distance_matrix_<checkpoint_name>.png)")
    parser.add_argument("--skip-state-collapse", action="store_true",
                         help="Skip counting how many distinct states psi collapses the full "
                              "reachable state space into")
    parser.add_argument("--collapse-fractions", type=str, default="0.01,0.05,0.1,0.25,0.5,0.75,1.0",
                         help="Comma-separated fractions of the max pairwise psi distance to use "
                              "as clustering thresholds when counting collapsed states")
    args = parser.parse_args()

    for pt_file in find_pt_files(args.path):
        inspect_file(pt_file)

    snapshot_dir = args.path if os.path.isdir(args.path) else os.path.dirname(os.path.dirname(args.path))

    if not args.skip_goal_collapse:
        try:
            inspect_goal_collapse(
                snapshot_dir, args.num_rooms, args.feature_dim, args.num_layers, args.num_filters,
                args.checkpoint_name, plot_out=args.plot_out,
            )
        except Exception as exc:
            print(f"\n(Skipping goal-collapse check: {exc})")

    if not args.skip_state_collapse:
        try:
            collapse_fractions = [float(f) for f in args.collapse_fractions.split(",") if f.strip()]
            inspect_state_collapse(
                snapshot_dir, args.num_rooms, args.feature_dim, args.num_layers, args.num_filters,
                args.checkpoint_name, collapse_fractions=collapse_fractions,
            )
        except Exception as exc:
            print(f"\n(Skipping state-collapse check: {exc})")


if __name__ == "__main__":
    main()
