"""Does BTA's error grow with the number of composition operations?

The stochastic sweep (`exp_stochastic_sweep.py`) shows the original Boolean
Task Algebra (BTA) composition degrading faster than goal-set composition once
the constituent value functions stop being exact. This experiment isolates the
*mechanism* behind that: the claim is that BTA is worse not because AND/OR/NOT
are wrong, but because reaching a given element of the algebra takes a number
of operations that grows with the goal set, and every operation folds in
another approximate value function.

Concretely, BTA builds a task M as

    Q_M  =  OR_{g in M}  AND_{i=1..m}  L_i(g),
    L_i(g) = Q_i                              if bit i of g's code is 1
             (Q_on + Q_off) - Q_i             if it is 0          (a NOT)

with m = ceil(log2 |G|). So a task costs `n_not` NOTs (one per zero bit, over
all its goals), `k * (m - 1)` ANDs and `k - 1` ORs, for k = |M|. Goal-set
composition costs *zero* operations at any k: it selects Q_on or Q_off per
goal and does no arithmetic. That makes it the control -- its curve should be
flat in the operation count while BTA's rises.

Two views, one figure each:

  1. `depth_single`  -- single-goal tasks only (k = 1). Here the AND chain is
     fixed at m - 1, there are no ORs, and every task is "reach one specific
     goal", so difficulty is held roughly constant and the *only* thing that
     varies is the number of NOTs, 0 through m. This is the controlled slice,
     and the cleanest test of the hypothesis.
  2. `depth_all`     -- every evaluated task, binned by total operation count.
     Broader but confounded: tasks with more goals need more operations *and*
     are easier to satisfy. The goal-set line is what controls for that.

Error is injected two ways, chosen with `--mode`:

  - `noise` (default): the constituents are computed *exactly* by dynamic
    programming, then perturbed by zero-mean Gaussian noise of scale
    `--noise-levels`. One draw per (seed, level), shared across all tasks --
    exactly as a single set of learned value functions would be reused. This
    isolates the mechanism: the dynamics stay deterministic, nothing is
    trained, and the only free variables are the error magnitude and the
    operation count. It also runs in minutes.
  - `trained`: the constituents are learned under slip with Goal-Oriented
    Q-learning, as in the sweep, so the error is real learning error. Slower,
    and confounded by slip also changing the task, but it confirms the
    noise-mode result carries over to the setting the paper actually reports.

Everything is written to a timestamped run directory under SCRATCH_DIR and
copied into `--figures-dir`, matching `exp_stochastic_sweep.py`. Run it once
and it produces both figures; there is no separate aggregate step.
"""
import argparse
import os
import socket
from collections import defaultdict
from datetime import datetime

import deepdish as dd
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc

from four_rooms.GridWorld import GridWorld
from four_rooms.config import Config_4, Config_8, Config_16
from four_rooms.library import EQ_P
from four_rooms.extension.utils import (
    build_EQ_from_on_off,
    build_EQ_from_boolean_ops,
    proportional_sample,
)
from four_rooms.extension.mdp_utils import (
    ABSORBED_STATE,
    build_states,
    build_transition_model,
    policy_evaluation,
    train_goal_oriented_until_convergence,
    value_gap,
    value_iteration,
)

np.object = object  # Hack to avoid error in save (kept consistent with the rest of the repo)

SCRATCH_DIR = "/scratch-shared/eterrescaballe/composition_depth/"
GAMMA = 1.0
MAX_EXHAUSTIVE_GOALS = 6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-rooms", type=int, default=int(os.environ.get("NUM_ROOMS", 8)),
                    choices=[4, 8, 16],
                    help="Larger goal sets give a longer operation-count axis "
                         "(m = 2/3/4 NOTs per goal, up to 9/32/91 total ops).")
parser.add_argument("--mode", choices=["noise", "trained"], default="noise",
                    help="How constituent error is produced. See the module docstring.")
parser.add_argument("--seeds", type=int, default=5,
                    help="Noise draws (mode=noise) or training seeds (mode=trained).")
parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.05],
                    help="Gaussian sigma added to every constituent entry (mode=noise). "
                         "Calibrate against the action-value margin, not the value range: "
                         "a step costs 0.1, so adjacent actions differ by about 0.2 and "
                         "sigma beyond ~0.05 randomises the argmax outright. sigma=0 is "
                         "kept as the sanity check -- both methods must read exactly 0.")
parser.add_argument("--slip-probs", nargs="+", type=float, default=[0.1, 0.3],
                    help="Slip probabilities to train under (mode=trained).")
parser.add_argument("--max-steps", type=int, default=200_000,
                    help="Per-EQ training budget (mode=trained).")
parser.add_argument("--tasks-per-cardinality", type=int, default=10,
                    help="Multi-goal tasks sampled per cardinality for the second "
                         "figure. Every single-goal task is always kept, since those "
                         "are the controlled slice and are cheap; the multi-goal tasks "
                         "dominate runtime, so lower this first if the run is too slow.")
parser.add_argument("--horizon", type=int, default=100,
                    help="DP horizon, matching exp_stochastic_sweep.py.")
parser.add_argument("--run-id", type=str, default=None)
parser.add_argument("--figures-dir", type=str, default="four_rooms/extension/figures")
args = parser.parse_args()

CONFIG = {4: Config_4, 8: Config_8, 16: Config_16}[args.num_rooms]
GOALS = CONFIG["Goals"]
T_STATES = CONFIG["T_states"]
BASES = CONFIG["Bases"]
RULES = CONFIG["Composition_rules"]
M_BITS = len(RULES[GOALS[0]])
MAP = "MAP_" + str(args.num_rooms)


# ------------------------------------------------------------
# How many operations does BTA need for a task?
# ------------------------------------------------------------
def composition_cost(task):
    """Operation counts for `build_EQ_from_boolean_ops` on this task.

    Mirrors that function exactly: one NOT per zero bit, `m - 1` ANDs to fold
    the m literals of each goal, and `k - 1` ORs to fold the k goals. `depth`
    is the longest chain any single value passes through, since both folds are
    sequential: 1 (NOT) + (m - 1) (AND chain) + (k - 1) (OR chain).
    """
    k = len(task)
    n_not = sum(M_BITS - int(np.sum(RULES[g])) for g in task)
    n_and = k * (M_BITS - 1)
    n_or = k - 1
    return {"n_not": n_not, "n_and": n_and, "n_or": n_or,
            "n_ops": n_not + n_and + n_or, "depth": M_BITS + k - 1, "k": k}


# ------------------------------------------------------------
# Exact constituents (mode=noise)
# ------------------------------------------------------------
def exact_EQ(env, horizon):
    """EQ*(s, g, a) by backward induction.

    Replicates the target Goal-Oriented Q-learning regresses to: terminating
    at g pays the env's reward at g, terminating at any other goal pays
    N = min(rmin, (rmin - rmax) * diameter).
    """
    states = build_states(env)
    P = build_transition_model(env)
    n_actions = env.action_space.n
    N = min(env.rmin, (env.rmin - env.rmax) * env.diameter)

    EQ = defaultdict(dict)
    for t_state in env.T_states:
        goal = str(t_state)
        Pg = {
            s: {a: [(ns, p, (r if s == goal else N)
                     if ns == ABSORBED_STATE and s != ABSORBED_STATE else r)
                    for ns, p, r in P[s][a]]
                for a in range(n_actions)}
            for s in states
        }
        V = defaultdict(float)
        for _ in range(horizon):
            V = defaultdict(float, {
                s: (0.0 if s == ABSORBED_STATE else
                    max(sum(p * (r + V[ns]) for ns, p, r in Pg[s][a]) for a in range(n_actions)))
                for s in states
            })
        for s in states:
            EQ[s][goal] = np.array(
                [sum(p * (r + V[ns]) for ns, p, r in Pg[s][a]) for a in range(n_actions)]
            )
    return EQ


def perturb(EQ, sigma, rng):
    """Zero-mean Gaussian noise on every entry, as a stand-in for learning error."""
    out = defaultdict(dict)
    for s in EQ:
        for g in EQ[s]:
            out[s][g] = EQ[s][g] + rng.normal(0.0, sigma, size=EQ[s][g].shape)
    return out


def make_envs(slip_prob):
    """The universal, empty and base-task envs, in the order the composition wants."""
    universal = GridWorld(MAP=MAP, goals=T_STATES, slip_prob=slip_prob)
    empty = GridWorld(MAP=MAP, goals=T_STATES, goal_reward=-0.1, slip_prob=slip_prob)
    bases = [GridWorld(MAP=MAP, goals=[[p, p] for p in b], T_states=T_STATES,
                       slip_prob=slip_prob) for b in BASES]
    return universal, empty, bases


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def task_optimum(task, slip_prob, cache):
    """(states, P, V*) for one composed task, cached across seeds and levels."""
    key = (tuple(task), slip_prob)
    if key not in cache:
        env = GridWorld(MAP=MAP, goals=[[p, p] for p in task], T_states=T_STATES,
                        slip_prob=slip_prob)
        states = build_states(env)
        P = build_transition_model(env)
        V_star, _, _ = value_iteration(states, P, env.action_space.n,
                                       gamma=GAMMA, horizon=args.horizon)
        cache[key] = (states, P, env.action_space.n, V_star)
    return cache[key]


def evaluate(tasks, EQ_on, EQ_off, EQ_basis, slip_prob, cache, desc):
    """Value gap of both compositions on every task, tagged with its cost."""
    rows = []
    for task in tqdm(tasks, desc=desc, leave=False):
        states, P, n_actions, V_star = task_optimum(task, slip_prob, cache)
        composed = {
            "onoff": build_EQ_from_on_off(task, GOALS, EQ_on, EQ_off),
            "boolean": build_EQ_from_boolean_ops(task, RULES, EQ_basis, EQ_on, EQ_off),
        }
        row = composition_cost(task)
        for method, EQ in composed.items():
            V_pi = policy_evaluation(states, P, EQ_P(EQ), n_actions,
                                     gamma=GAMMA, horizon=args.horizon)
            row[method] = value_gap(V_star, V_pi, states)
        # The quantity the experiment is really about. Both methods are scored
        # on the same task against the same V*, from the same perturbed
        # constituents, so their difference removes everything except the
        # arithmetic BTA does and goal-set composition does not -- including
        # the fact that a goal's NOT count is a property of *which* goal it is,
        # and so is confounded with how hard that goal is to reach.
        row["excess"] = row["boolean"] - row["onoff"]
        rows.append(row)
    return rows


def run():
    tasks = [t for t in CONFIG["Tasks"] if 0 < len(t) < len(GOALS)]
    if len(GOALS) > MAX_EXHAUSTIVE_GOALS:
        # Always keep every single-goal task -- they are the controlled slice,
        # and there are only |G| of them. Subsample the rest, which are what
        # the runtime actually goes on.
        sampled = proportional_sample([t for t in tasks if len(t) > 1],
                                      args.tasks_per_cardinality)
        tasks = [t for t in tasks if len(t) == 1] + sampled
    print(f"Evaluating {len(tasks)} tasks "
          f"({sum(1 for t in tasks if len(t) == 1)} single-goal)")

    cache = {}
    records = []
    levels = args.noise_levels if args.mode == "noise" else args.slip_probs
    for level in tqdm(levels, desc=f"{args.mode} levels"):
        for seed in range(args.seeds):
            if args.mode == "noise":
                rng = np.random.default_rng(seed)
                if not cache.get("_exact"):
                    u, e, bs = make_envs(0.0)
                    cache["_exact"] = (exact_EQ(u, args.horizon), exact_EQ(e, args.horizon),
                                       [exact_EQ(b, args.horizon) for b in bs])
                x_on, x_off, x_basis = cache["_exact"]
                EQ_on, EQ_off = perturb(x_on, level, rng), perturb(x_off, level, rng)
                EQ_basis = [perturb(b, level, rng) for b in x_basis]
                slip = 0.0
            else:
                np.random.seed(seed)
                u, e, bs = make_envs(level)
                train = lambda env: train_goal_oriented_until_convergence(
                    env, gamma=GAMMA, max_steps=args.max_steps,
                    check_every=args.max_steps, horizon=args.horizon)[0]
                EQ_on, EQ_off = train(u), train(e)
                EQ_basis = [train(b) for b in bs]
                slip = level

            for row in evaluate(tasks, EQ_on, EQ_off, EQ_basis, slip, cache,
                                f"{args.mode}={level} seed={seed}"):
                records.append({**row, "level": level, "seed": seed})

    return records


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
LABELS = {"onoff": "Goal-set composition (ours)", "boolean": "Original BTA composition",
          "excess": "Excess (BTA $-$ goal-set)"}
COLORS = {"onoff": "#1A5276", "boolean": "#C0560A", "excess": "#7D3C98"}
STYLES = {"onoff": "--", "boolean": "-", "excess": ":"}
SERIES = ["boolean", "onoff", "excess"]


def _style():
    rc("text", usetex=True)
    rc("font", family="serif")


def _mean_sem(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std() / max(len(values), 1) ** 0.5


def plot_single_goal(records, path, level_name):
    """Figure 1: the controlled slice -- k=1, only the NOT count varies."""
    _style()
    levels = sorted({r["level"] for r in records})
    fig, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        rows = [r for r in records if r["level"] == level and r["k"] == 1]
        xs = sorted({r["n_not"] for r in rows})
        for method in SERIES:
            means, sems = zip(*[_mean_sem([r[method] for r in rows if r["n_not"] == x])
                                for x in xs])
            means, sems = np.array(means), np.array(sems)
            ax.plot(xs, means, STYLES[method], marker="o", color=COLORS[method],
                    linewidth=2, markersize=6, label=LABELS[method])
            ax.fill_between(xs, means - sems, means + sems, color=COLORS[method], alpha=0.15)
        ax.set_xticks(xs)
        ax.set_xlabel(r"NOT operations in the composition", fontsize=14)
        ax.set_title(rf"{level_name} $= {level}$", fontsize=13)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel(r"Value gap $V^* - V^\pi$ (return)", fontsize=14)
    axes[0].legend(fontsize=11)
    fig.suptitle("Single-goal tasks: AND chain and task fixed, only NOT count varies",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_all_tasks(records, path, level_name, n_bins=6):
    """Figure 2: every task, binned by total operation count."""
    _style()
    levels = sorted({r["level"] for r in records})
    fig, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        rows = [r for r in records if r["level"] == level]
        ops = np.array([r["n_ops"] for r in rows], dtype=float)
        edges = np.unique(np.quantile(ops, np.linspace(0, 1, n_bins + 1)))
        centres, series = [], {m: ([], []) for m in SERIES}
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r in rows if lo <= r["n_ops"] <= hi]
            if not sel:
                continue
            centres.append(np.mean([r["n_ops"] for r in sel]))
            for method in series:
                mean, sem = _mean_sem([r[method] for r in sel])
                series[method][0].append(mean)
                series[method][1].append(sem)
        for method in SERIES:
            means, sems = np.array(series[method][0]), np.array(series[method][1])
            ax.plot(centres, means, STYLES[method], marker="o", color=COLORS[method],
                    linewidth=2, markersize=6, label=LABELS[method])
            ax.fill_between(centres, means - sems, means + sems,
                            color=COLORS[method], alpha=0.15)
        ax.set_xlabel(r"Composition operations (NOT $+$ AND $+$ OR)", fontsize=14)
        ax.set_title(rf"{level_name} $= {level}$", fontsize=13)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel(r"Value gap $V^* - V^\pi$ (return)", fontsize=14)
    axes[0].legend(fontsize=11)
    fig.suptitle("All evaluated tasks, binned by operation count", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
def main():
    run_id = (args.run_id or os.environ.get("RUN_ID")
              or os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
              or datetime.now().strftime("%Y%m%d-%H%M%S"))
    directory = os.path.join(SCRATCH_DIR, f"run_{run_id}", f"num_rooms_{args.num_rooms}")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "run_info.txt"), "w") as handle:
        handle.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
                     f"host: {socket.gethostname()}\n"
                     f"num_rooms: {args.num_rooms}\nmode: {args.mode}\n"
                     f"seeds: {args.seeds}\nnoise_levels: {args.noise_levels}\n"
                     f"slip_probs: {args.slip_probs}\nhorizon: {args.horizon}\n"
                     f"max_steps: {args.max_steps}\n")
    print(f"Run directory: {directory}")

    records = run()
    dd.io.save(os.path.join(directory, f"composition_depth_{args.num_rooms}.h5"), records)

    level_name = r"$\sigma$" if args.mode == "noise" else "$p$"
    suffix = f"{args.mode}_{args.num_rooms}"
    dirs = [directory] + ([args.figures_dir] if args.figures_dir else [])
    for out in dirs:
        os.makedirs(out, exist_ok=True)
        plot_single_goal(records, os.path.join(out, f"composition_depth_single_{suffix}.png"),
                         level_name)
        plot_all_tasks(records, os.path.join(out, f"composition_depth_all_{suffix}.png"),
                       level_name)
        print(f"Wrote figures to {out}")

    # Slope of the controlled slice: the headline number.
    print("\nControlled slice (single-goal tasks), mean gap by NOT count:")
    for level in sorted({r["level"] for r in records}):
        rows = [r for r in records if r["level"] == level and r["k"] == 1]
        xs = sorted({r["n_not"] for r in rows})
        for method in SERIES:
            means = [np.mean([r[method] for r in rows if r["n_not"] == x]) for x in xs]
            slope = np.polyfit(xs, means, 1)[0]
            print(f"  {level_name.strip('$')}={level:<5} {method:8s} "
                  + "  ".join(f"{x}NOT={m:.4f}" for x, m in zip(xs, means))
                  + f"   slope={slope:+.4f}/op")


if __name__ == "__main__":
    main()
