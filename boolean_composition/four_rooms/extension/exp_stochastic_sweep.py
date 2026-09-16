"""Stochastic Rooms sweep.

Reviewer concern: the collapse result (goal-set composition == original BTA
composition, in expectation) is proved under deterministic transitions; the
counterexample shows it can fail under stochasticity, but there was no
empirical evidence of *how fast* it fails as that assumption is left.

This experiment adds slip to the Rooms env (`GridWorld` already supports a
`slip_prob` parameter: with probability `slip_prob` the intended action is
replaced by a uniformly random *other* direction, see
`GridWorld.pertube_action`) and sweeps `slip_prob` over `SLIP_PROBS`. For a
fixed set of composed tasks, we compare:

  - "onoff"   : Goal-set composition (this paper's method) -- built from the
                universal + empty task EQs.
  - "boolean" : Original BTA composition -- built from the
                ceil(log2(|G|)) base task EQs via AND/OR/NOT.
  - "V*"      : ground truth, exact value iteration on the composed task.

Three metrics are recorded for every (task, method), and each gets its own
figure. All three describe the same policies on the same tasks; they differ
only in how the number is reported.

  1. `subopt`  -- (V* - V^pi) / |V*|, averaged over start states. The sweep's
                  original normalized view, kept so its figure stays
                  comparable. Read it with care: the per-state denominator
                  passes through zero on tasks with few goals, so a handful of
                  states can dominate the mean.
  2. `gap`     -- V* - V^pi in units of episode return, no normalization.
                  Same comparison, nothing that can blow up or flip sign; at a
                  step cost of 0.1 a gap of 0.4 is "four wasted steps' worth".
  3. `rollout` -- the mean return of actual evaluation episodes, run in the
                  real env under slip rather than solved for. The Monte Carlo
                  counterpart of the other two: agreement with `gap` is also a
                  check that `mdp_utils.build_transition_model` really matches
                  the environment `GridWorld.step` samples from.

The DP metrics use exact dynamic programming (`mdp_utils.value_iteration` /
`mdp_utils.policy_evaluation`) over a fixed `--horizon`, the same episode cap
the rollouts run under (and the one `utils.evaluate` has always used). The
horizon is not a detail: undiscounted evaluation of a policy that never
reaches a goal -- which composed policies do become under slip -- diverges,
and an iteration-capped solve would silently report the cap rather than a
property of the policy.

We additionally report the *own-task* value of each constituent EQ -- the
universal EQ on the universal task, the empty EQ on the empty task, each base
EQ on its own base task -- with no composition involved, under all three
metrics. This separates suboptimality already present from imperfect
Goal-Oriented Q-learning under slip (which grows with `slip_prob` regardless
of composition) from suboptimality introduced by the composition step itself.
Each constituent EQ is trained via
`mdp_utils.train_goal_oriented_until_convergence` up to `--max-steps` steps;
how many steps that took is recorded per (seed, slip_prob, task) and written
out as a convergence table.

Everything this script produces -- learned EQs, per-seed results, convergence
records, and the three figures -- is written under a single timestamped run
directory inside `SCRATCH_DIR`, so a run's outputs can always be traced back
to the job that made them. See `run_directory`.

Two modes, selected by `--aggregate-results`:
  - Compute (default): trains everything for one seed (`--seed`) and saves the
    learned EQs, evaluation results, and convergence records after *every*
    slip probability -- so an interrupted job keeps what it finished. Meant to
    be run once per seed (e.g. as a Slurm array job, see
    `scripts/run_stochastic_sweep.sh`).
  - Aggregate (`--aggregate-results`): trains nothing; loads every seed's
    saved results from the run directory, averages *across seeds* (mean of
    each seed's own per-task mean, so the plotted error bars reflect
    seed-to-seed variance), and writes the three figures + convergence table.
"""
import argparse
import glob
import os
import random
import socket
from datetime import datetime

import deepdish as dd
import numpy as np
from tqdm import tqdm

from four_rooms.GridWorld import GridWorld
from four_rooms.library import EQ_P
from four_rooms.config import Config_4, Config_8, Config_16
from four_rooms.extension.utils import (
    get_composed_tasks,
    proportional_sample,
    convert_defaultdict_to_dict,
    write_convergence_table,
)
from four_rooms.extension.mdp_utils import (
    build_states,
    build_transition_model,
    value_iteration,
    policy_evaluation,
    normalized_suboptimality,
    value_gap,
    expected_return,
    rollout_returns,
    train_goal_oriented_until_convergence,
)
from four_rooms.extension.plot_utils import (
    plot_stochastic_sweep,
    plot_stochastic_sweep_gap,
    plot_stochastic_sweep_returns,
)

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
np.object = object  # Hack to avoid error in save (kept consistent with the rest of the repo)

N_SEEDS = 3  # expected number of independent training runs to aggregate over
SCRATCH_DIR = "/scratch-shared/eterrescaballe/stochastic/"

SLIP_PROBS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
GAMMA = 1.0  # matches Goal_Oriented_Q_learning's default (undiscounted)

# Metrics recorded per (task, method). One figure each for subopt/gap/rollout.
METRICS = ["subopt", "gap", "returns", "rollout"]

# All 2^|G| - 2 non-trivial subsets when |G| is small; otherwise sample.
MAX_EXHAUSTIVE_GOALS = 6  # 2^6 - 2 = 62 tasks
NUM_TASK_SAMPLES = 50

# How the own-task (uncomposed) constituent EQs are grouped in the figures.
OWN_TASK_GROUPS = {"universal": "universal", "empty": "empty"}  # base tasks handled separately

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num-rooms",
    type=int,
    default=int(os.environ.get("NUM_ROOMS", 4)),
    choices=[4, 8, 16],
    help="Which Rooms map/goal set to use (default: 4, or $NUM_ROOMS if set).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=int(os.environ.get("SEED", 0)),
    help="Training seed for this run (default: 0, or $SEED if set). Ignored "
    "with --aggregate-results.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=int(os.environ.get("MAX_STEPS", 200_000)),
    help="Per-EQ training step budget. Trained in full -- there is no early "
    "stop by default -- so this directly sets how converged the per-goal "
    "values are. 200k reaches a slip-0 composed gap of 0.0000 on the 4-room "
    "map at alpha=0.1; larger goal sets need more.",
)
parser.add_argument(
    "--check-every",
    type=int,
    default=25_000,
    help="How often (in steps) to record each EQ's own-task suboptimality. "
    "Reporting only unless --tol is set, so this is pure overhead -- kept "
    "coarse.",
)
parser.add_argument(
    "--tol",
    type=float,
    default=None,
    help="Optional early-stopping threshold on normalized suboptimality. Off "
    "by default, and should stay off: the check measures the max-over-goals "
    "greedy policy, while both composition methods consume the per-goal "
    "values, which converge much later. Stopping on it silently breaks the "
    "slip-0 sanity check (see train_goal_oriented_until_convergence).",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=100,
    help="Episode step cap. Used both as the DP horizon and as the rollout "
    "cap, so the solved and sampled numbers describe the same quantity "
    "(default: 100, matching utils.evaluate).",
)
parser.add_argument(
    "--rollout-episodes",
    type=int,
    default=200,
    help="Episodes per (task, method) for the measured-return figure "
    "(default: 200).",
)
parser.add_argument(
    "--run-id",
    type=str,
    default=None,
    help="Names the run directory under SCRATCH_DIR. Defaults to $RUN_ID, "
    "then the Slurm array/job id, then a local timestamp. Every array task of "
    "one submission must agree on this, or the seeds land in different "
    "directories and cannot be aggregated together.",
)
parser.add_argument(
    "--figures-dir",
    type=str,
    default="four_rooms/extension/figures",
    help="Repo directory to copy the aggregated figures and convergence table "
    "into, alongside the canonical copy in the run directory. Relative paths "
    "resolve against the working directory, which the submit scripts set to "
    "the repo root. Pass an empty string to write only to the run directory.",
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="exps_data_extension",
    help="Repo directory to copy the aggregated .h5 into. Empty string to "
    "write only to the run directory.",
)
parser.add_argument(
    "--aggregate-results",
    action="store_true",
    help="Skip training: load every seed's saved results from the run "
    "directory, average across seeds, and (re)generate the figures + table.",
)
args, _ = parser.parse_known_args()

NUM_ROOMS = args.num_rooms  # |Goals| = 4/8/16 -> 2^|Goals| goal subsets
configs = {
    4: Config_4,
    8: Config_8,
    16: Config_16,
}
config = configs[NUM_ROOMS]

terminal_states = config["T_states"]
goals = config["Goals"]
tasks = config["Tasks"]
base_tasks = config["Bases"]
composition_rules = config["Composition_rules"]

# Remove the universal (all goals) and empty tasks: they are the two EQs the
# "onoff" method is built from, not a composition to evaluate.
other_tasks = [task for task in tasks if not (len(task) == len(goals) or len(task) == 0)]


# ------------------------------------------------------------
# Run directory: one timestamped folder per submission
# ------------------------------------------------------------
def run_directory():
    """`SCRATCH_DIR/run_<id>`, holding every artefact this run produces.

    The id is `--run-id`, else `$RUN_ID`, else the Slurm array job id (shared
    by every task of one array, which is what makes the seeds aggregatable),
    else a local wall-clock timestamp. `scripts/run_stochastic_sweep.sh`
    exports `RUN_ID` so the folder is named by submission time.
    """
    run_id = (
        args.run_id
        or os.environ.get("RUN_ID")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return os.path.join(SCRATCH_DIR, f"run_{run_id}", f"num_rooms_{NUM_ROOMS}")


def latest_run_directory():
    """The most recently modified run directory holding results for this map.

    Used by aggregate mode when no `--run-id` is given, so the usual
    "train the array, then aggregate" flow needs no bookkeeping.
    """
    if args.run_id or os.environ.get("RUN_ID"):
        return run_directory()
    candidates = glob.glob(os.path.join(SCRATCH_DIR, "run_*", f"num_rooms_{NUM_ROOMS}"))
    candidates = [c for c in candidates if glob.glob(os.path.join(c, "results_seed_*.h5"))]
    if not candidates:
        return run_directory()
    return max(candidates, key=os.path.getmtime)


def write_run_info(directory):
    """A human-readable record of what produced this directory."""
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "num_rooms": NUM_ROOMS,
        "seed": args.seed,
        "slip_probs": SLIP_PROBS,
        "max_steps": args.max_steps,
        "horizon": args.horizon,
        "rollout_episodes": args.rollout_episodes,
        "slurm_job": os.environ.get("SLURM_JOB_ID", "-"),
        "slurm_array_job": os.environ.get("SLURM_ARRAY_JOB_ID", "-"),
    }
    with open(os.path.join(directory, f"run_info_seed_{args.seed}.txt"), "w") as handle:
        for key, value in info.items():
            handle.write(f"{key}: {value}\n")


def _train_and_record(env, task_name, slip_prob, convergence_records):
    """Trains one EQ to convergence and appends its convergence record."""
    learned_EQ, _, convergence = train_goal_oriented_until_convergence(
        env,
        gamma=GAMMA,
        max_steps=args.max_steps,
        check_every=args.check_every,
        tol=args.tol,
        horizon=args.horizon,
    )
    convergence_records.append(
        {
            "seed": args.seed,
            "slip_prob": slip_prob,
            "task": task_name,
            "steps": convergence["steps"],
            "converged": convergence["converged"],
            "final_subopt": convergence["final_subopt"],
        }
    )
    if not convergence["converged"]:
        print(
            f"  [INFO] {task_name} (p={slip_prob}) trained the full "
            f"{args.max_steps} steps (own-task subopt={100 * convergence['final_subopt']:.2f}%)"
        )
    return learned_EQ, convergence


def evaluate_policy(states, P, policy, task_env, V_star, rollout_seed):
    """Every metric for one (task, policy) pair.

    The DP numbers and the rollout numbers describe the same thing under the
    same `--horizon`: `policy_evaluation(..., horizon=H)` is the exact
    expected H-step return, and `rollout_returns(..., horizon=H)` samples it.
    """
    V_pi = policy_evaluation(
        states, P, policy, task_env.action_space.n, gamma=GAMMA, horizon=args.horizon
    )
    returns = rollout_returns(
        task_env, policy, args.rollout_episodes, args.horizon, seed=rollout_seed
    )
    return {
        "subopt": normalized_suboptimality(V_star, V_pi, states),
        "gap": value_gap(V_star, V_pi, states),
        "returns": expected_return(V_pi, states),
        "rollout": float(np.mean(returns)),
    }


def evaluate_slip_prob(slip_prob, learned_universal_EQ, learned_empty_EQ,
                       learned_base_tasks_EQs, sampled_other_tasks):
    """Every composed and own task at one slip probability.

    Returns metric -> method -> list of per-task values.
    """
    metrics = {metric: {} for metric in METRICS}

    def record(method, values):
        for metric in METRICS:
            metrics[metric].setdefault(method, []).append(values[metric])

    EQs_composed, _ = get_composed_tasks(
        tasks=sampled_other_tasks,
        goals=goals,
        EQ_on=learned_universal_EQ,
        EQ_off=learned_empty_EQ,
        EQ_basis=learned_base_tasks_EQs,
        composition_rules=composition_rules,
    )

    # --- The composed tasks, both composition methods ---
    for index, (task, EQs) in enumerate(
        tqdm(EQs_composed.items(), desc=f"Evaluating tasks (p={slip_prob})", leave=False)
    ):
        task_goals = [[pos, pos] for pos in task]
        task_env = GridWorld(
            MAP="MAP_" + str(NUM_ROOMS),
            goals=task_goals,
            T_states=terminal_states,
            slip_prob=slip_prob,
        )
        states = build_states(task_env)
        P = build_transition_model(task_env)
        V_star, _, optimal_policy = value_iteration(
            states, P, task_env.action_space.n, gamma=GAMMA, horizon=args.horizon
        )

        for method in ["onoff", "boolean"]:
            record(
                method,
                evaluate_policy(
                    states, P, EQ_P(EQs[method]), task_env, V_star,
                    rollout_seed=args.seed * 100_000 + index * 10 + int(slip_prob * 100),
                ),
            )

        # The optimal policy's own rollouts, as the reference line of the
        # measured-return figure. Its DP gap is 0 by construction.
        record(
            "optimal",
            evaluate_policy(
                states, P, optimal_policy, task_env, V_star,
                rollout_seed=args.seed * 100_000 + index * 10 + int(slip_prob * 100) + 7,
            ),
        )

    # --- The constituent EQs on their own tasks, uncomposed ---
    own_envs = [
        ("universal", learned_universal_EQ,
         dict(goals=terminal_states)),
        ("empty", learned_empty_EQ,
         dict(goals=terminal_states, goal_reward=-0.1)),
    ] + [
        ("base_tasks", eq,
         dict(goals=[[pos, pos] for pos in task], T_states=terminal_states))
        for eq, task in zip(learned_base_tasks_EQs, base_tasks)
    ]

    for index, (group, learned_EQ, env_kwargs) in enumerate(own_envs):
        own_env = GridWorld(MAP="MAP_" + str(NUM_ROOMS), slip_prob=slip_prob, **env_kwargs)
        states = build_states(own_env)
        P = build_transition_model(own_env)
        V_star, _, _ = value_iteration(
            states, P, own_env.action_space.n, gamma=GAMMA, horizon=args.horizon
        )
        record(
            group,
            evaluate_policy(
                states, P, EQ_P(learned_EQ), own_env, V_star,
                rollout_seed=args.seed * 100_000 + 90_000 + index * 10 + int(slip_prob * 100),
            ),
        )

    return metrics


# ------------------------------------------------------------
# Compute mode: train one seed, save raw results, no plotting
# ------------------------------------------------------------
def run_compute():
    random.seed(args.seed)
    np.random.seed(args.seed)

    if len(goals) > MAX_EXHAUSTIVE_GOALS:
        sampled_other_tasks = proportional_sample(other_tasks, NUM_TASK_SAMPLES)
    else:
        sampled_other_tasks = other_tasks

    directory = run_directory()
    os.makedirs(directory, exist_ok=True)
    write_run_info(directory)
    print(f"Run directory: {directory}")

    results_path = os.path.join(directory, f"results_seed_{args.seed}.h5")
    eqs_path = os.path.join(directory, f"eqs_seed_{args.seed}.h5")
    convergence_path = os.path.join(directory, f"convergence_seed_{args.seed}.h5")

    results = {}  # slip_prob -> metric -> method -> list of per-task values
    eqs_by_slip = {}  # slip_prob -> {"universal", "empty", "base_tasks": [...]}
    convergence_records = []

    for slip_prob in tqdm(SLIP_PROBS, desc=f"[seed {args.seed}] Slip probability sweep"):
        # --- Train universal / empty / base task EQs under this slip probability ---
        train_pbar = tqdm(
            total=2 + len(base_tasks),
            desc=f"Training tasks (p={slip_prob})",
            leave=False,
        )

        env = GridWorld(MAP="MAP_" + str(NUM_ROOMS), goals=terminal_states, slip_prob=slip_prob)
        learned_universal_EQ, _ = _train_and_record(
            env, "universal", slip_prob, convergence_records
        )
        train_pbar.update(1)

        env = GridWorld(
            MAP="MAP_" + str(NUM_ROOMS), goals=terminal_states, goal_reward=-0.1, slip_prob=slip_prob
        )
        learned_empty_EQ, _ = _train_and_record(env, "empty", slip_prob, convergence_records)
        train_pbar.update(1)

        learned_base_tasks_EQs = []
        for i, task in enumerate(base_tasks):
            task_goals = [[pos, pos] for pos in task]
            env = GridWorld(
                MAP="MAP_" + str(NUM_ROOMS),
                goals=task_goals,
                T_states=terminal_states,
                slip_prob=slip_prob,
            )
            learned_EQ, _ = _train_and_record(
                env, f"base_task_{i}", slip_prob, convergence_records
            )
            learned_base_tasks_EQs.append(learned_EQ)
            train_pbar.update(1)

        train_pbar.close()

        eqs_by_slip[slip_prob] = {
            "universal": convert_defaultdict_to_dict(learned_universal_EQ),
            "empty": convert_defaultdict_to_dict(learned_empty_EQ),
            "base_tasks": [convert_defaultdict_to_dict(eq) for eq in learned_base_tasks_EQs],
        }

        metrics = evaluate_slip_prob(
            slip_prob,
            learned_universal_EQ,
            learned_empty_EQ,
            learned_base_tasks_EQs,
            sampled_other_tasks,
        )
        results[slip_prob] = metrics

        print(
            f"[seed={args.seed}, slip={slip_prob}] "
            f"gap (V*-V^pi): ours={np.mean(metrics['gap']['onoff']):.3f}, "
            f"BTA={np.mean(metrics['gap']['boolean']):.3f} | "
            f"subopt: ours={100 * np.mean(metrics['subopt']['onoff']):.2f}%, "
            f"BTA={100 * np.mean(metrics['subopt']['boolean']):.2f}% | "
            f"rollout return: ours={np.mean(metrics['rollout']['onoff']):.3f}, "
            f"BTA={np.mean(metrics['rollout']['boolean']):.3f}, "
            f"optimal={np.mean(metrics['rollout']['optimal']):.3f} | "
            f"own-task gap: universal={np.mean(metrics['gap']['universal']):.3f}, "
            f"empty={np.mean(metrics['gap']['empty']):.3f}, "
            f"base={np.mean(metrics['gap']['base_tasks']):.3f}"
        )

        # Save after every slip probability, not once at the end: an
        # interrupted job then keeps everything it finished.
        dd.io.save(results_path, results)
        dd.io.save(eqs_path, eqs_by_slip)
        dd.io.save(convergence_path, convergence_records)

    print(f"Saved results to {results_path}")
    print(f"Saved learned EQs to {eqs_path}")
    print(f"Saved convergence records to {convergence_path}")


# ------------------------------------------------------------
# Aggregate mode: load every seed's saved results, average, plot
# ------------------------------------------------------------
def _output_dirs(run_dir, repo_dir):
    """Where one aggregated artefact should be written.

    Always the run directory; plus `repo_dir` when it is set, created if
    needed. Returned in that order so the run directory stays the copy that
    exists even if the repo path is unwritable from a compute node.
    """
    dirs = [run_dir]
    if repo_dir:
        os.makedirs(repo_dir, exist_ok=True)
        dirs.append(repo_dir)
    return dirs


def run_aggregate():
    directory = latest_run_directory()
    pattern = os.path.join(directory, "results_seed_*.h5")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        raise FileNotFoundError(
            f"No saved results found matching {pattern}. Run the compute mode "
            "(without --aggregate-results) for each seed first, or pass "
            "--run-id to point at a specific run."
        )
    print(f"Run directory: {directory}")
    if len(result_files) != N_SEEDS:
        print(
            f"[WARN] Expected {N_SEEDS} seeds but found {len(result_files)} result "
            f"file(s): {result_files}. Aggregating over what's available."
        )

    per_seed_results = [dd.io.load(f) for f in result_files]
    slip_probs = sorted(set.intersection(*[set(res) for res in per_seed_results]))
    if not slip_probs:
        raise ValueError("The saved seeds have no slip probability in common.")
    for path, res in zip(result_files, per_seed_results):
        missing = sorted(set(slip_probs) - set(res))
        if missing:
            print(f"[WARN] {path} is missing slip probabilities {missing}.")

    # Average *across seeds*: one scalar per seed (that seed's own mean over
    # its tasks/base tasks), then the plot's mean/SEM is taken over those
    # per-seed scalars -- i.e. error bars reflect seed-to-seed variance.
    aggregated = {}
    for metric in METRICS:
        methods = sorted({m for res in per_seed_results for m in res[slip_probs[0]][metric]})
        aggregated[metric] = {
            slip_prob: {
                method: [
                    float(np.mean(res[slip_prob][metric][method]))
                    for res in per_seed_results
                    if slip_prob in res and method in res[slip_prob][metric]
                ]
                for method in methods
            }
            for slip_prob in slip_probs
        }

    # The run directory is the canonical copy -- it is the one that can be
    # traced back to the job that made it. The repo directories get a copy so
    # the paper's figures are where the rest of the repo expects them.
    for data_dir in _output_dirs(directory, args.data_dir):
        path = os.path.join(data_dir, f"stochastic_sweep_{NUM_ROOMS}.h5")
        dd.io.save(path, aggregated)
        print(f"Wrote {path}")

    figures = [
        ("subopt", plot_stochastic_sweep, f"stochastic_sweep_{NUM_ROOMS}.png"),
        ("gap", plot_stochastic_sweep_gap, f"stochastic_sweep_gap_{NUM_ROOMS}.png"),
        ("rollout", plot_stochastic_sweep_returns, f"stochastic_sweep_returns_{NUM_ROOMS}.png"),
    ]
    figure_dirs = _output_dirs(directory, args.figures_dir)
    for metric, plot_fn, name in figures:
        for figure_dir in figure_dirs:
            path = os.path.join(figure_dir, name)
            plot_fn(aggregated[metric], save_name=path)
            print(f"Wrote {path}")

    convergence_files = sorted(glob.glob(os.path.join(directory, "convergence_seed_*.h5")))
    convergence_records = [r for f in convergence_files for r in dd.io.load(f)]
    if convergence_records:
        for figure_dir in figure_dirs:
            write_convergence_table(
                convergence_records,
                save_name=os.path.join(figure_dir, f"convergence_{NUM_ROOMS}.txt"),
            )
    else:
        print("[WARN] No convergence records found; skipping convergence.txt")

    print(f"Aggregated {len(result_files)} seed(s) into {directory}")


if __name__ == "__main__":
    if args.aggregate_results:
        run_aggregate()
    else:
        run_compute()
