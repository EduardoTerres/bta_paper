"""Stochastic Boxman sweep -- the function-approximation counterpart of
`four_rooms/extension/exp_stochastic_sweep.py`.

Reviewer concern: the collapse result (goal-set composition == original BTA
composition, in expectation) is proved under deterministic transitions. The
Rooms sweep shows empirically how fast it degrades as that assumption is left,
but Rooms is tabular: every EQ there is a lookup table trained to convergence,
so the only thing the sweep can be measuring is the composition. The obvious
follow-up question is whether the same picture holds when the value functions
are neural networks trained from pixels, where approximation error and
composition error can interact.

This experiment runs the identical sweep on Boxman. `CollectEnv` has no slip
parameter, so `extension/mdp_utils.SlipAction` adds one with exactly the
semantics of `GridWorld.pertube_action`: with probability `slip_prob` the
intended move is replaced by a uniformly random *other* direction, while the
pick-up action ("stay") is never perturbed. For each slip probability we train
the four constituent UVFAs from scratch under that slip level --

  - "on"     : universal task (every object is a goal),
  - "off"    : empty task (no object is a goal),
  - "blue"   : base task, blue objects,
  - "square" : base task, square objects,

-- and then zero-shot compose the five evaluation tasks (B, S, B+S, B.S,
BxorS) two ways:

  - "onoff"   : goal-set composition (this paper's method), from on/off.
  - "boolean" : original BTA composition, from blue/square via AND/OR/NOT.

Evaluation is exact rather than Monte Carlo. A Boxman episode ends at the
*first* pick-up, so before that the state is just the player's cell; that makes
the env's dynamics small enough to rebuild in closed form (`mdp_utils`) and
solve by finite-horizon dynamic programming over the same `MaxLength` budget
the rollouts would have used. The *policies* being evaluated are still the
greedy policies of the learned networks, read off by rendering each cell and
running the DQN on it -- so this is a zero-variance version of "roll the
composed policy out and measure the return", not a different quantity.

Four numbers are recorded per task, all from the same DP solve:

  - `gap`     : V* - V^pi averaged over start states. The headline metric, and
                the plotted one. Units are episode return, so at rmin = -0.1
                per step a gap of 0.4 is "about four wasted steps' worth".
                Nothing is normalized, so nothing can blow up or flip sign.
  - `returns` : the raw expected episode return, with V* for reference, so the
                gap can be read against the values it came from.
  - `regret`  : (V* - V^pi) / (V* - V_worst), in [0, 1]. The gap expressed as a
                fraction of the range achievable at that slip probability,
                which matters because that range shrinks as slip rises.
                Recorded but not plotted.
  - `subopt`  : (V* - V^pi) / |V*|, the Rooms sweep's metric, recorded so the
                two experiments can be compared directly. Not plotted: it goes
                unstable past ~p=0.6, where a 20-step budget stops being enough
                to reach an object reliably and V* itself crosses zero.

Averaging is over start states uniformly, which is not an arbitrary choice --
`CollectEnv.reset` places the player uniformly over the free cells, so the mean
over states *is* the expected return under the actual initial distribution.
Post-pick-up states are excluded; they carry no policy decision and would only
dilute the average.

As in Rooms we also report each constituent UVFA's *own-task* value -- the
"on" net on the universal task, "off" on the empty task, "blue"/"square" on
their own base tasks, with no composition involved. That separates error the
composition introduced from error DQN training had already baked in at that
slip level, which is the main thing function approximation adds to the story.

Modes:
  - Train (`--train-only`): trains the constituent UVFAs for one seed and a
    subset of slip probabilities, checkpointing to `SCRATCH_DIR`. Sized to be
    run as a Slurm array over (seed, slip probability, task); see
    `scripts/run_stochastic_sweep_train.sh`.
  - Evaluate (`--evaluate-only`): loads one seed's checkpoints, runs the DP
    evaluation, and saves that seed's results. Results are written after every
    slip probability, so an interrupted job keeps what it finished and a rerun
    resumes from there.
  - Compute (default): train then evaluate, for one seed.
  - Aggregate (`--aggregate-results`): trains and evaluates nothing; loads
    every seed's saved results, averages across seeds (mean of each seed's own
    per-task mean, so error bars reflect seed-to-seed variance), and writes the
    figure + training table.
"""
import argparse
import faulthandler
import glob
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
faulthandler.enable(all_threads=True)

import deepdish as dd
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mdp_utils
from dqn import Agent, ComposedDQN_onoff, FloatTensor
from exp_convergence import (
    START_POSITIONS,
    TASKS,
    TRAIN_TASKS,
    boolean_composition,
    close_env,
    set_seed,
)
from gym_repoman.envs import CollectEnv
from plot_utils import (
    plot_stochastic_sweep,
    plot_stochastic_sweep_gap,
    plot_stochastic_sweep_returns,
    write_training_table,
)
from trainer import load, save
from wrappers import WarpFrame

np.object = object  # Hack to avoid error in save (kept consistent with the rest of the repo)

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
N_SEEDS = 3  # expected number of independent training runs to aggregate over
SLIP_PROBS = [round(0.1 * i, 2) for i in range(10)]  # 0.0 .. 0.9
SCRATCH_DIR = Path("/scratch-shared/eterrescaballe/boxman_stochastic/")

# Object positions in `goals.h5` order (BC, BS, bS, PS, bC, PC), asserted
# against the board at run time so a layout change cannot silently misalign
# goal images with cells.
EXPECTED_GOAL_POSITIONS = [(1, 8), (8, 1), (1, 1), (6, 3), (1, 7), (7, 7)]

# Which goal indices each constituent UVFA's *own* task accepts -- the
# uncomposed baselines. Mirrors the universal/empty/base-task lines in Rooms.
OWN_TASK_GOAL_INDICES = {
    "on": [0, 1, 2, 3, 4, 5],
    "off": [],
    "blue": [0, 1],
    "square": [1, 2, 3],
}
# How the own-task lines are grouped in the figure.
OWN_TASK_GROUPS = {"universal": ["on"], "empty": ["off"], "base_tasks": ["blue", "square"]}

METRICS = ["gap", "regret", "subopt", "returns", "rollout"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)),
                        help="Training seed for this run. Ignored with --aggregate-results.")
    parser.add_argument("--slip-probs", nargs="+", type=float, default=SLIP_PROBS,
                        help="Subset of the sweep to handle in this job (default: the whole sweep).")
    parser.add_argument("--train-task", nargs="+", choices=list(TRAIN_TASKS), default=None,
                        help="Subset of the constituent UVFAs to train in this job (default: all).")
    parser.add_argument("--max-timesteps", type=int, default=200_000,
                        help="DQN training budget per constituent UVFA.")
    parser.add_argument("--eps-timesteps", type=int, default=100_000,
                        help="Exploration schedule length passed to Agent.")
    parser.add_argument("--max-trajectory", type=int, default=20,
                        help="Episode budget (the MaxLength wrapper), and the DP horizon.")
    parser.add_argument("--rollout-episodes", type=int, default=200,
                        help="Episodes per (task, method) for the measured-return figure.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Names the run directory holding results and figures. Defaults "
                             "to $RUN_ID, then the Slurm array/job id, then a local timestamp. "
                             "Every array task of one submission must agree on this, or the "
                             "seeds land in different directories and cannot be aggregated.")
    parser.add_argument("--train-only", action="store_true",
                        help="Train the constituent UVFAs and stop, without evaluating.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training: evaluate this seed's existing checkpoints.")
    parser.add_argument("--aggregate-results", action="store_true",
                        help="Load every seed's saved results, average, and write the figure + table.")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Retrain even where a finished checkpoint already exists.")
    parser.add_argument("--force-evaluate", action="store_true",
                        help="Re-evaluate slip probabilities already present in the saved results.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DIR)
    # Figure/output/table paths default to the run directory (see
    # `run_directory`); passing them explicitly writes an extra copy there too.
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--table", type=Path, default=None)
    return parser.parse_args()


# ------------------------------------------------------------
# Run directory: one timestamped folder per submission
# ------------------------------------------------------------
def run_directory(args):
    """`scratch_dir/run_<id>`, holding the results, figures and table.

    Model checkpoints deliberately stay *outside* this folder, under
    `scratch_dir/seed_<n>/slip_<p>/<task>/`: training 120 UVFAs is the
    expensive half of the experiment and is worth reusing across evaluation
    runs, and `model.done` already makes that resumable. Only what an
    evaluation produces is timestamped, which is what makes a figure traceable
    to the job that drew it.

    The id is `--run-id`, else `$RUN_ID`, else the Slurm array job id (shared
    by every task of one array, which is what makes the seeds aggregatable),
    else a local wall-clock timestamp.
    """
    run_id = (
        args.run_id
        or os.environ.get("RUN_ID")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return args.scratch_dir / f"run_{run_id}"


def latest_run_directory(args):
    """The most recently modified run directory holding results.

    Lets aggregate mode follow the usual "evaluate the array, then aggregate"
    flow with no bookkeeping, while `--run-id` still pins a specific run.
    """
    if args.run_id or os.environ.get("RUN_ID"):
        return run_directory(args)
    candidates = [
        path for path in args.scratch_dir.glob("run_*")
        if list(path.glob("results_seed_*.h5"))
    ]
    if not candidates:
        return run_directory(args)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_run_info(args, directory):
    """A human-readable record of what produced this directory."""
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "seed": args.seed,
        "slip_probs": args.slip_probs,
        "max_timesteps": args.max_timesteps,
        "max_trajectory": args.max_trajectory,
        "rollout_episodes": args.rollout_episodes,
        "slurm_job": os.environ.get("SLURM_JOB_ID", "-"),
        "slurm_array_job": os.environ.get("SLURM_ARRAY_JOB_ID", "-"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"run_info_seed_{args.seed}.txt", "w") as handle:
        for key, value in info.items():
            handle.write(f"{key}: {value}\n")


# ------------------------------------------------------------
# Paths. One directory per (seed, slip probability, constituent task).
# ------------------------------------------------------------
def model_dir(args, seed, slip_prob, task_name):
    return args.scratch_dir / f"seed_{seed}" / f"slip_{slip_prob:.2f}" / task_name


def model_path(args, seed, slip_prob, task_name):
    return model_dir(args, seed, slip_prob, task_name) / "model.dqn"


def done_marker(args, seed, slip_prob, task_name):
    """Written only after training finishes.

    `Agent` checkpoints `model.dqn` throughout training, so the model file
    existing does not mean the run completed -- a preempted job leaves a
    perfectly loadable partial network behind. This marker is what the resume
    logic actually keys on.
    """
    return model_dir(args, seed, slip_prob, task_name) / "model.done"


def results_path(args, seed):
    return run_directory(args) / f"results_seed_{seed}.h5"


def training_path(args, seed):
    return run_directory(args) / f"training_seed_{seed}.h5"


def make_env(condition, slip_prob):
    """The training env: Boxman with slip, warped to 84x84.

    No `MaxLength` here, matching `trainer.learn` and `exp_convergence`:
    training episodes run until an object is picked up. The 20-step budget
    only applies at evaluation time, where it is the DP horizon.
    """
    return WarpFrame(
        mdp_utils.SlipAction(
            CollectEnv(start_positions=START_POSITIONS, goal_condition=condition),
            slip_prob=slip_prob,
        )
    )


def make_rollout_env(condition, slip_prob):
    """A plain (unwarped) Boxman env with slip, for measured-return episodes.

    No observation wrapper and no `MaxLength`: `mdp_utils.rollout_returns`
    picks actions from the precomputed per-cell greedy policy rather than from
    pixels, and applies the horizon itself, so neither is needed here.
    """
    return mdp_utils.SlipAction(
        CollectEnv(start_positions=START_POSITIONS, goal_condition=condition),
        slip_prob=slip_prob,
    )


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_constituent(args, seed, slip_prob, task_name):
    """Trains one constituent UVFA under this slip probability, or skips it."""
    marker = done_marker(args, seed, slip_prob, task_name)
    if marker.exists() and not args.force_retrain:
        print(f"[seed {seed}] {task_name} (p={slip_prob:.2f}) already trained, skipping.")
        return

    directory = model_dir(args, seed, slip_prob, task_name)
    directory.mkdir(parents=True, exist_ok=True)

    # Distinct seed per (seed, slip probability, task) so the constituents of
    # one composition are not trained from identical randomness.
    task_idx = list(TRAIN_TASKS).index(task_name)
    set_seed(seed * 1000 + int(round(slip_prob * 10)) * 10 + task_idx)

    env = make_env(TRAIN_TASKS[task_name], slip_prob)
    try:
        agent = Agent(
            env,
            max_timesteps=args.max_timesteps,
            eps_timesteps=args.eps_timesteps,
            path=str(directory) + "/",  # Agent checkpoints <dir>/model.dqn as it goes
        )
        agent.train()
        save(str(model_path(args, seed, slip_prob, task_name)), agent)
    finally:
        close_env(env)

    marker.write_text(f"{args.max_timesteps}\n")
    print(f"[seed {seed}] trained {task_name} (p={slip_prob:.2f}) for {args.max_timesteps} steps.")


def incomplete_constituents(args, seed, slip_prob):
    """Constituent UVFAs with no *finished* checkpoint for this (seed, p).

    Keyed on `model.done`, not `model.dqn`: `Agent` checkpoints throughout
    training, so a preempted or cancelled job leaves a perfectly loadable but
    under-trained network behind. Evaluating that would silently report a
    half-trained net as a result, which is worse than reporting nothing.
    """
    return [
        task_name
        for task_name in TRAIN_TASKS
        if not done_marker(args, seed, slip_prob, task_name).exists()
    ]


def load_constituents(args, seed, slip_prob):
    """Loads the four constituent UVFAs for one (seed, slip probability)."""
    dqns = {}
    for task_name, condition in TRAIN_TASKS.items():
        path = model_path(args, seed, slip_prob, task_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {path}. Train it first (see "
                "scripts/run_stochastic_sweep_train.sh)."
            )
        env = make_env(condition, slip_prob)
        try:
            dqn = load(str(path), env, map_location="cpu")
        finally:
            close_env(env)
        dqn.eval()
        if torch.cuda.is_available():
            dqn.cuda()
        dqns[task_name] = dqn
    return dqns


# ------------------------------------------------------------
# Exact evaluation
# ------------------------------------------------------------
def evaluate_policy(P, states, horizon, V_star, V_worst, policy,
                    rollout_env=None, n_episodes=0, rollout_seed=None):
    """Every metric for one (task, policy) pair.

    The DP metrics all come from a single solve. `rollout` is the sampled
    counterpart of `returns`: same policy, same slip, same horizon, but run in
    the real env rather than solved for, so the two agreeing is a check that
    `build_transition_model` matches the environment.
    """
    V_pi = mdp_utils.policy_evaluation(P, policy, horizon)
    metrics = {
        "gap": mdp_utils.value_gap(V_star, V_pi, states),
        "regret": mdp_utils.normalized_regret(V_star, V_worst, V_pi, states),
        "subopt": mdp_utils.normalized_suboptimality(V_star, V_pi, states),
        "returns": mdp_utils.expected_return(V_pi, states),
        "rollout": float("nan"),
    }
    if rollout_env is not None and n_episodes > 0:
        returns = mdp_utils.rollout_returns(
            rollout_env, policy, n_episodes, horizon, seed=rollout_seed
        )
        metrics["rollout"] = float(np.mean(returns))
    return metrics


def evaluate_slip_prob(args, seed, slip_prob, dqns, model_env, observations, goal_images):
    """Evaluates every composed and own task at one slip probability.

    Returns (metrics, training_records) where `metrics` maps
    metric -> method -> list of per-task values.
    """
    horizon = args.max_trajectory
    states = mdp_utils.start_states(model_env)
    positions = mdp_utils.goal_positions(model_env, expected=EXPECTED_GOAL_POSITIONS)
    goal_tensor = torch.from_numpy(np.asarray(goal_images)).type(FloatTensor)

    metrics = {metric: {} for metric in METRICS}

    def record(method, values):
        for metric in METRICS:
            metrics[metric].setdefault(method, []).append(values[metric])

    # --- The five composed tasks, both composition methods ---
    for index, (task_name, task) in enumerate(tqdm(
        TASKS.items(), desc=f"Composed tasks (p={slip_prob:.2f})", leave=False
    )):
        on_positions = [positions[i] for i in task["goal_indices"]]
        P = mdp_utils.build_transition_model(model_env, on_positions, slip_prob)
        V_star = mdp_utils.value_iteration(P, horizon)
        V_worst = mdp_utils.value_iteration(P, horizon, optimize=min)
        optimal_policy = mdp_utils.optimal_policy(P, horizon)

        composed = {
            "onoff": ComposedDQN_onoff(
                dqns["on"], dqns["off"],
                on_goals=[goal_images[i] for i in task["goal_indices"]],
            ),
            "boolean": boolean_composition(dqns["blue"], dqns["square"], task_name),
        }
        # One rollout env per task, shared by both methods and the reference.
        rollout_env = make_rollout_env(task["condition"], slip_prob)
        try:
            for offset, (method, dqn) in enumerate(composed.items()):
                policy = mdp_utils.greedy_policy(dqn, observations, goal_tensor)
                record(method, evaluate_policy(
                    P, states, horizon, V_star, V_worst, policy,
                    rollout_env=rollout_env, n_episodes=args.rollout_episodes,
                    rollout_seed=seed * 100_000 + index * 100 + offset * 10
                    + int(round(slip_prob * 10)),
                ))

            # Reference lines. The DP one stays the true optimum V*; the
            # rollout one has to come from an actual policy, so it uses the
            # stationary greedy policy (see `mdp_utils.optimal_policy`).
            metrics["returns"].setdefault("optimal", []).append(
                mdp_utils.expected_return(V_star, states)
            )
            metrics["rollout"].setdefault("optimal", []).append(
                float(np.mean(mdp_utils.rollout_returns(
                    rollout_env, optimal_policy, args.rollout_episodes, horizon,
                    seed=seed * 100_000 + index * 100 + 70 + int(round(slip_prob * 10)),
                )))
            )
        finally:
            close_env(rollout_env)

    # --- The constituent UVFAs on their own tasks, uncomposed ---
    training_records = []
    own_values = {}
    for index, (task_name, indices) in enumerate(OWN_TASK_GOAL_INDICES.items()):
        on_positions = [positions[i] for i in indices]
        P = mdp_utils.build_transition_model(model_env, on_positions, slip_prob)
        V_star = mdp_utils.value_iteration(P, horizon)
        V_worst = mdp_utils.value_iteration(P, horizon, optimize=min)
        policy = mdp_utils.greedy_policy(dqns[task_name], observations, goal_tensor)
        rollout_env = make_rollout_env(TRAIN_TASKS[task_name], slip_prob)
        try:
            own_values[task_name] = evaluate_policy(
                P, states, horizon, V_star, V_worst, policy,
                rollout_env=rollout_env, n_episodes=args.rollout_episodes,
                rollout_seed=seed * 100_000 + 90_000 + index * 100
                + int(round(slip_prob * 10)),
            )
        finally:
            close_env(rollout_env)
        training_records.append({
            "seed": seed,
            "slip_prob": slip_prob,
            "task": task_name,
            "steps": args.max_timesteps,
            "own_gap": own_values[task_name]["gap"],
            "own_regret": own_values[task_name]["regret"],
            "own_return": own_values[task_name]["returns"],
            "optimal_return": mdp_utils.expected_return(V_star, states),
        })

    for group, task_names in OWN_TASK_GROUPS.items():
        for task_name in task_names:
            record(group, own_values[task_name])

    print(
        f"[seed={seed}, slip={slip_prob:.2f}] value gap (V* - V^pi): "
        f"goal-set (ours)={np.mean(metrics['gap']['onoff']):.3f}, "
        f"BTA={np.mean(metrics['gap']['boolean']):.3f} | "
        f"return: ours={np.mean(metrics['returns']['onoff']):.3f}, "
        f"BTA={np.mean(metrics['returns']['boolean']):.3f}, "
        f"V*={np.mean(metrics['returns']['optimal']):.3f} | "
        f"rollout: ours={np.mean(metrics['rollout']['onoff']):.3f}, "
        f"BTA={np.mean(metrics['rollout']['boolean']):.3f}, "
        f"optimal={np.mean(metrics['rollout']['optimal']):.3f} | "
        f"own-task gap -- universal={own_values['on']['gap']:.3f}, "
        f"empty={own_values['off']['gap']:.3f}, "
        f"base tasks={0.5 * (own_values['blue']['gap'] + own_values['square']['gap']):.3f}"
    )
    return metrics, training_records


def load_partial(path):
    """Whatever a previous (possibly interrupted) run of this seed saved."""
    if not path.exists():
        return {}
    try:
        return dd.io.load(str(path))
    except Exception as error:  # a job killed mid-write leaves an unreadable file
        print(f"[WARN] Could not read {path} ({error}); starting this seed from scratch.")
        return {}


def run_evaluate(args):
    seed = args.seed
    directory = run_directory(args)
    directory.mkdir(parents=True, exist_ok=True)
    write_run_info(args, directory)
    print(f"Run directory: {directory}")

    results = load_partial(results_path(args, seed))
    training = load_partial(training_path(args, seed))
    if not isinstance(training, dict):
        training = {}
    skipped = []  # (slip probability, constituents still untrained)

    # Rendering is independent of slip and of the task, so it is done once.
    model_env = WarpFrame(CollectEnv(start_positions=START_POSITIONS))
    try:
        observations = mdp_utils.render_observations(model_env)
        goal_images = dd.io.load(str(ROOT / "goals.h5"))

        for slip_prob in tqdm(args.slip_probs, desc=f"[seed {seed}] Slip probability sweep"):
            key = f"{slip_prob:.2f}"
            if key in results and not args.force_evaluate:
                print(f"[seed {seed}] p={key} already evaluated, skipping.")
                continue
            untrained = incomplete_constituents(args, seed, slip_prob)
            if untrained:
                # A partially trained grid is the normal state on a preemptible
                # queue, so skip and carry on rather than abandoning the slip
                # probabilities that *are* ready. The summary below says what
                # was left out, and a rerun picks these up once they exist.
                print(
                    f"[WARN] [seed {seed}] p={key}: not trained yet "
                    f"({', '.join(untrained)}); skipping this slip probability."
                )
                skipped.append((key, untrained))
                continue
            dqns = load_constituents(args, seed, slip_prob)
            metrics, records = evaluate_slip_prob(
                args, seed, slip_prob, dqns, model_env, observations, goal_images
            )
            results[key] = metrics
            training[key] = records

            # Save after every slip probability: a preempted job keeps what it
            # finished, and a rerun picks up where it stopped.
            dd.io.save(str(results_path(args, seed)), results)
            dd.io.save(str(training_path(args, seed)), training)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        close_env(model_env)

    print(f"Saved results to {results_path(args, seed)}")
    print(f"Saved training records to {training_path(args, seed)}")
    print(
        f"[seed {seed}] Evaluated {len(results)} of {len(args.slip_probs)} "
        "requested slip probabilities."
    )
    if skipped:
        print(f"[seed {seed}] Skipped, still untrained:")
        for key, untrained in skipped:
            print(f"    p={key}: {', '.join(untrained)}")
        print(
            "Finish those with run_stochastic_sweep_train.sh, then rerun this "
            "script with the same --run-id; slip probabilities already in the "
            "results file are cached and will not be recomputed."
        )


def run_train(args):
    task_names = args.train_task or list(TRAIN_TASKS)
    for slip_prob in args.slip_probs:
        for task_name in task_names:
            train_constituent(args, args.seed, slip_prob, task_name)


# ------------------------------------------------------------
# Aggregate mode: load every seed's saved results, average, plot
# ------------------------------------------------------------
def run_aggregate(args):
    directory = latest_run_directory(args)
    pattern = str(directory / "results_seed_*.h5")
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        raise FileNotFoundError(
            f"No saved results found matching {pattern}. Run the train and evaluate "
            "modes for each seed first, or pass --run-id to point at a specific run."
        )
    print(f"Run directory: {directory}")
    if len(result_files) != N_SEEDS:
        print(
            f"[WARN] Expected {N_SEEDS} seeds but found {len(result_files)} result "
            f"file(s): {result_files}. Aggregating over what's available."
        )

    per_seed = [dd.io.load(f) for f in result_files]
    slip_keys = sorted(set.intersection(*[set(res) for res in per_seed]), key=float)
    if not slip_keys:
        raise ValueError("The saved seeds have no slip probability in common.")
    for path, res in zip(result_files, per_seed):
        missing = sorted(set(slip_keys) - set(res), key=float)
        if missing:
            print(f"[WARN] {path} is missing slip probabilities {missing}.")

    # Average *across seeds*: one scalar per seed (that seed's own mean over
    # its tasks), so the plotted error bars reflect seed-to-seed variance.
    aggregated = {}
    for metric in METRICS:
        methods = sorted({m for res in per_seed for m in res[slip_keys[0]][metric]})
        aggregated[metric] = {
            float(key): {
                method: [
                    float(np.mean(res[key][metric][method]))
                    for res in per_seed
                    if key in res and method in res[key][metric]
                ]
                for method in methods
            }
            for key in slip_keys
        }

    # Everything lands in the run directory; --output/--figure/--table write
    # an extra copy wherever they point (e.g. into the repo's plots/ folder).
    outputs = [directory / "stochastic_sweep.h5"] + ([args.output] if args.output else [])
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        dd.io.save(str(path), aggregated)

    figures = [
        ("stochastic_sweep", plot_stochastic_sweep),
        ("stochastic_sweep_gap", plot_stochastic_sweep_gap),
        ("stochastic_sweep_returns", plot_stochastic_sweep_returns),
    ]
    for name, plot_fn in figures:
        targets = [directory / f"{name}.png"]
        if args.figure:
            # Keep the caller's stem for the headline figure and suffix the
            # other two, so an explicit --figure still names what it named.
            suffix = name.replace("stochastic_sweep", "")
            targets.append(args.figure.with_name(args.figure.stem + suffix + args.figure.suffix))
        for target in targets:
            plot_fn(aggregated, target)
            print(f"Wrote {target}")

    training_records = [
        record
        for path in sorted(glob.glob(str(directory / "training_seed_*.h5")))
        for records in dd.io.load(path).values()
        for record in records
    ]
    if training_records:
        for path in [directory / "stochastic_sweep_training.txt"] + (
            [args.table] if args.table else []
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            write_training_table(training_records, path)
    else:
        print("[WARN] No training records found; skipping the training table.")

    print(f"Aggregated {len(result_files)} seed(s) into {directory}")


def main():
    args = parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if args.aggregate_results:
        run_aggregate(args)
        return

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if not args.evaluate_only:
        run_train(args)
    if not args.train_only:
        run_evaluate(args)


if __name__ == "__main__":
    main()
