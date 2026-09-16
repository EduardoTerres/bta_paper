# A Goal-Set Characterization of Task Composition in the Boolean Task Algebra

Code for the paper.

Every experiment compares the same two zero-shot composition methods:

- `onoff` — goal-set composition (this paper), built from the universal and empty task value functions.
- `boolean` — the original Boolean Task Algebra composition, built from `ceil(log2|G|)` base tasks via AND/OR/NOT.

| Path | Environment |
| --- | --- |
| `boolean_composition/four_rooms/` | Rooms, tabular |
| `boolean_composition/boxman_sts/` | Boxman, pixels, DQN/UVFA |
| `skill_machines/` | Office, Skill Machines |

## Setup

Rooms and Boxman (Python 3.11, managed with `uv`):

```
cd boolean_composition
uv sync
```

Office (conda):

```
conda env create -f skill_machines/environment.yml
conda activate sm
```

## Before submitting jobs

Every `.sh` under `*/extension/scripts/` is a Slurm batch script written for one cluster. Edit, in each script you submit:

- the absolute `cd /home/eterrescaballe/bta_paper/...` line,
- `#SBATCH --partition`, `--gpus`, `--time`.

Edit `SCRATCH_DIR` (checkpoints, intermediate results, run directories) in:

- `four_rooms/extension/exp_stochastic_sweep.py`
- `four_rooms/extension/exp_composition_depth.py`
- `boxman_sts/extension/exp_stochastic_sweep.py`

`exp_train.py`, `exp_convergence.py`, `exp_train_all_tasks.py` and `exp2_comparison.py` take the map size from a `NUM_ROOMS` constant at the top of the file (4, 8 or 16), not from a flag.

All Python commands below are run from `boolean_composition/`. All `sbatch` commands are submitted from the repository root.

## Rooms

Train the universal, empty and base tasks, then evaluate both compositions:

```
uv run python four_rooms/extension/exp_train.py
```

Compositions at increasing levels of constituent optimality:

```
uv run python four_rooms/extension/exp_convergence.py
```

Train and cache the extended value function of every task:

```
uv run python four_rooms/extension/exp_train_all_tasks.py
```

Per-task comparison data:

```
uv run python four_rooms/extension/exp2_comparison.py
```

Paper figures 2 to 6:

```
uv run python four_rooms/extension/plots.py
```

### Stochastic sweep

Sweeps slip probability `p = 0.0, 0.1, ..., 0.9` and reports how fast each composition degrades as transitions stop being deterministic. Constituent value functions are trained to convergence with Goal-Oriented Q-learning under slip; ground truth and composed-policy values are computed by exact dynamic programming. Three seeds, one array task per seed.

```
sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=$(date +%Y%m%d-%H%M%S) \
  four_rooms/extension/scripts/run_stochastic_sweep.sh

sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=<same id> --dependency=afterany:<jobid> \
  four_rooms/extension/scripts/run_stochastic_sweep_aggregate.sh
```

The explicit `--export` is required: a `VAR=x sbatch ...` prefix does not reach the job, and the script silently falls back to its defaults. Pass the same `NUM_ROOMS` and `RUN_ID` to both stages. Without `RUN_ID`, the sweep names its run directory after the array job id and the aggregate step takes the most recently written run for that map.

`run_stochastic_sweep_4.sh`, `_8.sh` and `_16.sh` are fixed-map variants that hardcode `NUM_ROOMS` and do not set `RUN_ID`.

Without Slurm:

```
for s in 0 1 2; do
  RUN_ID=local uv run python four_rooms/extension/exp_stochastic_sweep.py --num-rooms 8 --seed $s
done
RUN_ID=local uv run python four_rooms/extension/exp_stochastic_sweep.py --num-rooms 8 --aggregate-results
```

Outputs, in `<SCRATCH_DIR>/run_<id>/num_rooms_<N>/` and copied into the repository:

```
four_rooms/extension/figures/stochastic_sweep_<N>.png
four_rooms/extension/figures/stochastic_sweep_gap_<N>.png
four_rooms/extension/figures/stochastic_sweep_returns_<N>.png
four_rooms/extension/figures/convergence_<N>.txt
exps_data_extension/stochastic_sweep_<N>.h5
```

Override the repository copies with `--figures-dir` / `--data-dir`, or pass `""` to either to write only into the run directory.

### Composition depth

Tests whether BTA's error grows with the number of composition operations, by perturbing exact constituents with Gaussian noise of scale `sigma` and binning tasks by operation count. One job per map; no separate aggregate step.

```
sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=$(date +%Y%m%d-%H%M%S) \
  four_rooms/extension/scripts/run_composition_depth.sh
```

The explicit `--export` is required: a `VAR=x sbatch ...` prefix does not reach the job, and the script silently falls back to its defaults.

Without Slurm:

```
uv run python four_rooms/extension/exp_composition_depth.py --num-rooms 8 --mode noise
uv run python four_rooms/extension/exp_composition_depth.py --num-rooms 8 --mode trained
```

## Boxman

### Convergence

```
sbatch boxman_sts/extension/scripts/run_exp_conv_parallel.sh
sbatch --dependency=afterany:<jobid> boxman_sts/extension/scripts/eval_exp_conv.sh
sbatch boxman_sts/extension/scripts/make_plots.sh
```

`run_exp_conv_parallel.sh` is the training array (3 runs x 4 tasks). `run_exp_convergence.sh`, `run_exp_conv_eval.sh` and `eval_exp_conv.sh` are evaluation only, over different checkpoint grids.

### Stochastic sweep

The function-approximation counterpart of the Rooms sweep: the same `p = 0.0, ..., 0.9` grid, with the four constituent UVFAs retrained from pixels at each slip level. Slip is added by a wrapper with the semantics of `GridWorld.pertube_action`; evaluation solves the exact finite-horizon MDP over the same `MaxLength` budget the rollouts use, while the policies evaluated are still the learned networks.

```
sbatch boxman_sts/extension/scripts/run_stochastic_sweep_train.sh
sbatch --dependency=afterany:<train_jobid> boxman_sts/extension/scripts/run_stochastic_sweep_eval.sh
sbatch --dependency=afterany:<eval_jobid>  boxman_sts/extension/scripts/run_stochastic_sweep_aggregate.sh
```

Grid: 3 seeds x 10 slip probabilities x 4 constituent UVFAs = 120 training jobs (array `0-119`, one network each). Throttle concurrency and resubmit subsets as needed; finished work is skipped through `model.done` markers, so re-running the array is safe:

```
sbatch --array=0-119%10 boxman_sts/extension/scripts/run_stochastic_sweep_train.sh
```

Array index maps to `seed = id / 40`, `slip = (id / 4) % 10`, `task = id % 4` over `(on, off, blue, square)`.

Submit the evaluation as the single `0-2` array, not as three separate jobs: the seeds must share `SLURM_ARRAY_JOB_ID` so their results land in one `run_<id>/` directory. Aggregation reads the most recent such directory, or the one named by `--run-id`. Slip probabilities whose constituents are not yet trained are skipped with a warning rather than failing the job.

Check training progress before evaluating:

```
python3 -c "
from pathlib import Path
R = Path('<SCRATCH_DIR>')
missing = [(s, p, t)
           for s in range(3)
           for p in [round(0.1 * i, 2) for i in range(10)]
           for t in ['on', 'off', 'blue', 'square']
           if not (R / f'seed_{s}' / f'slip_{p:.2f}' / t / 'model.done').exists()]
print(f'{120 - len(missing)}/120 trained'); print(missing[:12])
"
```

Outputs, in `<SCRATCH_DIR>/run_<id>/` and copied into the repository:

```
boxman_sts/plots/stochastic_sweep.png
boxman_sts/plots/stochastic_sweep_gap.png
boxman_sts/plots/stochastic_sweep_returns.png
boxman_sts/plots/stochastic_sweep_training.txt
boxman_sts/data/stochastic_sweep.h5
```

### Tests

```
sbatch boxman_sts/extension/scripts/run_tests.sh
```

Or locally:

```
uv run --with pytest python -m pytest boxman_sts/extension/
```

## Office

```
sbatch skill_machines/extension/office/scripts/train.sh
sbatch skill_machines/extension/office/scripts/eval.sh
sbatch skill_machines/extension/office/scripts/plots.sh
```

These scripts activate the `sm` conda environment and set `PYTHONPATH` themselves.

## Prior work

`boolean_composition/` extends the code released with "A Boolean Task Algebra For Reinforcement Learning" (https://arxiv.org/abs/2001.01394). New material lives under `four_rooms/extension/` and `boxman_sts/extension/`; the surrounding files are the original release.
