# A Goal-Set Characterization of Task Composition in the Boolean Task Algebra

Code for the paper. Rooms and Boxman are under `boolean_composition/`, Office under `skill_machines/`.

## Setup

```
cd boolean_composition && uv sync          # Rooms, Boxman (Python 3.11)
conda env create -f skill_machines/environment.yml && conda activate sm   # Office
```

All `uv run` commands are run from `boolean_composition/`, all `sbatch` commands from the repository root.

The `.sh` files are Slurm scripts. Before submitting, edit the `cd /home/eterrescaballe/bta_paper/...` line and the `#SBATCH` resources in each, and `SCRATCH_DIR` in `four_rooms/extension/exp_stochastic_sweep.py`, `four_rooms/extension/exp_composition_depth.py` and `boxman_sts/extension/exp_stochastic_sweep.py`.

## Rooms

```
uv run python four_rooms/extension/exp_train.py            # train tasks, evaluate compositions
uv run python four_rooms/extension/exp_convergence.py      # vary constituent optimality
uv run python four_rooms/extension/exp_train_all_tasks.py  # cache every task's EQ
uv run python four_rooms/extension/exp2_comparison.py      # per-task comparison data
uv run python four_rooms/extension/plots.py                # figures 2-6
```

Set the map size with the `NUM_ROOMS` constant at the top of each of those files (4, 8 or 16).

### GCB experiment (Four Rooms)

Run the Four Rooms GCB setup with:

```
uv run python four_rooms/extension/exp2_comparison.py
```

Use `NUM_ROOMS = 4` in `four_rooms/extension/exp2_comparison.py`.

### Stochastic sweep

Slip probability 0.0 to 0.9, three seeds.

```
sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=$(date +%Y%m%d-%H%M%S) \
  four_rooms/extension/scripts/run_stochastic_sweep.sh

sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=<same id> --dependency=afterany:<jobid> \
  four_rooms/extension/scripts/run_stochastic_sweep_aggregate.sh
```

Without Slurm:

```
for s in 0 1 2; do
  RUN_ID=local uv run python four_rooms/extension/exp_stochastic_sweep.py --num-rooms 8 --seed $s
done
RUN_ID=local uv run python four_rooms/extension/exp_stochastic_sweep.py --num-rooms 8 --aggregate-results
```

Figures in `four_rooms/extension/figures/`, data in `exps_data_extension/`.

### Composition depth

```
sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=$(date +%Y%m%d-%H%M%S) \
  four_rooms/extension/scripts/run_composition_depth.sh
```

Without Slurm:

```
uv run python four_rooms/extension/exp_composition_depth.py --num-rooms 8 --mode noise
```

## Boxman

### Convergence

```
sbatch boxman_sts/extension/scripts/run_exp_conv_parallel.sh                        # train
sbatch --dependency=afterany:<jobid> boxman_sts/extension/scripts/eval_exp_conv.sh  # evaluate
sbatch boxman_sts/extension/scripts/make_plots.sh                                   # figures
```

### Stochastic sweep

Same slip grid as Rooms, with the constituent UVFAs retrained from pixels at each slip level. Needs a GPU cluster: 120 training jobs.

```
sbatch --array=0-119%10 boxman_sts/extension/scripts/run_stochastic_sweep_train.sh
sbatch --dependency=afterany:<train_jobid> boxman_sts/extension/scripts/run_stochastic_sweep_eval.sh
sbatch --dependency=afterany:<eval_jobid>  boxman_sts/extension/scripts/run_stochastic_sweep_aggregate.sh
```

Re-running the training array is safe; finished jobs are skipped. Submit the evaluation as the single `0-2` array so all three seeds write into the same run directory.

Figures and table in `boxman_sts/plots/`, data in `boxman_sts/data/`.

### Tests

```
sbatch boxman_sts/extension/scripts/run_tests.sh
```

## Office

```
sbatch skill_machines/extension/office/scripts/train.sh
sbatch skill_machines/extension/office/scripts/eval.sh
sbatch skill_machines/extension/office/scripts/plots.sh
```

## Notes

`--export=ALL,...` is required where shown: a `VAR=x sbatch ...` prefix does not reach the job, and the script falls back to its defaults.

`boolean_composition/` extends the code released with "A Boolean Task Algebra For Reinforcement Learning" (https://arxiv.org/abs/2001.01394); new material is under `four_rooms/extension/` and `boxman_sts/extension/`.
