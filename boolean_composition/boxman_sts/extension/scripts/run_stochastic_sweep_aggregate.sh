#!/usr/bin/env bash
#SBATCH --partition=genoa
#SBATCH --job-name=boxman-stochastic-aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=boxman_stochastic_aggregate_%A.out

# Run once all array tasks of run_stochastic_sweep_eval.sh have finished: loads
# every seed's saved results, averages across seeds, and writes the figure and
# the training table. Trains and evaluates nothing.

set -euo pipefail

# Pass the same RUN_ID the eval array used; with none, the most recently
# written run folder is picked up.
run_id_args=()
if [[ -n "${RUN_ID:-}" ]]; then
    run_id_args=(--run-id "$RUN_ID")
fi

cd /home/eterrescaballe/bta_paper/boolean_composition

uv run python boxman_sts/extension/exp_stochastic_sweep.py \
    --aggregate-results \
    --output boxman_sts/data/stochastic_sweep.h5 \
    --figure boxman_sts/plots/stochastic_sweep.png \
    --table boxman_sts/plots/stochastic_sweep_training.txt \
    "${run_id_args[@]}" \
    "$@"
