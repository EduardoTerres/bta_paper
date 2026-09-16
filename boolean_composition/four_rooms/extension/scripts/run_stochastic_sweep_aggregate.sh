#!/usr/bin/env bash
#SBATCH --partition=genoa
#SBATCH --job-name=stochastic-sweep-aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=stochastic_sweep_aggregate_%A.out

# Run once all array tasks of run_stochastic_sweep.sh have finished: loads
# every seed's saved results from the run directory, averages across seeds, and
# writes the three figures (normalized suboptimality, value gap, measured
# rollout return) plus convergence.txt. Does no training.
#
# Each artefact is written twice: once into the run directory (the canonical
# copy, traceable to the job that made it) and once into the repo, so the
# paper's figures sit where the rest of the repo expects them --
#   four_rooms/extension/figures/  figures + convergence_<N>.txt
#   exps_data_extension/           stochastic_sweep_<N>.h5
# Override with --figures-dir / --data-dir, or pass "" to either to write only
# to the run directory. (The Boxman counterpart writes to boxman_sts/plots and
# boxman_sts/data the same way.)
#
# Submit with the same NUM_ROOMS and RUN_ID the sweep used, e.g.
#   sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=20260914-093000 run_stochastic_sweep_aggregate.sh
# The explicit --export matters: a `VAR=x sbatch ...` prefix does not reach the
# job on this cluster. With no RUN_ID it picks the most recently written run
# folder for that map.

set -euo pipefail

NUM_ROOMS="${NUM_ROOMS:-8}"

cd /home/eterrescaballe/bta_paper/boolean_composition

run_id_args=()
if [[ -n "${RUN_ID:-}" ]]; then
    run_id_args=(--run-id "$RUN_ID")
fi

uv run python four_rooms/extension/exp_stochastic_sweep.py \
    --num-rooms "$NUM_ROOMS" \
    --aggregate-results \
    "${run_id_args[@]}" \
    "$@"
