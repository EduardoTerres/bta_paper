#!/usr/bin/env bash
#SBATCH --partition=genoa
#SBATCH --job-name=stochastic-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=stochastic_sweep_%A_%a.out

# Trains one seed per array task (3 seeds total, N_SEEDS in
# exp_stochastic_sweep.py) and saves its raw results, learned EQs, and
# convergence records under SCRATCH_DIR/run_<RUN_ID>/num_rooms_<N>/ -- after
# every slip probability, so an interrupted task keeps what it finished. Once
# all 3 array tasks finish, run run_stochastic_sweep_aggregate.sh with the same
# NUM_ROOMS and RUN_ID to average across seeds and draw the three figures.
#
# Submit as e.g.
#   sbatch --export=ALL,NUM_ROOMS=8,RUN_ID=$(date +%Y%m%d-%H%M%S) run_stochastic_sweep.sh
#
# Note the explicit --export: on this cluster a `VAR=x sbatch ...` prefix does
# NOT reach the job (it starts from a fresh login environment), so the script
# would silently fall back to the defaults below. Without RUN_ID the folder is
# named by the array job id instead, which is still shared across the array --
# just not a timestamp.

set -euo pipefail

NUM_ROOMS="${NUM_ROOMS:-8}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
# Trained in full, with no early stop (see mdp_utils), so this budget is what
# decides how converged the per-goal values are. Bigger goal sets need more:
# 200k clears the slip-0 sanity check on the 4-room map at alpha=0.1.
MAX_STEPS="${MAX_STEPS:-200000}"
export RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

cd /home/eterrescaballe/bta_paper/boolean_composition

echo "num_rooms=$NUM_ROOMS seed=$SEED run_id=$RUN_ID"

uv run python four_rooms/extension/exp_stochastic_sweep.py \
    --num-rooms "$NUM_ROOMS" \
    --seed "$SEED" \
    --max-steps "$MAX_STEPS" \
    "$@"
