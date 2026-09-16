#!/usr/bin/env bash
#SBATCH --partition=genoa
#SBATCH --job-name=composition-depth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=composition_depth_%A.out

# Does BTA's error grow with the number of composition operations?
# See four_rooms/extension/exp_composition_depth.py. One job per map -- the
# script loops over noise levels and seeds itself and writes both figures at
# the end, so there is no separate aggregate step.
#
# Submit as e.g.
#   sbatch --export=ALL,NUM_ROOMS=16,RUN_ID=$(date +%Y%m%d-%H%M%S) run_composition_depth.sh
#
# The explicit --export matters: on this cluster a `VAR=x sbatch ...` prefix
# does NOT reach the job, so the script would silently fall back to the
# defaults below.

set -euo pipefail

NUM_ROOMS="${NUM_ROOMS:-8}"
SEEDS="${SEEDS:-5}"
# Multi-goal tasks dominate runtime; the single-goal controlled slice is cheap
# and is always kept in full. Lower this first if a job runs long.
TASKS_PER_CARDINALITY="${TASKS_PER_CARDINALITY:-10}"
export RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"

cd /home/eterrescaballe/bta_paper/boolean_composition

echo "num_rooms=$NUM_ROOMS seeds=$SEEDS tasks_per_cardinality=$TASKS_PER_CARDINALITY run_id=$RUN_ID"

uv run python four_rooms/extension/exp_composition_depth.py \
    --num-rooms "$NUM_ROOMS" \
    --seeds "$SEEDS" \
    --tasks-per-cardinality "$TASKS_PER_CARDINALITY" \
    "$@"
