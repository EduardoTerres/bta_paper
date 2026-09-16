#!/usr/bin/env bash
#SBATCH --partition=genoa
#SBATCH --job-name=stochastic-sweep-16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=stochastic_sweep_16_%A_%a.out

set -euo pipefail

NUM_ROOMS="16"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

cd /home/eterrescaballe/bta_paper/boolean_composition

uv run python four_rooms/extension/exp_stochastic_sweep.py --num-rooms "$NUM_ROOMS" --seed "$SEED" "$@"
