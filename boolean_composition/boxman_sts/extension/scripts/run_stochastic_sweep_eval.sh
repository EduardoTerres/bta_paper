#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-stochastic-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=boxman_stochastic_eval_%A_%a.out

# One array task per seed. Loads that seed's checkpoints (written by
# run_stochastic_sweep_train.sh), composes every evaluation task both ways, and
# solves the exact finite-horizon MDP for each. Results are saved to
# the run directory after every slip probability, so an interrupted task
# resumes where it stopped. Run run_stochastic_sweep_aggregate.sh once all 3
# finish.
#
# --rollout-episodes is 50 rather than the default 200 because `CollectEnv.step`
# renders a 400x400 frame every step (~40 steps/s), even though the rollout
# never looks at the pixels -- it acts from the per-cell greedy policy
# `greedy_policy` already read off the DQN. 200 episodes across 19 policies and
# 10 slip probabilities would not fit in the wall clock; 50 does, and the
# figure averages over 5 tasks and 3 seeds on top of that.

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID:-0}"
# Every array task must agree on the run folder or the seeds cannot be
# aggregated together, so this falls back to the array job id (shared across
# the array) rather than to a per-task timestamp. To name the folder yourself,
# submit with an explicit --export:
#   sbatch --export=ALL,RUN_ID=$(date +%Y%m%d-%H%M%S) run_stochastic_sweep_eval.sh
# A `RUN_ID=x sbatch ...` prefix does NOT reach the job on this cluster, so the
# fallback above would silently win.
export RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
echo "seed=$SEED run_id=$RUN_ID"

cd /home/eterrescaballe/bta_paper/boolean_composition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python boxman_sts/extension/exp_stochastic_sweep.py \
    --seed "$SEED" \
    --max-trajectory 20 \
    --rollout-episodes 50 \
    --require-cuda \
    --evaluate-only \
    "$@"
