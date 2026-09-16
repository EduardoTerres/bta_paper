#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-stochastic-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --array=0-119
#SBATCH --output=boxman_stochastic_train_%A_%a.out

# Trains one constituent UVFA per array task: 3 seeds x 10 slip probabilities
# x 4 tasks (on/off/blue/square) = 120 trainings. Each writes its checkpoint to
# SCRATCH_DIR (see exp_stochastic_sweep.py) and a model.done marker; re-running
# the array skips whatever already finished, so preempted tasks can just be
# resubmitted. No evaluation here -- run run_stochastic_sweep_eval.sh once this
# array has finished, then run_stochastic_sweep_aggregate.sh.

set -euo pipefail

cd /home/eterrescaballe/bta_paper/boolean_composition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

tasks=(on off blue square)
slips=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
task=${tasks[$((task_id % ${#tasks[@]}))]}
slip=${slips[$(((task_id / ${#tasks[@]}) % ${#slips[@]}))]}
seed=$((task_id / (${#tasks[@]} * ${#slips[@]})))

echo "seed=$seed slip=$slip task=$task"

uv run python boxman_sts/extension/exp_stochastic_sweep.py \
    --seed "$seed" \
    --slip-probs "$slip" \
    --train-task "$task" \
    --max-timesteps 200000 \
    --eps-timesteps 100000 \
    --require-cuda \
    --train-only \
    "$@"
