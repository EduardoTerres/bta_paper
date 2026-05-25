#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=boxman_conv_%A_%a.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/boolean_composition

tasks=(on off blue square)
num_tasks=${#tasks[@]}
run=$((SLURM_ARRAY_TASK_ID / num_tasks))
task=${tasks[$((SLURM_ARRAY_TASK_ID % num_tasks))]}

uv run python boxman_sts/extension/exp_convergence.py \
    --max-timesteps 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 120000 140000 160000 180000 200000 220000 240000 260000 280000 300000 \
    --run "$run" \
    --train-task "$task" \
    --max-trajectory 20 \
    --eps-timesteps 100000 \
    --require-cuda \
    --wandb
