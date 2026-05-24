#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-conv-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=boxman_conv_eval_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/boolean_composition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval_args=(
    --max-timesteps 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 120000 140000 160000 180000 200000 220000 240000 260000 280000 300000
    --max-trajectory 20
    --num-eval-episodes 100
    --require-cuda
    --output boxman_sts/data/convergence_returns.h5
    --figure boxman_sts/plots/convergence_returns.png
    --eval-only
)

# --max-timesteps 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 120000 140000 160000 180000 200000 220000 240000 260000 280000 300000

uv run python boxman_sts/extension/exp_convergence.py "${eval_args[@]}"

plot_args=(
    --output boxman_sts/data/convergence_returns.h5
    --figure boxman_sts/plots/convergence_returns.png
    --make-plots
)

uv run python boxman_sts/extension/exp_convergence.py "${plot_args[@]}"
