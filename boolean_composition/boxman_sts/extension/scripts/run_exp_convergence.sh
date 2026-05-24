#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=20:00:00
#SBATCH --output=boxman_conv_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/boolean_composition

args=(
    --max-timesteps 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 120000 140000 160000 180000 200000 250000 300000 350000 400000 450000 500000 550000 600000 650000 700000 750000 800000
    --max-trajectory 20
    --output boxman_sts/data/convergence_returns.h5
    --figure boxman_sts/plots/convergence_returns.png
    --eval-only
    --make-plots
    --wandb
    --require-cuda
    # --debug
)

uv run python boxman_sts/extension/exp_convergence.py "${args[@]}"
