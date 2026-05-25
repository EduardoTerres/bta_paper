#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-conv-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=boxman_conv_eval_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/boolean_composition
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

plot_args=(
    --output boxman_sts/data/convergence_returns.h5
    --figure boxman_sts/plots/convergence_returns.png
)

uv run python boxman_sts/extension/make_plots.py "${plot_args[@]}"
