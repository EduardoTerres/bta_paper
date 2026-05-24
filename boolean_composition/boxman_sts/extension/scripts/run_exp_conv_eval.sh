#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-conv-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=boxman_conv_eval_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/boolean_composition

uv run python boxman_sts/extension/exp_convergence.py \
    --max-timesteps 5000 10000 15000 20000 \
    --max-trajectory 20 \
    --num-eval-episodes 10 \
    --require-cuda \
    --output boxman_sts/data/convergence_returns.h5 \
    --figure boxman_sts/plots/convergence_returns.png \
    --eval-only
