#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=boxman_conv_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/boolean_composition

uv run python boxman_sts/extension/exp_convergence.py
    # --wandb
    # --force-train 
