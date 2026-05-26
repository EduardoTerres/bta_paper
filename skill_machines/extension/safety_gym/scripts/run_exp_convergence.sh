#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=safety-conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=safety_conv_%A_%a.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python skill_machines/extension/safety_gym/exp_convergence.py \
  --run "$SLURM_ARRAY_TASK_ID" \
  --runs 3 \
  --maxiters 100000,200000,400000,600000,800000,1000000 \
  --wandb
