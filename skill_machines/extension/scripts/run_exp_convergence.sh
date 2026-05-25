#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=conv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=conv_%A_%a.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

labels=(boolean)
num_labels=${#labels[@]}
run=$((SLURM_ARRAY_TASK_ID / num_labels))
label=${labels[$((SLURM_ARRAY_TASK_ID % num_labels))]}


python skill_machines/extension/exp_convergence.py \
  --tasks coffee,patrol,coffee_mail,long \
  --maxiters 1000,10000,50000,100000,200000,500000,800000,1000000 \
  --optimal_reference none \
  --num_runs 500 \
  --run "$run" \
  --train-label "$label" \
  --wandb
