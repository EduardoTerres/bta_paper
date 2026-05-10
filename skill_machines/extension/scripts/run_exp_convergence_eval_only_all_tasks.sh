#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=conv-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=conv_eval_%A.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python skill_machines/extension/exp_convergence.py \
  --eval_only \
  --tasks coffee,patrol,coffee_mail,long \
  --checkpoint_env Office-CoffeeMail-Task-v0 \
  --maxiters 100000,200000,400000,600000,800000,1000000 \
  --optimal_reference max_observed \
  --num_runs 500
