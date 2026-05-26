#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=safety-conv-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --output=safety_conv_eval_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python skill_machines/extension/safety_gym/exp_convergence.py \
  --eval_only \
  --runs 3 \
  --eval_episodes 100 \
  --maxiters 100000,200000,400000,600000,800000,1000000 \
  --wandb
