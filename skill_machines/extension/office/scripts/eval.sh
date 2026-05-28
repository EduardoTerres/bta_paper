#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=3
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=eval_%A_%a.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export PYTHONPATH="$PWD/skill_machines:${PYTHONPATH:-}"
mkdir -p "$MPLCONFIGDIR"

python skill_machines/extension/office/exp_convergence.py \
  --eval_only \
  --tasks coffee,patrol,coffee_mail,long \
  --maxiters 1000,3000,7000,10000,15000,20000,25000,30000,40000,50000,70000,90000,110000,130000,150000,170000,190000,210000,230000,250000,270000,290000,300000 \
  --optimal_reference max_observed \
  --rmin 0 \
  --runs 3 \
  --run "$SLURM_ARRAY_TASK_ID" \
  --num_runs 50 \
  --wandb

  # --maxiters 1000,3000,7000,10000,15000,20000,25000,30000,35000,40000,45000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000 \
