#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=9
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export PYTHONPATH="$PWD/skill_machines:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"

labels=(boolean)
num_labels=${#labels[@]}
run=$((SLURM_ARRAY_TASK_ID / num_labels))
label=${labels[$((SLURM_ARRAY_TASK_ID % num_labels))]}


python -u skill_machines/extension/office/exp_convergence.py \
  --tasks coffee,patrol,coffee_mail,long \
  --maxiters 1000,3000,7000,10000,15000,20000,25000,30000,35000,40000,45000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000,210000,220000,230000,240000,250000,260000,270000,280000,290000,300000 \
  --optimal_reference none \
  --rmin 0 \
  --run "$run" \
  --train-label "$label" \
  --wandb
