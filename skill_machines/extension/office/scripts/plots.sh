#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=plots
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:01:00
#SBATCH --output=plots_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export PYTHONPATH="$PWD/skill_machines:${PYTHONPATH:-}"
mkdir -p "$MPLCONFIGDIR"

python skill_machines/extension/office/exp_convergence.py \
  --plot_only \
  --maxiters 1000,3000,7000,10000,15000,20000,25000,30000,40000,50000,70000,90000,110000,130000,150000,170000,190000,210000,230000,250000,270000,290000,300000
