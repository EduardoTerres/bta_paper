#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=conv-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=10:00:00
#SBATCH --output=conv_eval_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

eval_args=(
  --eval-only
  --tasks coffee,patrol,coffee_mail,long
  --maxiters 1000,10000,50000,100000,200000,500000,800000,1000000
  --optimal_reference max_observed
  --rmin 0
  --num_runs 500
  --output skill_machines/extension/exps_data_extension/sm_convergence_new.pkl
)

# python skill_machines/extension/exp_convergence.py "${eval_args[@]}"

plot_args=(
  --output skill_machines/extension/exps_data_extension/sm_convergence_new.pkl
  --make-plots
)

python skill_machines/extension/exp_convergence.py "${plot_args[@]}"
