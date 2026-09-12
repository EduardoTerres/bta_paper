#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=goal-slice-collapse-plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=goal_slice_collapse_plot_%A.out

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root/GCB/extension"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb

export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python plot_goal_slice_gcb_collapse.py \
  --usetex \
  --output figures/goal_slice_gcb_rooms8_collapse.pdf

# Render all figure text through the system LaTex installation.
