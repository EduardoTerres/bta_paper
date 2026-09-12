#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=goal-slice-gcb-state
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:50:00
#SBATCH --output=goal_slice_gcb_state_%A.out

set -euo pipefail
cd /home/eterrescaballe/bta_paper/GCB/extension/
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb
export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB:${PYTHONPATH:-}"
export WANDB_MODE=online
MIN_LR="${MIN_LR:-1e-9}"

python goal_slice_gcb_four_rooms_state.py \
  --num-rooms 8 \
  --dataset-steps 60000 \
  --number-training-points 50000 \
  --training-iterations 500000 \
  --eval-freq 500 \
  --log-freq 100 \
  --batch-size 256 \
  --seed 1 \
  --r-universal 1.0 \
  --r-empty -1.0 \
  --r-min 0 \
  --lr 1e-5 \
  --min-lr "$MIN_LR" \
  --use-wandb \
  --log-umap \
  --num-umap-images 50 \
  --umap-usetex \
  --feature-dim 256 \
  --query-hit-oversample 1 \
  --evaluate-on-held-out
