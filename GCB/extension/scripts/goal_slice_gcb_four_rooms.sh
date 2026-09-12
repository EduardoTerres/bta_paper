#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=goal-slice-gcb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --output=goal_slice_gcb_%A.out

set -euo pipefail
cd /home/eterrescaballe/bta_paper/GCB/extension/
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb
export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB:${PYTHONPATH:-}"
export WANDB_MODE=online

python goal_slice_gcb_four_rooms.py \
  --num-rooms 16 \
  --dataset-steps 200000 \
  --number-training-points 180000 \
  --training-iterations 100000 \
  --eval-freq 500 \
  --log-freq 100 \
  --batch-size 256 \
  --seed 1 \
  --r-universal 1.0 \
  --r-empty -1.0 \
  --r-min -0.25 \
  --lr 1e-4 \
  --min-lr 1e-9 \
  --use-wandb \
  --log-umap \
  --num-umap-images 50 \
  --umap-usetex \
  --feature-dim 1024 \
  --num-layers 12 \
  --num-filters 32 \
  --query-hit-oversample 1 \
  --evaluate-on-held-out
