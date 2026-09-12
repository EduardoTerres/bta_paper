#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=goal-slice-gcb-fdim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:15:00
#SBATCH --array=0-5
#SBATCH --output=goal_slice_gcb_fdim_%A_%a.out

# Quick feature-dimension sweep for the visual Four Rooms goal-slice experiment.
# Each array task trains one dimension for a short 10,000-iteration run.
set -euo pipefail

cd /home/eterrescaballe/bta_paper/GCB/extension/
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb

export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB:${PYTHONPATH:-}"
export WANDB_MODE=online

feature_dims=(8 16 32 64 128 256)
feature_dim="${feature_dims[$SLURM_ARRAY_TASK_ID]}"

echo "Running feature-dimension sweep: feature_dim=${feature_dim}"

python goal_slice_gcb_four_rooms.py \
  --num-rooms 4 \
  --dataset-steps 60000 \
  --number-training-points 50000 \
  --training-iterations 10000 \
  --eval-freq 500 \
  --log-freq 100 \
  --batch-size 256 \
  --seed 1 \
  --r-universal 1.0 \
  --r-empty -1.0 \
  --r-min 0 \
  --lr 1e-5 \
  --use-wandb \
  --feature-dim "$feature_dim"
