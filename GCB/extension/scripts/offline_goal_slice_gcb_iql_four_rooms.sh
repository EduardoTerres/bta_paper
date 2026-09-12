#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=goal-slice-gcb-iql
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --output=goal_slice_gcb_iql_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/GCB/extension
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb

export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB:${PYTHONPATH:-}"
export WANDB_MODE=online

python /home/eterrescaballe/bta_paper/GCB/extension/offline_goal_slice_gcb_iql_four_rooms.py \
  --num-rooms 4 \
  --obs-img-dim 64 \
  --max-episode-steps 75 \
  --dataset-loc /home/eterrescaballe/bta_paper/GCB/extension/replay_goal_slice_four_rooms_8_heldout80_20.pt \
  --dataset-steps 200000 \
  --number-training-points 180000 \
  --no-collect \
  --training-iterations 50000 \
  --batch-size 256 \
  --query-hit-oversample 1.0 \
  --validation-fraction 0.1 \
  --validation-freq 500 \
  --lr-patience 5 \
  --lr-factor 0.5 \
  --min-lr 1e-9 \
  --eval-freq 500 \
  --log-freq 100 \
  --num-checkpoints 10 \
  --feature-dim 1024 \
  --num-layers 8 \
  --num-filters 32 \
  --metric-loss l1 \
  --discount 0.99 \
  --action-weight 25.0 \
  --lr 1e-4 \
  --weight-decay 5e-4 \
  --r-universal 1.0 \
  --r-empty -1.0 \
  --r-min -0.25 \
  --evaluate-on-held-out \
  --collapse-eval-states 32 \
  --collapse-sets-per-status 16 \
  --log-umap \
  --num-umap-images 50 \
  --global-umap-states 32 \
  --individual-umap-contexts 4 \
  --umap-sets-per-class 64 \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1 \
  --umap-usetex \
  --iql-hidden-dim 256 \
  --actor-lr 1e-4 \
  --critic-lr 1e-4 \
  --iql-expectile 0.7 \
  --iql-beta 3.0 \
  --iql-advantage-clip 100.0 \
  --iql-target-tau 0.005 \
  --rl-eval-episodes-per-status 32 \
  --device cuda \
  --seed 0 \
  --output-dir /home/eterrescaballe/bta_paper/GCB/extension/goal_slice_gcb_iql_runs \
  --use-wandb \
  --project-name gcb-four-rooms \
  --group goal-slice-gcb-iql-rooms8
