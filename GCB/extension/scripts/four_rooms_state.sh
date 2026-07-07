#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gcb-4-rooms-state
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:59:00
#SBATCH --output=gcb_rooms_state_%A.out

cd /home/eterrescaballe/bta_paper/GCB/extension/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb

export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB:$PYTHONPATH"
export WANDB_MODE=online

python offline_gcab_four_rooms.py \
    --num-rooms 4 \
    --training-iterations 30000 \
    --eval-freq 100 \
    --log-freq 50 \
    --use-wandb
