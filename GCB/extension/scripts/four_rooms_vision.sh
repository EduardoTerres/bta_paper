#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=gcb-4-rooms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --output=gcb_4_rooms_%A.out

cd /home/eterrescaballe/bta_paper/GCB/extension/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb

export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB:$PYTHONPATH"
export WANDB_MODE=online

python offline_gcab_four_rooms_vision.py \
    --num-rooms 4 \
    --training-iterations 50000 \
    --eval-freq 200 \
    --log-freq 200 \
    --seed 0 \
    --use-wandb \
    --compositionality-weight 2 \
    --dense-rewards \

    # --train-multi-goal \
