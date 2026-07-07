#!/bin/bash

cd /home/eterrescaballe/bta_paper/GCB/extension/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb

export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB:$PYTHONPATH"

python inspect_checkpoint.py \
    "snapshots/GCRB_four_rooms_state_4_81540_1783341398" \
    --num-rooms 4

# python inspect_checkpoint.py \
#     "snapshots/GCRB_four_rooms_state_8_38161_1783341399" \
#     --num-rooms 8

# python inspect_checkpoint.py \
#     "snapshots/GCRB_four_rooms_state_16_82979_1783341518" \
#     --num-rooms 16