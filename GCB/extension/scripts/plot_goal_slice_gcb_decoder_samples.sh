#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=goal-slice-decoder-samples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=goal_slice_decoder_samples_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper/GCB/extension

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcb
export PYTHONPATH="/home/eterrescaballe/bta_paper/GCB"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/goal_slice_gcb_matplotlib"
mkdir -p "$MPLCONFIGDIR"

RUNS=(
    # goal-slice-gcb-rooms8-fdim256-rmin-0.25-iters50000-bs256-seed1
    goal-slice-gcb-rooms8-fdim512-rmin-0.25-iters100000-bs256-qhit1-seed1
    # goal-slice-gcb-rooms8-fdim1024-rmin-0.25-iters100000-bs256-qhit1-seed1
)
RUNS_DIR=/home/eterrescaballe/bta_paper/GCB/extension/goal_slice_gcb_runs
FIGURES_DIR=/home/eterrescaballe/bta_paper/GCB/extension/figures

CHECKPOINT_ARGS=()
for run in "${RUNS[@]}"; do
    run_dir="$RUNS_DIR/$run"
    final_checkpoint="$run_dir/goal_slice_gcb.pt"
    if [[ -f "$final_checkpoint" ]]; then
        checkpoint="$final_checkpoint"
    else
        shopt -s nullglob
        checkpoints=("$run_dir"/goal_slice_gcb_step_*.pt)
        shopt -u nullglob
        if ((${#checkpoints[@]} == 0)); then
            echo "No checkpoint found for run: $run" >&2
            exit 1
        fi
        checkpoint="${checkpoints[${#checkpoints[@]} - 1]}"
    fi

    CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
done

python plot_goal_slice_gcb_decoder_samples.py \
    "${CHECKPOINT_ARGS[@]}" \
    --output "$FIGURES_DIR/goal_slice_gcb_decoder_samples.png" \
    --num-samples 4 \
    --seed 0 \
    --goal-set-split held-out
