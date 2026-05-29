#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=3
#SBATCH --job-name=safety-train-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=safety_train_single_%A_%a.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"

export SAFETY_GYM_DATA_DIR="${SAFETY_GYM_DATA_DIR:-/scratch-shared/${USER}/bta_paper/safety_gym/exps_data_extension}"
mkdir -p "$SAFETY_GYM_DATA_DIR/runs" "$SAFETY_GYM_DATA_DIR/logs" "$SAFETY_GYM_DATA_DIR/slurm"

PROJECT_MUJOCO="$PWD/.local/mujoco/mujoco210"
PROJECT_MUJOCO_PY="$PWD/.local/mujoco_py"
PROJECT_MUJOCO_COMPAT="$PWD/.local/mujoco_compat/usr/lib64"
if [[ -z "${MUJOCO_PY_MUJOCO_PATH:-}" && -d "$PROJECT_MUJOCO" ]]; then
  export MUJOCO_PY_MUJOCO_PATH="$PROJECT_MUJOCO"
else
  export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
fi
if [[ -d "$PROJECT_MUJOCO_PY" ]]; then
  export PYTHONPATH="$PROJECT_MUJOCO_PY:${PYTHONPATH:-}"
fi
export PYTHONPATH="$PWD/skill_machines:${PYTHONPATH:-}"
export CPATH="/usr/include:${CONDA_PREFIX}/include:${CPATH:-}"
export C_INCLUDE_PATH="/usr/include:${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export LIBRARY_PATH="/usr/lib64:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
export LDFLAGS="-L/usr/lib64 -L${CONDA_PREFIX}/lib ${LDFLAGS:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_PY_MUJOCO_PATH/bin:$PROJECT_MUJOCO_COMPAT:${CONDA_PREFIX}/lib:/usr/lib64"
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"

num_runs=2
run="${SLURM_ARRAY_TASK_ID}"

if (( run >= num_runs )); then
  echo "Invalid array index ${SLURM_ARRAY_TASK_ID}: run ${run} >= ${num_runs}"
  exit 1
fi

python -u skill_machines/extension/safety_gym/exp_convergence.py \
  --run "$run" \
  --training-output single \
  --resume-training \
  --runs "$num_runs" \
  --maxiters 50000,100000,200000,400000,700000,1000000,1500000,2000000,2500000,3000000,3500000,4000000 \
  --runs_dir "$SAFETY_GYM_DATA_DIR/runs" \
  --log_dir "$SAFETY_GYM_DATA_DIR/logs" \
  --wandb
