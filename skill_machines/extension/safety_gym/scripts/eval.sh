#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=safety-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --output=/scratch-shared/eterrescaballe/bta_paper/safety_gym/exps_data_extension/slurm/safety_eval_%j.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export MPLBACKEND=Agg
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

python skill_machines/extension/safety_gym/exp_convergence.py \
  --eval_only \
  --runs 3 \
  --eval_episodes 100 \
  --maxiters 100000,200000,400000,600000,800000,1000000 \
  --runs_dir "$SAFETY_GYM_DATA_DIR/runs" \
  --log_dir "$SAFETY_GYM_DATA_DIR/logs" \
  --output "$SAFETY_GYM_DATA_DIR/sm_convergence.pkl" \
  --wandb \
  --no_plot
