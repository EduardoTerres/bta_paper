#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=safety-plots
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=safety_plots_%A.out

set -euo pipefail

cd /home/eterrescaballe/bta_paper

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sm

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"

export SAFETY_GYM_DATA_DIR="${SAFETY_GYM_DATA_DIR:-/scratch-shared/${USER}/bta_paper/safety_gym/exps_data_extension}"
mkdir -p skill_machines/extension/safety_gym/exps_data_extension/slurm

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
  --plot_only \
  --training-output single \
  --runs 1 \
  --maxiters 50000,100000,200000,400000,700000,1000000,1500000,2000000,2500000,3000000,3500000,4000000 \
  --runs_dir "$SAFETY_GYM_DATA_DIR/runs" \
  --output "$SAFETY_GYM_DATA_DIR/runs/run_000/sm_convergence_1run_2.pkl" \
  --figures_dir "$PWD/skill_machines/extension/safety_gym/exps_data_extension/figures_1run"

# 2 runs
# --maxiters 10000,100000,150000,300000,400000,700000,1000000,1500000,2000000 \

# 1 run with more iters
# --maxiters 50000,100000,200000,400000,700000,1000000,1500000,2000000,2500000,3000000,3500000,4000000 \
