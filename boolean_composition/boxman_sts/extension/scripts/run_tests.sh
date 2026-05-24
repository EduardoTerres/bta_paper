#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=boxman-tests
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=boxman_tests_%A.out

set -euo pipefail

cd "$(dirname "$0")/../../.."

uv run pytest boxman_sts/extension/test_uvfa_composition.py "$@"
