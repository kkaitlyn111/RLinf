#!/bin/bash
#SBATCH --job-name=PPOmlp
#SBATCH --output=logs/PPOmlp_%A.out
#SBATCH --error=logs/PPOmlp_%A.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --partition=sphinx
#SBATCH --account=nlp

export TMPDIR=/juice5b/scr5b/kaitwang/tmp
export RAY_TMPDIR=/juice5b/scr5b/kaitwang/tmp/ray
export UV_CACHE_DIR=/juice5b/scr5b/kaitwang/.uv_cache


cd /juice5b/scr5b/kaitwang/cs234/RLinf
source .venv/bin/activate
export RAY_ADDRESS="local"
bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp
