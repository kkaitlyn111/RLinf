#!/bin/bash
#SBATCH --job-name=PPOpi05
#SBATCH --output=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/PPOpi05_%A.out
#SBATCH --error=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/PPOpi05_%A.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --constraint=48G
#SBATCH --partition=jag-standard
#SBATCH --account=nlp

export TMPDIR=/juice5b/scr5b/kaitwang/tmp
export RAY_TMPDIR=/juice5b/scr5b/kaitwang/tmp/ray
export UV_CACHE_DIR=/juice5b/scr5b/kaitwang/.uv_cache


cd /juice5b/scr5b/kaitwang/cs234/RLinf
source .venv/bin/activate
export RAY_ADDRESS="local"
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05_push_cube
