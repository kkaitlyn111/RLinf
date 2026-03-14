#!/bin/bash
#SBATCH --job-name=sftpi05
#SBATCH --output=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/sftpi05_%A.out
#SBATCH --error=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/sftpi05_%A.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --partition=jag-standard
#SBATCH --account=nlp

cd /juice5b/scr5b/kaitwang/cs234/RLinf                                                                       
source /juice5b/scr5b/kaitwang/cs234/RLinf/.venv/bin/activate
                                                                                                                            
export HF_LEROBOT_HOME=/juice5b/scr5b/kaitwang/cs234      
export EMBODIED_PATH=$(pwd)
export RAY_ADDRESS="local"

python examples/sft/train_embodied_sft.py --config-name maniskill_sft_openpi_lora_quick