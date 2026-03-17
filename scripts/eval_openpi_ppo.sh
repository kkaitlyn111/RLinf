#!/bin/bash
#SBATCH --job-name=eval_pi05                                                                                                          
#SBATCH --output=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/eval_pi05_%A.out
#SBATCH --error=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/eval_pi05_%A.err                                                             
#SBATCH --cpus-per-task=20                                                                                                          
#SBATCH --gres=gpu:1                                                                                                             
#SBATCH --constraint=48G
#SBATCH --partition=jag-standard
#SBATCH --account=nlp

export TMPDIR=/juice5b/scr5b/kaitwang/tmp
export RAY_TMPDIR=/juice5b/scr5b/kaitwang/tmp/ray
export UV_CACHE_DIR=/juice5b/scr5b/kaitwang/.uv_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export EMBODIED_PATH=/juice5b/scr5b/kaitwang/cs234/RLinf/examples/embodiment
export PYTHONPATH=/juice5b/scr5b/kaitwang/cs234/RLinf:$PYTHONPATH

cd /juice5b/scr5b/kaitwang/cs234/RLinf
source /nlp/scr/djghosh/vladata/rlinf_env_openpi/bin/activate
# source .venv/bin/activate
export RAY_ADDRESS="local"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

lr=7.5e-4
step=1600
# modelpt=/juice5b/scr5b/kaitwang/cs234/results/maniskill_sft_openpi_pi05_lora_lr${lr}_bs16/checkpoints/global_step_${step}/actor/model.pt
# modelpt=/juice5b/scr5b/kaitwang/cs234/results/maniskill_sft_openpi_pi05_lora_lr5e-05_bs16/checkpoints/global_step_600/actor/model.pt
modelpt=/juice5b/scr5b/kaitwang/cs234/results/maniskill_sft_openpi_pi05_lora_lr0.00075_bs16/checkpoints/global_step_1600/actor/model.pt
logdir=/juice5b/scr5b/kaitwang/cs234/results/eval_openpi_pickcube_lr${lr}_step${step}

python $EMBODIED_PATH/eval_embodied_agent.py \
  --config-path $EMBODIED_PATH/config/ \
  --config-name openpi_pickcube_eval \
  "++runner.ckpt_path=$modelpt" \
  "runner.logger.experiment_name=eval_openpi_pickcube_lr${lr}_step${step}" \
  "runner.logger.log_path=$logdir" \
  "++eval.lr=${lr}" \
  "++eval.step=${step}"