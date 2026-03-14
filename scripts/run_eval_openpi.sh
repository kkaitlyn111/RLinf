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

cd /juice5b/scr5b/kaitwang/cs234/RLinf
source .venv/bin/activate
export RAY_ADDRESS="local"

model=/juice5b/scr5b/kaitwang/cs234/results/maniskill_sft_openpi_pi05_lora/checkpoints/global_step_1000

# python rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py \
# --dcp_path $model/actor/dcp_checkpoint \
# --output_path $model/actor/model.pt

bash examples/embodiment/eval_embodiment.sh maniskill_eval_openpi_pickcube \
  "runner.ckpt_path=$model/actor/model.pt"