cd /juice5b/scr5b/kaitwang/cs234/RLinf
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/uv_cache

# make hydra searchpath in your yaml resolve env/maniskill_pick_cube.yaml
export EMBODIED_PATH=/juice5b/scr5b/kaitwang/cs234/RLinf/examples/embodiment
export PYTHONPATH=/juice5b/scr5b/kaitwang/cs234/RLinf:$PYTHONPATH

uv run python examples/sft/train_embodied_sft.py \
  --config-path /juice5b/scr5b/kaitwang/cs234/RLinf/examples/sft/config \
  --config-name custom_sft_openpi
