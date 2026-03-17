#!/bin/bash
#SBATCH --job-name=ppopi05                                                                                                          
#SBATCH --output=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/ppopi05_%A.out
#SBATCH --error=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/ppopi05_%A.err                                                             
#SBATCH --cpus-per-task=20                                                                                                          
#SBATCH --gres=gpu:4                                                                                                            
#SBATCH --constraint=48G
#SBATCH --partition=jag-standard
#SBATCH --account=nlp

export REPO_PATH=/juice5b/scr5b/kaitwang/cs234/RLinf
export EMBODIED_PATH=${REPO_PATH}/examples/embodiment
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

export TF_CPP_MIN_LOG_LEVEL=2

CONFIG_NAME=fromsft_ppo_openpi_pickcube

export TMPDIR=/juice5b/scr5b/kaitwang/tmp
export RAY_TMPDIR=/juice5b/scr5b/kaitwang/tmp/ray
export UV_CACHE_DIR=/juice5b/scr5b/kaitwang/.uv_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

cd /juice5b/scr5b/kaitwang/cs234/RLinf
source /nlp/scr/djghosh/vladata/rlinf_env_openpi/bin/activate

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
pkill -u "$USER" -f "ray" 2>/dev/null || true
python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME}