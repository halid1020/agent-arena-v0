conda deactivate
conda activate agent-arena-v0

export PYTORCH_JIT=0
export EGL_GPU=$CUDA_VISIBLE_DEVICES

### Agent-Arena
export AGENT_ARENA_PATH=${PWD}/agent_arena
export PYTHONPATH=${AGENT_ARENA_PATH}/../:$PYTHONPATH

### Raven
export RAVENS_ASSETS_DIR=${AGENT_ARENA_PATH}/arena/raven/environments/assets

### Deformable Raven
export DEFORMABLE_RAVEN_ASSETS_DIR=${AGENT_ARENA_PATH}/arena/deformable_raven/src/assets

### For ROS
source $CONDA_PREFIX/setup.bash

pip install .