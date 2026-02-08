conda deactivate
conda activate actoris-harena

export PYTORCH_JIT=0
export EGL_GPU=$CUDA_VISIBLE_DEVICES

### Actoris-Harene
export ACTORIS_HARENA_PATH=${PWD}/actoris_harena
export PYTHONPATH=${ACTORIS_HARENA_PATH}/../:$PYTHONPATH

### Raven
export RAVENS_ASSETS_DIR=${ACTORIS_HARENA_PATH}/arena/raven/environments/assets

### Deformable Raven
export DEFORMABLE_RAVEN_ASSETS_DIR=${ACTORIS_HARENA_PATH}/arena/deformable_raven/src/assets