#!/usr/bin/env bash
# Activate the SIMULATION environment (the [sim] extra).
#
# There are two environments and they are not interchangeable. This one runs
# the benchmark arenas: numpy<2, gym 0.26, robosuite, pybullet, dm_control.
#
# The RIG pipeline -- the collection / visualisation / training / deployment
# code the robot repos import -- lives in the same package but runs beside
# LeRobot under numpy>=2 on Python >=3.12, and is installed into each robot
# repo's own venv with:
#
#     pip install -e "/path/to/actoris_harena[rig]"
#
# Do not install [sim] and [rig] into one environment. MEASURED: LeRobot pins
# numpy>=2.0,<2.3 and the sim stack pins numpy<2.0, so pip cannot satisfy both.
# `import actoris_harena` is lazy precisely so that a venv with only [rig]
# installed can reach actoris_harena.recording without any of the above.

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