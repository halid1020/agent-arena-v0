import os
from pathlib import Path

# --- STEP 1: SET ENVIRONMENT VARIABLES FIRST ---
# Get the absolute path to the folder containing this __init__.py
PACKAGE_ROOT = Path(__file__).parent.resolve()

# Set variables BEFORE any other imports happen
os.environ["ACTORIS_HARENA_PATH"] = str(PACKAGE_ROOT)
os.environ["RAVENS_ASSETS_DIR"] = str(PACKAGE_ROOT / "arena/raven/environments/assets")
os.environ["DEFORMABLE_RAVEN_ASSETS_DIR"] = str(PACKAGE_ROOT / "arena/deformable_raven/src/assets")

if "PYTORCH_JIT" not in os.environ:
    os.environ["PYTORCH_JIT"] = "0"

# --- STEP 2: NOW PERFORM IMPORTS ---
from .agent.agent import Agent
from .agent.trainable_agent import TrainableAgent
from .agent.rl_agent import RLAgent
from .arena.arena import Arena
from .arena.task import Task

from .utilities.trajectory_dataset import TrajectoryDataset
from .utilities.transform.transform import Transform
from .utilities.logger.logger_interface import Logger
from .utilities.logger.standard_logger import StandardLogger

from .api import (
    build_arena, 
    train_and_evaluate_single, 
    train_plural_eval_single, 
    build_transform, 
    evaluate,
    retrieve_config, 
    build_agent, 
    run, 
    retrieve_config_from_path,
    register_agent, 
    register_arena, 
    get_arena_class
)

from .utilities.perform_single import perform_single
from .utilities.perform_parallel import perform_parallel
from .utilities.visual_utils import save_video