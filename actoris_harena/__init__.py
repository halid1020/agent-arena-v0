"""actoris_harena -- simulated benchmark arenas, and the shared robot pipeline.

IMPORTING THIS PACKAGE IS CHEAP, AND MUST STAY CHEAP.

This package has two consumers with incompatible dependency sets (see the
comment at the head of pyproject.toml). The sim side wants torch, pybullet,
dm_control, robosuite and gym 0.26 under numpy<2; the rig side runs beside
LeRobot under numpy>=2 on Python 3.12 and has none of those installed.

This file used to import Agent, Arena, TrajectoryDataset and the whole of
api at module load, which transitively pulled the entire sim stack. That made
``import actoris_harena.recording`` -- one pure module that reads parquet --
impossible in a robot venv, because Python imports the package before the
submodule and the package raised ModuleNotFoundError on torch.

So the names below resolve on FIRST ACCESS, through PEP 562's module-level
``__getattr__``. Every existing spelling still works unchanged::

    from actoris_harena import Agent          # imports .agent.agent, then
    import actoris_harena as ag_ar            # and nothing else
    ag_ar.build_arena(config)                 # imports .api on this line

and a rig venv reaches its own subpackages without touching any of it.

Two consequences worth knowing:

* An import error in one of these modules now surfaces where the NAME is used,
  not where the package is imported. That is the point, but it does move the
  traceback.
* ``dir(actoris_harena)`` and tab-completion go through ``__dir__`` below, so
  the names stay discoverable without being loaded.
"""

import os
from pathlib import Path
from typing import Any

# --- Environment, set before anything that reads it can be imported ---------
# These must happen at package import (not lazily): the arena modules read them
# at THEIR import time, and PEP 562 guarantees this module body has already run
# before any submodule of it is executed.
PACKAGE_ROOT = Path(__file__).parent.resolve()

os.environ["ACTORIS_HARENA_PATH"] = str(PACKAGE_ROOT)
os.environ["RAVENS_ASSETS_DIR"] = str(PACKAGE_ROOT / "arena/raven/environments/assets")
os.environ["DEFORMABLE_RAVEN_ASSETS_DIR"] = str(
    PACKAGE_ROOT / "arena/deformable_raven/src/assets"
)

if "PYTORCH_JIT" not in os.environ:
    os.environ["PYTORCH_JIT"] = "0"

# --- The lazy surface ------------------------------------------------------
# name -> the module it lives in. Exactly the names the eager version exported,
# in the same order, so this is a mechanical transcription and not a redesign.
_LAZY: "dict[str, str]" = {
    "Agent": "actoris_harena.agent.agent",
    "TrainableAgent": "actoris_harena.agent.trainable_agent",
    "RLAgent": "actoris_harena.agent.rl_agent",
    "Arena": "actoris_harena.arena.arena",
    "Task": "actoris_harena.arena.task",
    "TrajectoryDataset": "actoris_harena.utilities.trajectory_dataset",
    "Transform": "actoris_harena.utilities.transform.transform",
    "Logger": "actoris_harena.utilities.logger.logger_interface",
    "StandardLogger": "actoris_harena.arena.loggers.standard_logger",
    "build_arena": "actoris_harena.api",
    "train_and_evaluate_single": "actoris_harena.api",
    "train_plural_eval_single": "actoris_harena.api",
    "build_transform": "actoris_harena.api",
    "evaluate": "actoris_harena.api",
    "retrieve_config": "actoris_harena.api",
    "build_agent": "actoris_harena.api",
    "run": "actoris_harena.api",
    "retrieve_config_from_path": "actoris_harena.api",
    "register_agent": "actoris_harena.api",
    "register_arena": "actoris_harena.api",
    "get_arena_class": "actoris_harena.api",
    "perform_single": "actoris_harena.utilities.perform_single",
    "perform_parallel": "actoris_harena.utilities.perform_parallel",
    "save_video": "actoris_harena.utilities.visual_utils",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    """Resolve one of :data:`_LAZY` on first access. PEP 562."""
    try:
        module_path = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    # Cache on the module so the second access costs a normal attribute lookup
    # and never re-enters here.
    globals()[name] = value
    return value


def __dir__() -> "list[str]":
    return sorted(set(globals()) | set(_LAZY))
