import os
from pathlib import Path

# Get the absolute path to the folder containing this __init__.py
PACKAGE_ROOT = Path(__file__).parent.resolve()

# Automatically set environment variables for the current session
os.environ["ACTORIS_HARENA_PATH"] = str(PACKAGE_ROOT) / "actoris_harena"
os.environ["RAVENS_ASSETS_DIR"] = str(PACKAGE_ROOT / "actoris_harena/arena/raven/environments/assets")
os.environ["DEFORMABLE_RAVEN_ASSETS_DIR"] = str(PACKAGE_ROOT / "actoris_harena/arena/deformable_raven/src/assets")

# Optional: Set hardware defaults if they aren't already set
if "PYTORCH_JIT" not in os.environ:
    os.environ["PYTORCH_JIT"] = "0"