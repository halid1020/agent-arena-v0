"""Where this rig writes things: one answer, asked for in one place.

Run logs, analyses, camera views and staged datasets all land under one root,
and three modules used to decide it separately -- two by reading
``$SO101_OUTPUT_DIR`` and one by counting ``Path(__file__).parents[3]``. The
first names one rig; the second was right only while the file lived inside that
rig's checkout, and became "wherever pip put this package" the moment it did
not.

So a rig declares its root, and everything here asks. ``$RIG_OUTPUT_DIR`` and
the older ``$SO101_OUTPUT_DIR`` are both honoured, because so101_garment's
setup.sh, its Slurm jobs and a dozen of its documents name the old one and a
rename that silently moved a running rig's outputs would be a poor trade.
"""

import os
from pathlib import Path

#: Environment variables naming the output root, most specific first.
OUTPUT_DIR_VARS = ("RIG_OUTPUT_DIR", "SO101_OUTPUT_DIR")

_DECLARED: "Path | None" = None


def set_output_root(path: "Path | str") -> None:
    """Declare where this rig's outputs go. Overrides the environment."""
    global _DECLARED
    _DECLARED = Path(path)


def output_root() -> Path:
    """The declared root, else the first environment variable set, else ``outputs``.

    The last fallback is relative on purpose: it resolves against the working
    directory, which for every entry point here is the rig's checkout. An
    absolute guess derived from this file's location would be a confident wrong
    answer instead of an obvious one.
    """
    if _DECLARED is not None:
        return _DECLARED
    for var in OUTPUT_DIR_VARS:
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser()
    return Path("outputs")
