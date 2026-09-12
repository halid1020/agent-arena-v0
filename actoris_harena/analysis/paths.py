"""Where an analysis is written, so two of them can be found and compared.

One rule: **every analysis lands under a directory named for the day it was
run, and a subdirectory named for what it contains.**

::

    outputs/analysis/<YYYY-MM-DD>/<content>/          the numbers and figures
    outputs/analysis/<YYYY-MM-DD>/<content>/slides/   the deck built from them

The date is the answer to the question that was actually being asked of the old
layout -- *is this the current result?* A bare ``outputs/analysis/act-all`` said
nothing about whether it came from this week's code or from a checkpoint two
retrainings ago, and the only way to tell was the file mtimes. Naming the
directory for the day makes a stale deck obvious at a glance and lets a rerun
sit beside its predecessor instead of overwriting it.

``<content>`` is ``<policy>-<camera slug>``: the slug is the one
``recording.dataset_view.view_slug`` already produces for a camera set, so an
analysis directory and the run directory it analysed are named the same way and
can be matched by eye.

``$SO101_OUTPUT_DIR`` is honoured, as everything else in the repo does -- the
analysis tools were the exception and wrote relative to the working directory,
so running one from anywhere but the repo root scattered results.
"""

from __future__ import annotations

import datetime as _datetime
import re
from pathlib import Path

from actoris_harena.outputs import output_root

#: The one place the analysis tree is rooted.
ANALYSIS_SUBDIR = "analysis"

#: A day, as it appears in a directory name.
DATE_FORMAT = "%Y-%m-%d"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: What may appear in a `<content>` name. Deliberately narrow: these become
#: directory names that end up in commands, in slide captions and in filenames.
_CONTENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def outputs_root() -> Path:
    """The rig's output root -- see ``actoris_harena.outputs``.

    This used to add ``parents[3]`` as its own fallback, which named the rig's
    checkout only while this file was inside it.
    """
    return output_root()


def today() -> str:
    """The current day in the directory format. Local time, like a lab notebook."""
    return _datetime.date.today().strftime(DATE_FORMAT)


def check_date(date: str) -> None:
    """Refuse anything that is not a ``YYYY-MM-DD`` day."""
    if not _DATE_RE.match(date):
        raise ValueError(
            f"analysis date {date!r} is not YYYY-MM-DD. The date is part of the "
            "convention, not a free-text label -- put the label in the content name."
        )


def check_content(content: str) -> None:
    """Refuse a content name that would not survive being a directory."""
    if not content or not _CONTENT_RE.match(content):
        raise ValueError(
            f"analysis name {content!r} must start with a letter or digit and hold "
            "only letters, digits, dot, underscore, plus or hyphen -- it becomes a "
            "directory name, a filename and a slide caption. A camera set is joined "
            "with '+' the way a run directory joins it."
        )


def content_name(policy: str, cameras: "str | None" = None) -> str:
    """``<policy>-<camera slug>``, the standard content name.

    ``cameras`` may be the slug a view already carries (``all``,
    ``central+left_arm_left_gripper``) or ``None`` for an analysis that is not
    about one camera set.
    """
    name = policy if not cameras else f"{policy}-{cameras}"
    check_content(name)
    return name


def analysis_dir(content: str, date: "str | None" = None) -> Path:
    """The directory this analysis belongs in. Does not create it."""
    # `None` means today; "" does not. An empty string reaching here is a caller
    # that meant to pass a date and computed one wrongly, and silently filing the
    # result under today would hide that.
    day = today() if date is None else date
    check_date(day)
    check_content(content)
    return outputs_root() / ANALYSIS_SUBDIR / day / content


def slides_dir(analysis: Path) -> Path:
    """Where the deck built from an analysis directory goes."""
    return Path(analysis) / "slides"
