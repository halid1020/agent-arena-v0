"""One line format, so a new curve costs one call instead of four edits.

The driver's stdout log is the only metric record this pipeline has --
``--wandb.enable=false`` is unconditional in all five scripts, the pinned
LeRobot ships no ``SummaryWriter``, and nothing writes a jsonl. A number
therefore has to cross three hops before it can be drawn, and each of them was
a closed list: the remote grep in :mod:`actoris_harena.training.runs` drops what it
does not recognise ON THE FAR SIDE, :mod:`actoris_harena.training.progress` parses an
enumerated set of lerobot's own field names, and the page named three of them
literally.

So the hops are generic now, and this module is the contract that makes that
safe. A policy, a callback or a tool prints

    so101-metric step=12000 eval/loss=0.0421 eval/psnr_central=31.2

and the value appears as a curve, with no change to the grep, the parser, the
API or the front end. ``test/unit/test_training_metrics.py`` pins that whole
round trip, because a format nobody re-reads is a format that quietly stops
working.

Two rules the format exists to enforce:

**The step is explicit and exact.** It is not taken from ``step:``, which
``format_big_number`` rounds above a thousand (10 500 and 10 600 both print as
``10K``), and not from a line ordinal, which only works because lerobot logs at
one fixed interval. Whoever emits the metric knows the step; they say it.

**The name carries its namespace.** ``train/``, ``eval/`` and ``pred/`` group
the panel grid, so a name without one is refused rather than landing in a
group chosen for it by accident.

Pure: string in, string out. No I/O beyond the ``print`` in :func:`emit`.
"""

from __future__ import annotations

import math
import re
from typing import Any

#: The sentinel a metric line starts with. It is also what the remote grep in
#: ``actoris_harena.training.runs`` keeps, so changing it here changes what survives the
#: trip off the machine -- ``KEEP_PATTERN`` names this constant for that reason.
SENTINEL = "harena-metric"

#: The sentinel this channel used before the pipeline was shared. Logs written
#: by finished and in-flight runs still carry it, and a training log is the only
#: metric record that exists here -- so it is accepted on READ for ever, while
#: nothing emits it any more.
LEGACY_SENTINELS = ("so101-metric",)

#: The groups a metric name may belong to. Not a style rule: the page builds one
#: panel block per group, so an ungrouped name would have nowhere to be drawn.
NAMESPACES = ("train", "eval", "pred")

_NAME = r"[A-Za-z0-9_.\-]+"
_NUMBER = r"-?(?:\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf|-inf)"
# Matches the current sentinel OR any legacy one, so a log written before the
# rename still yields its curves.
_ANY_SENTINEL = (
    "(?:" + "|".join(re.escape(t) for t in (SENTINEL, *LEGACY_SENTINELS)) + ")"
)
_LINE_RE = re.compile(
    _ANY_SENTINEL
    + r"\s+step=(?P<step>\d+)(?P<rest>(?:\s+"
    + _NAME
    + r"/"
    + _NAME
    + r"="
    + _NUMBER
    + r")+)"
)
_PAIR_RE = re.compile(
    r"(?P<key>" + _NAME + r"/" + _NAME + r")=(?P<value>" + _NUMBER + r")"
)


class MetricError(ValueError):
    """A metric that cannot be emitted. The message names what is wrong."""


def check_name(name: str) -> str:
    """Validate a metric name, returning it. Raises :class:`MetricError`.

    Checked at the emitting end rather than the reading end, because a name
    rejected here costs a traceback in the run that produced it, while one
    rejected at the far end costs a silently missing curve nobody looks for.
    """
    group, _, leaf = name.partition("/")
    if not leaf:
        raise MetricError(
            f"metric {name!r} has no namespace. Name it "
            f"{'/, '.join(NAMESPACES)}/... so the page knows which panel block "
            "it belongs in."
        )
    if group not in NAMESPACES:
        raise MetricError(
            f"metric {name!r} is in the unknown group {group!r}; "
            f"want one of {', '.join(NAMESPACES)}."
        )
    if not re.fullmatch(_NAME, group) or not re.fullmatch(_NAME, leaf):
        raise MetricError(
            f"metric {name!r} may use letters, digits, dot, dash and "
            "underscore only -- a space would split the line into two metrics."
        )
    return name


def format_line(step: int, values: "dict[str, float]") -> str:
    """The line :func:`emit` prints, as a string. Pure, so a test can read it."""
    if int(step) < 0:
        raise MetricError(f"step {step} is negative")
    if not values:
        raise MetricError("no metrics given; a line with no values is not one")
    parts = []
    for name, value in values.items():
        check_name(name)
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            # A curve cannot plot these, and writing them down would make a
            # diverged run look like a parse failure rather than a diverged run.
            raise MetricError(
                f"metric {name} is {number}; the run has diverged, which is "
                "worth failing on rather than plotting."
            )
        parts.append(f"{name}={number:.6g}")
    return f"{SENTINEL} step={int(step)} " + " ".join(parts)


def emit(step: int, **values: float) -> str:
    """Print one metric line and return it.

    ``emit(12000, **{"eval/loss": 0.0421})`` -- keyword form works for names
    with no slash in them, which none of ours have, so the mapping form is the
    usual one.
    """
    line = format_line(step, values)
    print(line, flush=True)
    return line


def parse_line(text: str) -> "dict[str, Any] | None":
    """One line back to ``{"step": int, "values": {name: float}}``, or None."""
    match = _LINE_RE.search(text)
    if match is None:
        return None
    values = {
        pair.group("key"): float(pair.group("value"))
        for pair in _PAIR_RE.finditer(match.group("rest"))
    }
    return {"step": int(match.group("step")), "values": values} if values else None


def parse(text: str) -> "list[dict[str, Any]]":
    """Every metric line in a log, in order. Pure."""
    out = []
    for match in _LINE_RE.finditer(text):
        values = {
            pair.group("key"): float(pair.group("value"))
            for pair in _PAIR_RE.finditer(match.group("rest"))
        }
        if values:
            out.append(
                {
                    "at": match.start(),
                    "step": int(match.group("step")),
                    "values": values,
                }
            )
    return out
