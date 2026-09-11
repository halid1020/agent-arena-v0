"""Which (strategy, hyper-parameter) combinations a chunking sweep should run.

``common.chunking`` says what each splice DOES; this module says which ones are
worth measuring. They are separate because the strategies are not comparable as
bare names: ``receding`` at an execute ratio of 1.0 is a different animal from
``receding`` at 0.25, and a ``blend`` over three ticks answers the seam question
differently from one over ten. A sweep that varies only the name therefore
compares one arbitrary tuning of each strategy against another, and its ranking
is an artefact of the defaults. So a sweep cell here is a strategy TOGETHER with
the numbers it consumes, and the grid is the thing that gets run.

THE BUILT-IN GRID (``expand_grid("all")``) is seventeen cells:

  * ``sync``, ``append``, ``replace`` -- one each; none of them reads a
    parameter, so there is nothing to vary.
  * ``receding`` at execute ratios 0.25 / 0.5 / 0.75 / 1.0 -- the whole span
    from "re-observe four times a chunk" to "run the plan out", which is the
    axis that decides whether its blocking is paid for.
  * ``blend`` over windows 3 / 5 / 10 ticks, each with a linear and an
    exponential ramp -- short enough to be a join, long enough to be a
    refusal to act on the new plan, and both shapes of getting there.
  * ``ensemble`` at new-plan weights 0.3 / 0.5 / 0.7 / 0.9 -- from trusting
    what is already queued to nearly discarding it.

``rtc`` IS DELIBERATELY LEFT OUT. Its smoothing happens inside the denoiser on
the GPU host, so it is only a distinct strategy when the host advertises
``meta["rtc"]``; against a host that does not, the client-side splice is exactly
``replace`` and a row labelled ``rtc`` would be a duplicate of one already in the
grid, wearing a name nobody could trust afterwards. ``tool/policy_server.py``
does not advertise it today. Ask for it explicitly -- ``--grid rtc`` -- once a
host does, and the handshake in ``policy_client`` will refuse if it does not.

THE CUSTOM SPEC is a ``;``-separated list of terms, one term per strategy::

    strategy[:param=v1,v2][:param=v3,v4]

``:`` introduces a parameter, ``,`` separates the values that parameter takes,
and every listed parameter of a term is cartesian-producted with the others. So

    receding:execute_ratio=0.5,1.0;blend:blend_window=5:ramp_kind=linear,exp

is four cells: receding at each of two ratios, and a five-tick blend with each
of two ramps. A term with no parameters (``append``, or ``blend`` on its own)
is one cell carrying that strategy's defaults, so the results table always
states what was actually run rather than leaving it to be guessed.

Parameter names are the ones ``RemoteActionSource.set_strategy`` accepts --
``execute_ratio``, ``blend_window``, ``new_weight``, ``ramp_kind`` -- so a cell
can be splatted straight into it. Naming a parameter a strategy does not read is
an error rather than a no-op: it would silently produce duplicate cells that
differ only in a label, and two identical rows in a comparison table are worse
than a refusal.

Pure: a string in, a list of dicts out. No numpy, no network, no clock.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Mapping, Sequence

from actoris_harena.deploy.chunking import (
    DEFAULT_BLEND_WINDOW,
    DEFAULT_ENSEMBLE_WEIGHT,
    DEFAULT_EXECUTE_RATIO,
    RAMPS,
    STRATEGIES,
)

#: The parameters each strategy actually reads, in the order a label shows them.
#: A strategy absent from a splice's arithmetic gets an empty tuple, and that is
#: what makes "one cell" the right answer for it.
CONSUMES: "dict[str, tuple[str, ...]]" = {
    "sync": (),
    "receding": ("execute_ratio",),
    "append": (),
    "replace": (),
    "blend": ("blend_window", "ramp_kind"),
    "ensemble": ("new_weight",),
    "rtc": (),
}

#: What a term fills in for a parameter it does not list. Kept identical to the
#: splice's own defaults so an unparameterised term means "as shipped".
DEFAULTS: "dict[str, Any]" = {
    "execute_ratio": DEFAULT_EXECUTE_RATIO,
    "blend_window": DEFAULT_BLEND_WINDOW,
    "new_weight": DEFAULT_ENSEMBLE_WEIGHT,
    "ramp_kind": "linear",
}

#: Strategies the built-in grid runs. See the module docstring for why ``rtc``
#: is not among them.
GRID_STRATEGIES = ("sync", "append", "replace", "receding", "blend", "ensemble")

#: The axes the built-in grid sweeps.
GRID_AXES: "dict[str, dict[str, tuple]]" = {
    "receding": {"execute_ratio": (0.25, 0.5, 0.75, 1.0)},
    "blend": {"blend_window": (3, 5, 10), "ramp_kind": ("linear", "exp")},
    "ensemble": {"new_weight": (0.3, 0.5, 0.7, 0.9)},
}

#: One line of ``--help``, kept here so the tool and this module cannot drift.
SPEC_HELP = (
    "'all' for the built-in 17-cell grid, or a ';'-separated list of "
    "strategy[:param=v1,v2] terms, e.g. "
    "'receding:execute_ratio=0.5,1.0;blend:blend_window=5:ramp_kind=linear,exp'. "
    "Parameters are execute_ratio, blend_window, new_weight, ramp_kind; a "
    "strategy may only be given the ones it reads."
)


class SweepSpecError(ValueError):
    """A grid spec asked for something that cannot be run."""


def _parse_ratio(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"execute_ratio must be in (0, 1], got {value}")
    return value


def _parse_weight(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"new_weight must be in [0, 1], got {value}")
    return value


def _parse_window(text: str) -> int:
    value = int(text)
    if value < 1:
        raise ValueError(f"blend_window must be at least 1, got {value}")
    return value


def _parse_ramp(text: str) -> str:
    if text not in RAMPS:
        raise ValueError(f"unknown ramp '{text}' (want one of {', '.join(RAMPS)})")
    return text


#: How each parameter's written value becomes the value the splice consumes.
#: Each raises ``ValueError`` on a value outside the range the splice accepts,
#: so a nonsense grid fails at parse time rather than an hour into a sweep.
PARSERS: "dict[str, Callable[[str], Any]]" = {
    "execute_ratio": _parse_ratio,
    "blend_window": _parse_window,
    "new_weight": _parse_weight,
    "ramp_kind": _parse_ramp,
}

#: How each parameter is written into a cell label: short, and unambiguous
#: between the parameters of one strategy.
_LABEL: "dict[str, Callable[[Any], str]]" = {
    "execute_ratio": lambda v: f"{float(v):g}",
    "blend_window": lambda v: f"w{int(v)}",
    "new_weight": lambda v: f"{float(v):g}",
    "ramp_kind": str,
}


def cell_label(cell: Mapping[str, Any]) -> str:
    """A short stable name for one sweep cell -- the row key in the table.

    ``receding@0.25``, ``blend@w5/exp``, ``ensemble@0.7``; a strategy that reads
    nothing is just its own name. Stable because it is derived from the cell and
    nothing else, so the same cell names the same row in every run, and a table
    from last week can be read beside one from today.
    """
    strategy = str(cell["strategy"])
    parts = [
        _LABEL[name](cell[name])
        for name in CONSUMES.get(strategy, ())
        if cell.get(name) is not None
    ]
    return strategy if not parts else f"{strategy}@{'/'.join(parts)}"


def _cells(
    strategy: str, axes: Mapping[str, Sequence[Any]], defaults: Mapping[str, Any]
) -> "list[dict]":
    """The cartesian product of one strategy's listed values. Pure."""
    names = CONSUMES[strategy]
    values = [
        tuple(axes[name]) if name in axes else (defaults[name],) for name in names
    ]
    return [
        {"strategy": strategy, **dict(zip(names, combo))} for combo in product(*values)
    ]


def _parse_term(term: str, defaults: Mapping[str, Any]) -> "list[dict]":
    """One ``strategy[:param=v1,v2]`` term -> its cells. Raises on nonsense."""
    fields = [f.strip() for f in term.split(":")]
    strategy = fields[0]
    if strategy not in STRATEGIES:
        raise SweepSpecError(
            f"unknown strategy in '{term}': '{strategy}' "
            f"(want one of {', '.join(STRATEGIES)})"
        )
    axes: "dict[str, tuple]" = {}
    for field in fields[1:]:
        if "=" not in field:
            raise SweepSpecError(
                f"malformed term '{term}': '{field}' is not param=value"
            )
        name, _, written = field.partition("=")
        name = name.strip()
        if name not in PARSERS:
            raise SweepSpecError(
                f"unknown parameter '{name}' in '{term}' "
                f"(want one of {', '.join(sorted(PARSERS))})"
            )
        if name not in CONSUMES[strategy]:
            consumed = ", ".join(CONSUMES[strategy]) or "no parameters"
            raise SweepSpecError(
                f"'{strategy}' does not read '{name}' in '{term}' "
                f"(it reads {consumed})"
            )
        if name in axes:
            raise SweepSpecError(f"'{name}' given twice in '{term}'")
        written_values = [v.strip() for v in written.split(",") if v.strip()]
        if not written_values:
            raise SweepSpecError(f"malformed term '{term}': '{name}' has no value")
        try:
            axes[name] = tuple(PARSERS[name](v) for v in written_values)
        except ValueError as exc:
            raise SweepSpecError(f"bad value in '{term}': {exc}") from exc
    return _cells(strategy, axes, defaults)


def expand_grid(
    spec: str, *, defaults: "Mapping[str, Any] | None" = None
) -> "list[dict]":
    """A grid spec -> the sweep cells to run, in the order to run them. Pure.

    Each cell is ``{"strategy": name, **the parameters that strategy reads}``,
    ready to be splatted into ``RemoteActionSource.set_strategy`` or handed to
    its constructor. ``spec`` is ``"all"`` for the built-in grid, or the
    ``;``-separated term syntax documented at the top of this module.

    ``defaults`` overrides what an unlisted parameter is filled in with, so a
    tool whose ``--blend-window`` flag still means something can keep meaning it
    for a term that does not name a window. It never touches a value the spec
    states, so ``expand_grid("all")`` answers the same way whatever is passed.

    Raises :class:`SweepSpecError` -- a ``ValueError`` -- naming the offending
    term for anything malformed, an unknown strategy, a parameter the strategy
    does not read, or a value outside the range the splice accepts.
    """
    filled = {**DEFAULTS, **dict(defaults or {})}
    text = (spec or "").strip()
    if not text:
        raise SweepSpecError("empty grid spec (want 'all' or strategy[:param=v,...])")

    if text == "all":
        cells: "list[dict]" = []
        for strategy in GRID_STRATEGIES:
            cells.extend(_cells(strategy, GRID_AXES.get(strategy, {}), filled))
    else:
        cells = []
        for term in text.split(";"):
            if not term.strip():
                raise SweepSpecError(f"empty term in grid spec '{spec}'")
            cells.extend(_parse_term(term, filled))

    # Two rows that differ only in which term produced them would be two runs of
    # the same experiment under one label, so the later duplicate is dropped.
    seen: "set[str]" = set()
    unique: "list[dict]" = []
    for cell in cells:
        label = cell_label(cell)
        if label in seen:
            continue
        seen.add(label)
        unique.append(cell)
    return unique
