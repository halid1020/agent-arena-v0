"""Judging a chunking strategy by what the arms actually did.

Success rate alone cannot separate these strategies: on an easy scene every one
of them succeeds, and on a hard one the noise swamps the difference. What
distinguishes them is visible in the executed trajectory itself, at the ticks
where one plan gave way to the next.

THE SEAM RATIO is the number to read first. Within a chunk the actions come from
one forward pass and move smoothly; at a seam they come from two, and any
disagreement between the plans shows up as a step change in the commanded joint
vector. So compare the size of the step AT the seams with the size of the step
everywhere else:

    1.0  the joins are invisible -- a seam looks like any other tick
    3.0  every chunk boundary is a visible flinch
   10.0  the arms lurch, and a human watching would call it broken

It is a ratio rather than an absolute so that a slow careful task and a fast one
can be compared, and so that a policy that simply moves more does not look worse.

A caveat worth keeping in mind while reading it: a strategy can buy a beautiful
ratio by ignoring new information -- ``ensemble`` with a low weight on the new
plan, or a very long ``blend`` window, will smooth a seam by refusing to act on
what the policy just saw. Read the ratio beside the success rate and the hold
count, never alone.

Pure: arrays in, numbers out.
"""

from __future__ import annotations

import numpy as np

#: A run with fewer than this many seams cannot say anything about seams.
MIN_SEAMS = 2


def step_sizes(actions) -> np.ndarray:
    """Euclidean distance between consecutive commanded actions. Pure.

    Returns one value per transition, so ``step_sizes(a)[i]`` is the move from
    ``a[i]`` to ``a[i + 1]``.
    """
    arr = np.asarray(actions, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"actions must be (ticks, dim), got shape {arr.shape}")
    if len(arr) < 2:
        return np.zeros(0, dtype=float)
    return np.linalg.norm(np.diff(arr, axis=0), axis=1)


def seam_ratio(actions, seams) -> "float | None":
    """How much bigger a step is at a chunk boundary than anywhere else. Pure.

    ``seams`` are tick indices at which a NEW chunk's first action was executed,
    so the transition of interest is the one INTO that tick: ``seam - 1``.

    Returns ``None`` rather than a number when the run cannot support the
    comparison -- too few seams, or a trajectory that barely moved, where the
    ratio would be dominated by numerical noise and would read as a result.
    """
    steps = step_sizes(actions)
    if len(steps) == 0:
        return None
    at_seam = sorted({int(s) - 1 for s in seams if 0 < int(s) <= len(steps)})
    if len(at_seam) < MIN_SEAMS:
        return None
    mask = np.zeros(len(steps), dtype=bool)
    mask[at_seam] = True
    within = steps[~mask]
    if len(within) == 0:
        return None
    baseline = float(np.median(within))
    if baseline <= 1e-9:
        return None
    return float(np.median(steps[mask]) / baseline)


def path_length(actions) -> float:
    """Total distance travelled through action space. Pure.

    A strategy that thrashes between two plans covers more ground for the same
    task, which is the cost that a seam ratio near 1 can hide.
    """
    return float(step_sizes(actions).sum())


def summarise(
    actions,
    seams,
    *,
    held_ticks: int = 0,
    total_ticks: "int | None" = None,
    success: "bool | None" = None,
    round_trips_s=(),
    plan_lag_ticks: "int | None" = None,
) -> dict:
    """One rollout -> the numbers a strategy is compared on. Pure.

    ``held_ticks`` are ticks with nothing to command -- the queue ran dry and
    the arms kept their last goal. On a strategy that discards stale rows this
    is the price paid for alignment, and it is what stops ``replace`` from being
    free.
    """
    arr = np.asarray(actions, dtype=float)
    ticks = int(total_ticks if total_ticks is not None else len(arr))
    trips = np.asarray(list(round_trips_s), dtype=float)
    return {
        "ticks": ticks,
        "chunks": int(len(list(seams))),
        "success": None if success is None else bool(success),
        "seam_ratio": seam_ratio(arr, seams),
        "path_length": path_length(arr),
        "held_ticks": int(held_ticks),
        "held_fraction": (float(held_ticks) / ticks) if ticks else 0.0,
        "plan_lag_ticks": plan_lag_ticks,
        "round_trip_ms_median": (float(np.median(trips) * 1e3) if trips.size else None),
        "round_trip_ms_max": (float(trips.max() * 1e3) if trips.size else None),
    }


def compare(rows: "list[dict]") -> str:
    """A markdown table of one summary per segment, best seam first. Pure.

    Sorted by the seam ratio because that is the question these strategies were
    written to answer; the columns beside it are there so a smooth-but-useless
    result cannot pass unnoticed.

    A row also carries the TRIAL it was measured in, where the caller counts
    them. Two attempts under one splice are two rows and not an average: they
    were run against a scene that was put back by hand in between, and the
    difference between them is often the whole of what was being looked at.
    """
    if not rows:
        return "_no runs_\n"

    def key(row: dict):
        ratio = row.get("seam_ratio")
        return (ratio is None, ratio if ratio is not None else 0.0)

    header = (
        "| trial | strategy | success | seam ratio | held | path | rtt median |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in sorted(rows, key=key):
        ratio = row.get("seam_ratio")
        success = row.get("success")
        rtt = row.get("round_trip_ms_median")
        trial = row.get("trial")
        lines.append(
            "| {trial} | {name} | {ok} | {ratio} | {held:.0%} | {path:.1f} "
            "| {rtt} |\n".format(
                trial="—" if trial is None else int(trial),
                name=row.get("strategy", "?"),
                ok="—" if success is None else ("yes" if success else "no"),
                ratio="—" if ratio is None else f"{ratio:.2f}",
                held=float(row.get("held_fraction", 0.0) or 0.0),
                path=float(row.get("path_length", 0.0) or 0.0),
                rtt="—" if rtt is None else f"{rtt:.0f} ms",
            )
        )
    return header + "".join(lines)
