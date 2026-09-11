"""How a chunk that arrives late is spliced into the one already executing.

A chunking policy plans many steps from ONE observation. By the time that plan
comes back from another machine, the robot has already moved: it has been
executing the previous plan for the whole round trip. So every arriving chunk
poses the same question, and this module is the answer to it --

    the plan was written for the world as it looked at ``t_obs``;
    it will start being executed at ``t_obs + delay``.
    Which of its actions do we run, and how do we join them to the motion
    already underway?

``delay`` is that gap in control ticks (:func:`delay_ticks`), and it is the only
thing that makes this hard. Get it wrong in one direction and the arms replay
motion the world has moved past; get it wrong in the other and they skip.

THE STRATEGIES. Each takes the actions still queued (``leftover``, next one
first) and the chunk that just landed (``incoming``, whose row 0 was planned for
``t_obs``), and returns the queue to execute from now on.

  * ``append``   -- queue the new chunk BEHIND the leftovers. Nothing is
    discarded and nothing is blended. This is what the rig did before this
    module existed, and its cost is stated below.
  * ``sync``     -- there are never any leftovers, because the caller blocked
    for the reply. Simple and reproducible; the arms pause at every boundary.
  * ``receding`` -- ``sync``, but only the first FRACTION of the chunk is kept
    and the rest is thrown away, so the policy re-observes long before its plan
    runs out. Classic receding horizon, and the baseline the others argue
    against. It BLOCKS: the arms hold still for every round trip, and the name
    describes the horizon, not any concurrency.
  * ``replace``  -- drop the leftovers, drop the first ``delay`` rows of the new
    chunk, execute the rest. Action ``k`` then runs at the time it was planned
    for. The seam is a step change.
  * ``blend``    -- ``replace``, then cross-fade the first ``window`` actions
    from the leftovers into the new chunk. Row 0 is exactly the action that was
    going to be executed anyway, so the join is continuous by construction.
  * ``ensemble`` -- average the overlap instead of choosing: a fixed convex
    combination per timestep, the cheap cousin of ACT's temporal ensembling
    (which needs one forward pass per control step and so cannot be afforded
    over a network).
  * ``rtc``      -- client-side identical to ``replace``; the smoothing happens
    on the GPU host, which guides the denoiser towards the leftovers instead of
    correcting the sample afterwards. Flow-matching policies only.

WHY ``append`` IS THE ONE TO BEAT. It never discards, so the new chunk's row 0
waits for every leftover to drain -- and the request was fired when the queue
fell to the prefetch threshold, which is sized to cover the round trip. The plan
is therefore executed a full threshold behind the observation it was drawn from,
about 0.9 s for a 32-action diffusion chunk at 30 Hz and about 1.1 s for ACT.
The prefetch threshold ends up controlling two unrelated things: how much runway
covers the network, and how stale every executed action is. Every other strategy
here separates them.

Pure: numpy in, numpy out, no clock, no network, no robot.
"""

from __future__ import annotations

import math

import numpy as np

#: Strategy names, in the order they are worth trying.
STRATEGIES = (
    "sync",
    "receding",
    "append",
    "replace",
    "blend",
    "ensemble",
    "rtc",
)

#: Strategies whose smoothing is computed by the policy rather than here. The
#: client-side splice is ``replace``; what differs is that the request carries
#: the leftovers so the sampler can be guided towards them.
GUIDED = ("rtc",)

#: How the cross-fade in ``blend`` travels from the old plan to the new one.
RAMPS = ("linear", "exp")

#: Default weight given to the NEWLY arrived action where two plans overlap,
#: matching the weighting LeRobot's asynchronous client defaults to.
DEFAULT_ENSEMBLE_WEIGHT = 0.7

#: Default cross-fade length, in control ticks.
DEFAULT_BLEND_WINDOW = 5

#: Default fraction of a returned chunk that ``receding`` executes before it
#: asks again. A fraction rather than a count, so it means the same thing
#: against ACT's hundred actions and diffusion's thirty-two.
DEFAULT_EXECUTE_RATIO = 0.5


class ChunkingError(ValueError):
    """A strategy was asked for something it cannot do."""


def delay_ticks(round_trip_s: float, hz: float) -> int:
    """Control ticks that elapse while a request is in flight. Pure.

    Rounded UP: half a tick of staleness is a tick the arms have already spent,
    and discarding one action too many costs a 33 ms gap, where keeping one too
    many replays motion the world has moved past.
    """
    if round_trip_s <= 0.0 or hz <= 0.0:
        return 0
    return int(math.ceil(round_trip_s * hz))


def ramp(window: int, kind: str = "linear") -> np.ndarray:
    """Weights for the new plan across a cross-fade of ``window`` ticks. Pure.

    Starts at 0 -- the first action of the join is exactly what was already
    going to be executed, which is what makes the seam continuous -- and reaches
    1 at the end of the window. ``exp`` stays near the old plan for longer and
    then commits quickly, which suits a policy that has genuinely changed its
    mind: the arms do not lurch, but they do not dawdle either.
    """
    if kind not in RAMPS:
        raise ChunkingError(f"unknown ramp: {kind} (want one of {', '.join(RAMPS)})")
    if window <= 0:
        return np.zeros(0, dtype=float)
    if window == 1:
        return np.zeros(1, dtype=float)
    linear = np.linspace(0.0, 1.0, window, endpoint=False)
    if kind == "linear":
        return linear
    # expm1 normalised to [0, 1): same endpoints, slower to leave the old plan.
    return np.expm1(linear) / math.expm1(1.0)


def _as_2d(actions, name: str) -> np.ndarray:
    arr = np.asarray(actions, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 0) if arr.ndim < 2 else arr
    if arr.ndim != 2:
        raise ChunkingError(f"{name} must be (rows, dim), got shape {arr.shape}")
    return arr


def splice(
    strategy: str,
    leftover,
    incoming,
    delay: int = 0,
    *,
    window: int = DEFAULT_BLEND_WINDOW,
    ramp_kind: str = "linear",
    new_weight: float = DEFAULT_ENSEMBLE_WEIGHT,
    execute_ratio: float = DEFAULT_EXECUTE_RATIO,
) -> np.ndarray:
    """The queue to execute once ``incoming`` lands. Pure.

    ``leftover`` is what is still queued, next action first; ``incoming`` is the
    chunk that just arrived, row 0 planned for the observation's own tick.
    ``delay`` is how many ticks passed in between (:func:`delay_ticks`).

    Returns a fresh ``(rows, dim)`` array. An empty result is legitimate -- a
    round trip longer than the whole chunk leaves nothing worth executing -- and
    the caller must treat it as a stall rather than as an error.
    """
    if strategy not in STRATEGIES:
        raise ChunkingError(
            f"unknown strategy: {strategy} (want one of {', '.join(STRATEGIES)})"
        )
    old = _as_2d(leftover, "leftover")
    new = _as_2d(incoming, "incoming")
    if new.size == 0:
        raise ChunkingError("an arriving chunk must have at least one action")
    delay = max(0, int(delay))

    if strategy == "append":
        # Kept exactly as the rig behaved before: nothing dropped, nothing
        # aligned. The lag this produces is the reason for every other branch.
        if old.size == 0:
            return new.copy()
        _check_dims(old, new)
        return np.concatenate([old, new], axis=0)

    if strategy in ("sync", "receding"):
        # The caller blocked for this reply, so the arms did not move while it
        # was in flight and row 0 is still the right place to start. A leftover
        # here means the caller is not actually synchronous.
        if old.size:
            raise ChunkingError(
                f"{strategy} spliced a chunk while actions were still queued: "
                "a blocking caller must drain before it requests"
            )
        if strategy == "sync":
            return new.copy()
        # Receding: keep the front of the plan and discard the rest, so the
        # next observation is taken while this one is still recent. At least
        # one action, or the run makes no progress at all.
        if not 0.0 < execute_ratio <= 1.0:
            raise ChunkingError(f"execute_ratio must be in (0, 1], got {execute_ratio}")
        keep = max(1, int(math.ceil(execute_ratio * len(new))))
        return new[:keep].copy()

    # Everything below aligns the new plan to the present: rows 0..delay-1 were
    # planned for ticks that have already been executed from the previous chunk.
    aligned = new[delay:]
    if aligned.size == 0:
        return np.zeros((0, new.shape[1]), dtype=float)

    if strategy in ("replace",) + GUIDED:
        return aligned.copy()

    if old.size == 0:
        # Nothing to join to -- the first chunk of a run, or a queue that ran
        # dry. Both blend and ensemble degrade to replace, which is correct.
        return aligned.copy()
    _check_dims(old, new)

    if strategy == "ensemble":
        overlap = min(len(old), len(aligned))
        out = aligned.copy()
        w = float(new_weight)
        if not 0.0 <= w <= 1.0:
            raise ChunkingError(f"new_weight must be in [0, 1], got {w}")
        out[:overlap] = (1.0 - w) * old[:overlap] + w * aligned[:overlap]
        return out

    # blend
    span = min(int(window), len(old), len(aligned))
    if span <= 0:
        return aligned.copy()
    weights = ramp(span, ramp_kind).reshape(-1, 1)
    out = aligned.copy()
    out[:span] = (1.0 - weights) * old[:span] + weights * aligned[:span]
    return out


def _check_dims(old: np.ndarray, new: np.ndarray) -> None:
    if old.shape[1] != new.shape[1]:
        raise ChunkingError(
            f"action width changed mid-run: queued {old.shape[1]}, "
            f"arrived {new.shape[1]}"
        )


def plan_lag(strategy: str, leftover_len: int, delay: int) -> int:
    """Ticks by which execution runs BEHIND the plan it is executing. Pure.

    An action written for tick ``k`` of a chunk should be executed at tick ``k``,
    counted from the observation. This returns how much later it actually runs,
    and it is the number these strategies exist to drive to zero.

    ``append`` pays the round trip AND the whole queue the chunk was put behind.
    Every aligning strategy pays nothing: it discards precisely the rows whose
    moment has passed, so the first action it executes is the one meant for now.
    ``sync`` pays nothing either, for the opposite reason -- the arms held still
    for the reply, so no tick of the plan went by unexecuted.
    """
    if strategy not in STRATEGIES:
        raise ChunkingError(f"unknown strategy: {strategy}")
    if strategy == "append":
        return max(0, int(delay)) + max(0, int(leftover_len))
    return 0
