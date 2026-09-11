"""Timestamped-history buffers and nearest/interpolated sample selection.

The data-collection recorder aligns every stream to a single reference time per
30 Hz frame instead of grabbing each stream's latest value independently. This
module provides the pure machinery for that:

* :class:`TimestampedHistory` — a small thread-safe ``(t_capture, value)`` ring
  (bounded by age and length) that each publisher appends to and the collector
  samples;
* :func:`select_nearest` / :func:`select_interp` — pick the sample nearest a
  reference time, or interpolate between the two bracketing samples, and report
  the *drift* (the residual gap between the reference time and the nearest real
  sample) so temporal-alignment quality is measurable online and offline.

Interpolation is linear for scalars/vectors and SLERP for orientations, so a
4×4 pose interpolates its translation linearly and its rotation along the
shortest geodesic. Only numpy is used — no MuJoCo, pinocchio or scipy — so the
whole module unit-tests fast.

Timestamps are ``time.monotonic()`` seconds throughout (monotonic for
intervals; wall-clock is only for human-readable stamps elsewhere).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Deque, Sequence

import numpy as np

# ── Sample selection (pure) ──────────────────────────────────────────────────


def nearest_index(times: Sequence[float], t: float) -> int:
    """Index of the entry in ascending ``times`` closest to ``t``.

    ``times`` must be non-empty and sorted ascending (the history keeps it so).
    Ties resolve to the earlier (smaller) index.
    """
    if len(times) == 0:
        raise ValueError("times must be non-empty")
    best_i = 0
    best_d = abs(times[0] - t)
    for i in range(1, len(times)):
        d = abs(times[i] - t)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def select_nearest(
    samples: Sequence[tuple[float, Any]], t: float
) -> tuple[Any, float] | None:
    """Return ``(value, drift)`` for the sample nearest reference time ``t``.

    ``drift`` is ``t - t_capture`` of the chosen sample (positive when the
    reference time is ahead of the data, i.e. the sample is stale). Returns
    ``None`` for an empty history.
    """
    if not samples:
        return None
    times = [s[0] for s in samples]
    i = nearest_index(times, t)
    return samples[i][1], t - times[i]


def select_interp(
    samples: Sequence[tuple[float, Any]],
    t: float,
    interp_fn: Callable[[Any, Any, float], Any],
) -> tuple[Any, float] | None:
    """Interpolate the buffered value to reference time ``t``.

    When ``t`` falls between two samples, ``interp_fn(before, after, frac)``
    blends them (``frac`` in ``[0, 1]``). When ``t`` is outside the buffered
    span the nearest end sample is held (no extrapolation). ``drift`` is
    ``t - t_nearest`` — the gap to the closest *real* sample, so it is ~0 when a
    sample brackets ``t`` tightly and grows when the reference time runs past
    the freshest sample. Returns ``None`` for an empty history.
    """
    if not samples:
        return None
    times = [s[0] for s in samples]
    if t <= times[0]:
        return samples[0][1], t - times[0]
    if t >= times[-1]:
        return samples[-1][1], t - times[-1]
    # Find the bracketing pair (times is ascending).
    hi = 0
    for i in range(1, len(times)):
        if times[i] >= t:
            hi = i
            break
    lo = hi - 1
    t_lo, t_hi = times[lo], times[hi]
    span = t_hi - t_lo
    frac = 0.0 if span <= 0.0 else (t - t_lo) / span
    value = interp_fn(samples[lo][1], samples[hi][1], frac)
    nearest_t = t_lo if (t - t_lo) <= (t_hi - t) else t_hi
    return value, t - nearest_t


# ── Interpolators (pure) ─────────────────────────────────────────────────────


def lerp(a: Any, b: Any, frac: float) -> np.ndarray:
    """Linear blend of two scalars/vectors: ``a + (b - a) * frac``."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return aa + (bb - aa) * float(frac)


def mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix (3×3) → unit quaternion ``[w, x, y, z]`` (numpy only)."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` → rotation matrix (3×3)."""
    q = np.asarray(quat, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_quat(q0: np.ndarray, q1: np.ndarray, frac: float) -> np.ndarray:
    """Shortest-path spherical linear interpolation of ``[w,x,y,z]`` quaternions."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # take the shorter arc
        b = -b
        dot = -dot
    if dot > 0.9995:  # nearly aligned → linear blend + renormalise
        out = a + (b - a) * float(frac)
        return out / (np.linalg.norm(out) or 1.0)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * float(frac)
    sin0 = np.sin(theta0)
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    return s0 * a + s1 * b


def interp_pose(pose_a: np.ndarray, pose_b: np.ndarray, frac: float) -> np.ndarray:
    """Blend two 4×4 homogeneous poses: LERP translation, SLERP rotation."""
    a = np.asarray(pose_a, dtype=np.float64)
    b = np.asarray(pose_b, dtype=np.float64)
    out = np.eye(4)
    out[:3, 3] = lerp(a[:3, 3], b[:3, 3], frac)
    qa = mat_to_quat(a[:3, :3])
    qb = mat_to_quat(b[:3, :3])
    out[:3, :3] = quat_to_mat(slerp_quat(qa, qb, frac))
    return out


# ── History container (thread-safe) ──────────────────────────────────────────


class TimestampedHistory:
    """A bounded, thread-safe ring of ``(t_capture, value)`` samples.

    Publishers call :meth:`append` with a ``time.monotonic()`` stamp; the
    collector calls :meth:`snapshot` and passes the result to
    :func:`select_nearest` / :func:`select_interp`. Old samples are trimmed by
    both age (``max_age_s``) and count (``max_len``) so memory stays bounded
    regardless of stream rate.
    """

    def __init__(self, max_age_s: float = 0.5, max_len: int = 256) -> None:
        self.max_age_s = float(max_age_s)
        self._buf: Deque[tuple[float, Any]] = deque(maxlen=int(max_len))
        self._lock = threading.Lock()

    def append(self, t: float, value: Any) -> None:
        """Append a sample stamped at monotonic time ``t`` and trim old entries."""
        with self._lock:
            self._buf.append((float(t), value))
            cutoff = float(t) - self.max_age_s
            while len(self._buf) > 1 and self._buf[0][0] < cutoff:
                self._buf.popleft()

    def snapshot(self) -> list[tuple[float, Any]]:
        """Return a shallow copy of the buffered samples (ascending by time)."""
        with self._lock:
            return list(self._buf)

    def latest(self) -> tuple[float, Any] | None:
        """Return the most recent ``(t, value)``, or ``None`` if empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None
