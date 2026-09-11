"""Per-frame temporal-alignment telemetry for the data-collection recorder.

Each recorded frame is assembled at a single reference time; every stream is
sampled at that time and reports a *drift* — the residual seconds between the
reference time and the stream's nearest real sample. :class:`DriftLog` buffers
those drifts per episode, prints a compact summary, and writes a per-episode
``<root>/extra/drift_XXXXXX.parquet`` so alignment quality is auditable offline
alongside the ~100 Hz sidecar.

The summary statistics are pure (numpy only); parquet writing imports pyarrow
locally (same pattern as ``sidecar.py``) so the module stays import-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class DriftLog:
    """Buffer and summarise per-frame per-stream drift (seconds)."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._rows = []

    def add(self, frame_index: int, t_ref: float, drifts: dict[str, float]) -> None:
        """Record one frame's drifts. ``drifts`` maps stream name → seconds."""
        row: dict[str, Any] = {"frame_index": int(frame_index), "t_ref": float(t_ref)}
        for name, d in drifts.items():
            row[f"drift_ms_{name}"] = float(d) * 1e3
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return per-stream ``{mean_ms, p95_abs_ms, max_abs_ms}`` over the episode.

        p95/max use the absolute drift (magnitude of misalignment either way);
        mean keeps its sign so a consistent lead/lag is visible.
        """
        if not self._rows:
            return {}
        streams = [
            k[len("drift_ms_") :] for k in self._rows[0] if k.startswith("drift_ms_")
        ]
        out: dict[str, dict[str, float]] = {}
        for name in streams:
            vals = np.array(
                [r[f"drift_ms_{name}"] for r in self._rows], dtype=np.float64
            )
            absvals = np.abs(vals)
            out[name] = {
                "mean_ms": float(np.mean(vals)),
                "p95_abs_ms": float(np.percentile(absvals, 95)),
                "max_abs_ms": float(np.max(absvals)),
            }
        return out

    def format_summary(self) -> str:
        """One-line-per-stream human summary (worst streams first)."""
        summ = self.summary()
        if not summ:
            return "no drift samples"
        lines = []
        for name in sorted(summ, key=lambda n: -summ[n]["p95_abs_ms"]):
            s = summ[name]
            lines.append(
                f"    {name:<20} mean {s['mean_ms']:+6.1f} ms  "
                f"p95 {s['p95_abs_ms']:5.1f} ms  max {s['max_abs_ms']:5.1f} ms"
            )
        return "\n".join(lines)

    def write_parquet(self, root: str | Path, ep_idx: int) -> Path | None:
        """Write the buffered rows to ``<root>/extra/drift_XXXXXX.parquet``."""
        if not self._rows:
            return None
        import pyarrow as pa  # type: ignore[import]
        import pyarrow.parquet as pq  # type: ignore[import]

        extra_dir = Path(root) / "extra"
        extra_dir.mkdir(parents=True, exist_ok=True)
        out_path = extra_dir / f"drift_{ep_idx:06d}.parquet"
        columns = list(self._rows[0].keys())
        table = pa.Table.from_pydict(
            {col: [r[col] for r in self._rows] for col in columns}
        )
        pq.write_table(table, out_path)
        return out_path
