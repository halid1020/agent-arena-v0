"""A crash-proof record of the episodes a chunking sweep has already run.

A full grid is seventeen cells of tens of episodes, which is hours. The sweep
used to hold every result in memory and write its report only once the last
episode finished, so anything that killed the process -- a host that refused a
request, a full disk, an operator's Ctrl+C -- threw away every hour of it. That
is not a theoretical worry: a seventeen-cell run died in its second cell and
left nothing behind but console output.

So each finished episode is appended here as one line of JSON the moment it is
scored, and the report is composed from the file rather than from memory. A
resumed sweep reads the journal, skips the episodes it names, and carries on.

THE KEY IS THE CELL AND THE TRIAL INDEX, not the seed. The overfit-one-scenario
protocol repeats a single seed for every trial, so a seed alone cannot say which
of ten repeats a row is.

Pure except for the file: no numpy, no network, no clock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def journal_for(out_path: "str | Path") -> Path:
    """The journal that belongs beside a report. ``x.md`` -> ``x.episodes.jsonl``."""
    p = Path(out_path)
    return p.with_suffix(p.suffix + ".episodes.jsonl")


def row_key(cell: str, trial: int) -> str:
    """The identity of one episode within a sweep."""
    return f"{cell}#{int(trial)}"


def append_row(path: "str | Path", row: Mapping[str, Any]) -> None:
    """Append one scored episode. Creates the file and its parent if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True) + "\n")
        fh.flush()


def load_rows(path: "str | Path") -> "list[dict]":
    """Every episode the journal holds, in the order they were scored.

    A half-written final line -- the process died mid-append -- is dropped
    rather than raising: the point of the journal is to survive a crash, so it
    must tolerate the crash's own last write.
    """
    p = Path(path)
    if not p.is_file():
        return []
    rows: "list[dict]" = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def done_keys(rows: Iterable[Mapping[str, Any]]) -> "set[str]":
    """The episodes already scored, as keys ``row_key`` would produce."""
    keys: "set[str]" = set()
    for row in rows:
        cell = row.get("cell")
        trial = row.get("trial")
        if cell is None or trial is None:
            continue
        keys.add(row_key(str(cell), int(trial)))
    return keys
