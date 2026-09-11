"""Is this dataset whole, and if not, what exactly is missing?

A LeRobot dataset counts its episodes in ``meta/info.json`` and stores them in
three parallel places: a row in ``meta/episodes/``, frames in ``data/``, and our
own per-episode side files under ``extra/``. Nothing keeps those in step. A
recording that is counted and then not written -- an episode aborted between the
counter and the files -- leaves a PHANTOM: an index the dataset believes in and
nothing on disk backs.

That is not a cosmetic fault. ``DatasetReader._check_cached_episodes_sufficient``
requires ``set(range(total_episodes))`` to be a subset of the episodes actually
present, so one phantom makes LeRobot judge the whole local copy incomplete,
and ``LeRobotDataset.__init__`` then goes to the Hub for a version tag. Offline
-- which is how this rig runs on purpose -- that surfaces as
``OfflineModeIsEnabled: Cannot reach https://huggingface.co/...``, an error about
a network, for a dataset that never left the drive, raised by every rewrite path
there is: compaction, deletion and merging alike.

So this module reads the dataset the way the dataset itself is written --
parquet and JSON, no LeRobot import -- and can therefore describe one that
LeRobot refuses to open. ``dataset_integrity`` says what is wrong,
``ensure_loadable`` is the guard every rewrite calls first, and
``repair_phantom_episodes`` removes an episode nobody wrote.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from actoris_harena.recording.dataset_edit import (
    episode_meta_files,
    extra_reindex_ops,
    read_soft_deleted,
    repair_episode_metadata,
    write_soft_deleted,
)

EPISODE_KEY = "episode_index"
LENGTH_KEY = "length"
DATA_CHUNK_KEY = "data/chunk_index"
DATA_FILE_KEY = "data/file_index"
FROM_KEY = "dataset_from_index"
TO_KEY = "dataset_to_index"
EPISODE_STAT_PREFIX = "stats/episode_index/"
#: Statistics of the episode-index column that are simply the episode number.
#: ``std`` is zero for a single episode and ``count`` is its length, so both
#: survive a renumbering untouched.
EPISODE_STAT_KEEP = ("std", "count")

#: Per-camera columns of the episode metadata: where an episode's frames sit in
#: that camera's video file.
VIDEO_PREFIX = "videos/observation.images."
VIDEO_FROM = "from_timestamp"
VIDEO_TO = "to_timestamp"


class DatasetDamaged(Exception):
    """A dataset cannot be rewritten until something on disk is put right.

    Raised instead of letting LeRobot mistake a local gap for a missing download
    and report it as an unreachable Hugging Face endpoint.
    """


def _read_json(path: Path) -> "dict[str, Any]":
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def data_files(root: Path) -> "list[Path]":
    return sorted(Path(root).glob("data/chunk-*/file-*.parquet"))


def episodes_in_metadata(root: Path) -> "dict[int, dict[str, Any]]":
    """``episode -> {length, from, to, file}`` from ``meta/episodes/``."""
    import pyarrow.parquet as pq  # type: ignore[import]

    wanted = [
        EPISODE_KEY,
        LENGTH_KEY,
        FROM_KEY,
        TO_KEY,
        DATA_CHUNK_KEY,
        DATA_FILE_KEY,
    ]
    found: dict[int, dict[str, Any]] = {}
    for path in episode_meta_files(Path(root)):
        # Read the six bookkeeping columns, not the table: most of an episode
        # row is per-feature statistics, and a session-sized dataset has one
        # file per episode. Reading whole tables here costs seconds per click.
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if EPISODE_KEY not in names:
            continue
        table = pq.read_table(path, columns=[k for k in wanted if k in names])
        columns = {
            key: table.column(key).to_pylist()
            for key in wanted
            if key in table.column_names
        }
        for row in range(table.num_rows):
            index = int(columns[EPISODE_KEY][row])
            found[index] = {
                "length": _at(columns, LENGTH_KEY, row),
                "from": _at(columns, FROM_KEY, row),
                "to": _at(columns, TO_KEY, row),
                "data_chunk": _at(columns, DATA_CHUNK_KEY, row),
                "data_file": _at(columns, DATA_FILE_KEY, row),
                "file": path,
            }
    return found


def camera_span_problems(root: Path) -> "list[dict[str, Any]]":
    """Episodes whose video does not last as long as their rows say. Cheap.

    A camera that stops delivering mid-episode leaves the recording SHORTER than
    the frames the dataset claims for it, and nothing else notices: the metadata
    row, the parquet rows and the side files are all written, and the counts in
    ``info.json`` add up. The dataset opens, trains, and then dies partway
    through the first pass with a ``FrameTimestampError`` naming a query the
    video cannot answer -- on a rented GPU, hours from the operator.

    MEASURED on ``fold-short-from-flattend-tactile``: episode 63 held 260 rows
    and 260 frames from each of the four fingertip cameras, but only 213 from
    the overhead one. Every other check in this module passed it.

    The invariant is per camera and per episode: ``to - from == length / fps``.
    It reads two more columns from metadata already being read, so it costs
    nothing and needs no video decoded.
    """
    import pyarrow.parquet as pq  # type: ignore[import]

    root = Path(root)
    fps = float(_read_json(root / "meta" / "info.json").get("fps") or 0)
    if not fps:
        return []
    # Half a frame: wide enough for float noise in a stored timestamp, far
    # narrower than the smallest dropout worth reporting.
    tolerance = 0.5 / fps

    problems: "list[dict[str, Any]]" = []
    for path in episode_meta_files(root):
        names = list(pq.ParquetFile(path).schema_arrow.names)
        cameras = sorted(
            {
                n[len(VIDEO_PREFIX) : n.rindex("/")]
                for n in names
                if n.startswith(VIDEO_PREFIX) and n.endswith("/" + VIDEO_TO)
            }
        )
        if not cameras or EPISODE_KEY not in names or LENGTH_KEY not in names:
            continue
        wanted = [EPISODE_KEY, LENGTH_KEY] + [
            f"{VIDEO_PREFIX}{c}/{end}"
            for c in cameras
            for end in (VIDEO_FROM, VIDEO_TO)
        ]
        table = pq.read_table(path, columns=[c for c in wanted if c in names])
        columns = {k: table.column(k).to_pylist() for k in table.column_names}
        for row in range(table.num_rows):
            length = columns[LENGTH_KEY][row]
            if not length:
                continue
            want = float(length) / fps
            for camera in cameras:
                start = columns.get(f"{VIDEO_PREFIX}{camera}/{VIDEO_FROM}", [None])[row]
                end = columns.get(f"{VIDEO_PREFIX}{camera}/{VIDEO_TO}", [None])[row]
                if start is None or end is None:
                    continue
                span = float(end) - float(start)
                if abs(span - want) > tolerance:
                    problems.append(
                        {
                            "episode": int(columns[EPISODE_KEY][row]),
                            "camera": camera,
                            "rows": int(length),
                            "frames": round(span * fps),
                        }
                    )
    return sorted(problems, key=lambda p: (p["episode"], p["camera"]))


def _at(columns: "dict[str, list]", key: str, row: int) -> "int | None":
    values = columns.get(key)
    return None if values is None else int(values[row])


def episodes_in_data(root: Path) -> "tuple[dict[int, int], int]":
    """``episode -> frames`` from the data files, and the total row count."""
    import pyarrow.parquet as pq  # type: ignore[import]

    counts: dict[int, int] = {}
    total = 0
    for path in data_files(Path(root)):
        table = pq.read_table(path, columns=[EPISODE_KEY])
        total += table.num_rows
        for index in table.column(EPISODE_KEY).to_pylist():
            counts[int(index)] = counts.get(int(index), 0) + 1
    return counts, total


def _has_data(root: Path, row: "dict[str, Any]") -> bool:
    """Does the data file this metadata row names exist? Cheap, no read."""
    chunk, index = row.get("data_chunk"), row.get("data_file")
    if chunk is None or index is None:
        return True  # the row does not say; the deep pass is the one that knows
    return (Path(root) / f"data/chunk-{chunk:03d}/file-{index:03d}.parquet").is_file()


def offsets_consistent(episodes: "dict[int, dict[str, Any]]") -> bool:
    """Are ``dataset_from_index``/``to_index`` the cumulative episode lengths?

    That is the invariant a healthy dataset holds (verified against the rig's
    own collections) and what LeRobot slices episodes with. A phantom leaves a
    hole in it: the last episode's end runs past the end of the data.
    """
    at = 0
    for index in sorted(episodes):
        row = episodes[index]
        length = row.get("length")
        if length is None or row.get("from") is None or row.get("to") is None:
            return False
        if row["from"] != at or row["to"] != at + length:
            return False
        at += int(length)
    return True


def dataset_integrity(root: Path, deep: bool = True) -> "dict[str, Any]":
    """Everything wrong with the dataset on disk, in one pass.

    ``phantom`` episodes are counted by ``info.json`` and present nowhere --
    repairable, because there is nothing to lose. An ``orphan`` (in the metadata
    but not the data, or the reverse) is NOT repaired here: real frames or a real
    row would have to be invented or discarded, and which of those is right
    depends on what happened, so it is reported for a person to decide.

    ``deep`` reads the episode index out of every data file, which is the only
    way to see frames that no metadata row claims. It costs a column read per
    file, so the episode list (which asks on every click, only to explain a
    ``?f``) asks for the shallow answer: the metadata rows, and whether the file
    each one names exists.
    """
    root = Path(root)
    info = _read_json(root / "meta" / "info.json")
    counted = int(info.get("total_episodes") or 0)
    in_meta = episodes_in_metadata(root)
    if deep:
        in_data, frames_present = episodes_in_data(root)
    else:
        in_data = {
            i: int(r["length"] or 0) for i, r in in_meta.items() if _has_data(root, r)
        }
        frames_present = sum(in_data.values())

    everywhere = set(in_meta) & set(in_data)
    phantom = [i for i in range(counted) if i not in in_meta and i not in in_data]
    orphan_meta = sorted(set(in_meta) - set(in_data))
    orphan_data = sorted(set(in_data) - set(in_meta))
    beyond = sorted(i for i in everywhere if i >= counted)

    report: dict[str, Any] = {
        "counted": counted,
        "frames_counted": int(info.get("total_frames") or 0),
        "present": sorted(everywhere),
        "frames_present": frames_present,
        "phantom": phantom,
        "orphan_meta": orphan_meta,
        "orphan_data": orphan_data,
        "beyond_count": beyond,
        "stale_meta_files": [],
        "short_video": camera_span_problems(root) if deep else [],
        "bad_offsets": not offsets_consistent(in_meta),
        "bad_totals": counted != len(everywhere)
        or int(info.get("total_frames") or 0) != frames_present,
    }
    report["ok"] = not (
        phantom
        or orphan_meta
        or orphan_data
        or beyond
        or report["short_video"]
        or report["bad_offsets"]
        or report["bad_totals"]
    )
    # Repairable means: the only thing wrong is that the dataset counts episodes
    # nobody wrote (and whatever bookkeeping that alone knocked out of step).
    # A camera that recorded fewer frames than the episode claims is NOT
    # repairable here: the frames are gone, and whether to drop the episode or
    # keep the part that was recorded depends on what it shows.
    report["repairable"] = not report["ok"] and not (
        orphan_meta or orphan_data or beyond or report["short_video"]
    )
    report["summary"] = damage_summary(report)
    return report


def damage_summary(report: "dict[str, Any]") -> str:
    """One line an operator can act on. Pure."""
    if report.get("ok"):
        return f"{len(report.get('present') or [])} episode(s), all accounted for"
    parts: list[str] = []
    phantom = report.get("phantom") or []
    if phantom:
        shown = ", ".join(str(i) for i in phantom[:6])
        more = "" if len(phantom) <= 6 else f" (+{len(phantom) - 6} more)"
        parts.append(
            f"episode(s) {shown}{more} are counted by the dataset but were never "
            "written: no recording, no metadata, no side files"
        )
    if report.get("orphan_meta"):
        parts.append(
            f"episode(s) {report['orphan_meta']} have metadata but no recording"
        )
    if report.get("orphan_data"):
        parts.append(
            f"episode(s) {report['orphan_data']} have a recording but no metadata"
        )
    if report.get("beyond_count"):
        parts.append(
            f"episode(s) {report['beyond_count']} exist beyond the counted total"
        )
    short = report.get("short_video") or []
    if short:
        shown = "; ".join(
            f"episode {p['episode']} has {p['rows']} rows but only {p['frames']} "
            f"frames from {p['camera']}"
            for p in short[:3]
        )
        more = "" if len(short) <= 3 else f" (+{len(short) - 3} more)"
        parts.append(
            f"{shown}{more} — that camera stopped delivering mid-episode, and "
            "training dies partway through the first pass on it. Remove the "
            "episode, or train on a camera set that leaves it out"
        )
    if not parts:
        if report.get("bad_totals"):
            parts.append(
                f"the dataset counts {report.get('counted')} episode(s) and "
                f"{report.get('frames_counted')} frame(s), but holds "
                f"{len(report.get('present') or [])} and "
                f"{report.get('frames_present')}"
            )
        if report.get("bad_offsets"):
            parts.append("the episode offsets do not match the episode lengths")
    return "; ".join(parts)


def ensure_loadable(root: Path) -> "dict[str, Any]":
    """Raise ``DatasetDamaged`` unless LeRobot can open this dataset locally.

    Called before anything constructs a ``LeRobotDataset``, so that a local gap
    is reported as a local gap instead of as an unreachable Hugging Face.
    """
    report = dataset_integrity(root)
    if report["ok"]:
        return report
    fix = (
        " Repair it (in the console, or repair_phantom_episodes) and try again."
        if report["repairable"]
        else " This needs a decision about what to keep, so it is not repaired"
        " automatically."
    )
    raise DatasetDamaged(f"{Path(root).name}: {report['summary']}.{fix}")


def _rewrite(table, updates: "dict[str, list]", path: Path) -> None:
    """Replace whole columns of a parquet table, atomically, keeping its types."""
    import pyarrow as pa  # type: ignore[import]
    import pyarrow.parquet as pq  # type: ignore[import]

    for key, values in updates.items():
        position = table.column_names.index(key)
        field = table.schema.field(position)
        table = table.set_column(position, field, pa.array(values, type=field.type))
    temp = path.with_suffix(".parquet.partial")
    pq.write_table(table, temp)
    os.replace(temp, path)


def repair_phantom_episodes(
    root: Path, depth_names: "tuple[str, ...] | list[str]" = ()
) -> "dict[str, Any]":
    """Forget episodes the dataset counts but never wrote. Returns what changed.

    The surviving episodes are renumbered to a contiguous ``0..N-1`` -- LeRobot
    requires that -- which means indices shift: with episode 4 missing, what was
    episode 5 becomes episode 4. File NAMES are deliberately left alone, so a
    file called ``file-005.parquet`` may hold episode 4 afterwards; nothing
    requires the two to agree, because every metadata row names the files it
    belongs to, and ``repair_episode_metadata`` (run last) guarantees it.

    Every write is a ``.partial`` plus an atomic rename, and the tables go
    through pyarrow rather than pandas so the per-episode statistics keep their
    exact types.
    """
    import pyarrow.parquet as pq  # type: ignore[import]

    root = Path(root)
    report = dataset_integrity(root)
    if report["ok"]:
        return {"changed": False, "report": report, "dropped": [], "renumbered": 0}
    if not report["repairable"]:
        raise DatasetDamaged(f"{root.name}: {report['summary']}. Not repaired.")

    present = list(report["present"])
    if not present:
        # Every episode missing is far more likely to be a drive that is not
        # mounted, or a half-finished copy, than a dataset that truly holds
        # nothing -- and "repairing" it would throw away the metadata that says
        # what used to be here.
        raise DatasetDamaged(
            f"{root.name} holds no recordings at all: it counts "
            f"{report['counted']} and none of them is on disk. That is a "
            "missing drive or an unfinished copy, not something to repair. "
            "Delete the dataset if it really is empty."
        )
    mapping = {old: new for new, old in enumerate(present)}
    lengths = episodes_in_metadata(root)

    # New offsets: the cumulative lengths, in the new order.
    offsets: dict[int, tuple[int, int]] = {}
    at = 0
    for old in present:
        length = int(lengths[old]["length"] or 0)
        offsets[old] = (at, at + length)
        at += length
    frames = at

    # 1. the episode rows: their index, their offsets, and the statistics of the
    #    index column, which are the episode number itself.
    for path in episode_meta_files(root):
        table = pq.read_table(path)
        if EPISODE_KEY not in table.column_names:
            continue
        old_indices = [int(v) for v in table.column(EPISODE_KEY).to_pylist()]
        updates: dict[str, list] = {EPISODE_KEY: [mapping[old] for old in old_indices]}
        if FROM_KEY in table.column_names and TO_KEY in table.column_names:
            updates[FROM_KEY] = [offsets[old][0] for old in old_indices]
            updates[TO_KEY] = [offsets[old][1] for old in old_indices]
        for key in table.column_names:
            if not key.startswith(EPISODE_STAT_PREFIX):
                continue
            if key[len(EPISODE_STAT_PREFIX) :] in EPISODE_STAT_KEEP:
                continue
            updates[key] = [[float(mapping[old])] for old in old_indices]
        _rewrite(table, updates, path)

    # 2. the frames themselves.
    for path in data_files(root):
        table = pq.read_table(path)
        if EPISODE_KEY not in table.column_names:
            continue
        old_indices = [int(v) for v in table.column(EPISODE_KEY).to_pylist()]
        _rewrite(table, {EPISODE_KEY: [mapping[old] for old in old_indices]}, path)

    # 3. our own side files, through the mapping LeRobot's own renumbering uses.
    for src_rel, dst_rel in extra_reindex_ops(mapping, list(depth_names)):
        src, dst = root / src_rel, root / dst_rel
        if src == dst or not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)

    # 4. a reviewer's marks follow their episodes.
    marked = read_soft_deleted(root)
    if marked:
        write_soft_deleted(root, [mapping[i] for i in marked if i in mapping])

    # 5. the counts, from what is actually there.
    info_path = root / "meta" / "info.json"
    info = _read_json(info_path)
    if info:
        info["total_episodes"] = len(present)
        info["total_frames"] = frames
        if isinstance(info.get("splits"), dict) and "train" in info["splits"]:
            info["splits"]["train"] = f"0:{len(present)}"
        temp = info_path.with_suffix(".json.partial")
        with open(temp, "w") as f:
            json.dump(info, f, indent=4)
        os.replace(temp, info_path)

    repaired = repair_episode_metadata(root)
    after = dataset_integrity(root)
    return {
        "changed": True,
        "dropped": report["phantom"],
        "renumbered": sum(1 for old, new in mapping.items() if old != new),
        "episodes": len(present),
        "frames": frames,
        "meta_files_fixed": repaired,
        "report": after,
    }
