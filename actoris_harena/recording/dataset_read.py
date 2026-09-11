"""Fast, dependency-light reads of one episode from a collected dataset.

Reviewing a session asks a narrow question of the dataset -- give me episode N's
camera frames and its joint columns -- and the general dataset API answers it far
too slowly for a click. ``LeRobotDataset.__getitem__`` addresses frames
individually, so each frame costs a seek and a decode into an AV1 stream:
measured at 57 ms per frame, which is 24 seconds of decoding for a
fourteen-second episode before anything is drawn. Read sequentially instead and
the same frames cost 1.9 ms each, because that is what a video codec is built
for: one linear pass over the range the episode occupies.

This module therefore reads an episode the way its files are laid out. An
episode's metadata row names the data file holding its frames and, per camera,
the video file plus the timestamp window it occupies inside it; the joint columns
come straight out of the data parquet through pyarrow; the frames come from one
seek-and-decode pass per camera. Nothing here loads the dataset as a whole, so
reading is also unaffected by an unrelated episode's file being damaged or still
open for writing.

The layout helpers (``episode_video_path``, ``episode_window``) are pure and
unit-tested; the readers touch the filesystem. Shared by the review tool's
renderer and its per-episode data route so the layout knowledge lives once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_IMAGE_PREFIX = "observation.images."


def episode_video_path(root: Path, row: "dict[str, Any]", video_key: str) -> Path:
    """Path of the video file that contains ``video_key``'s frames for this row.

    Several episodes share one video file, which is why the row carries a chunk
    and file index per camera rather than a name per episode. Pure.
    """
    chunk = int(row[f"videos/{video_key}/chunk_index"])
    index = int(row[f"videos/{video_key}/file_index"])
    return (
        Path(root)
        / "videos"
        / video_key
        / f"chunk-{chunk:03d}"
        / f"file-{index:03d}.mp4"
    )


def episode_window(row: "dict[str, Any]", video_key: str) -> "tuple[float, float]":
    """``(from, to)`` seconds this episode occupies inside its video file. Pure.

    The end is EXCLUSIVE: an episode's ``to_timestamp`` is exactly the next
    episode's ``from_timestamp``, so the frame sitting at ``to`` belongs to the
    next recording, not this one. Reported as the file records it; anything
    displaying frames wants ``playable_window`` instead.
    """
    return (
        float(row[f"videos/{video_key}/from_timestamp"]),
        float(row[f"videos/{video_key}/to_timestamp"]),
    )


def playable_window(
    row: "dict[str, Any]", video_key: str, fps: float
) -> "tuple[float, float]":
    """``(first_frame, last_frame)`` times for this episode's own frames. Pure.

    Several episodes share one video file, and the recorded window's end is
    exclusive, so an episode's frames sit at ``from``, ``from + 1/fps``, ... ,
    ``to - 1/fps`` and the frame AT ``to`` is the next recording's first. Playing
    or seeking to the recorded end therefore shows a frame from the following
    episode -- which is exactly what a viewer that treated ``to`` as inclusive
    did. This returns the inclusive pair, so a player can stop on the last frame
    the episode actually owns.

    A single-frame episode collapses to a zero-length window rather than an
    inverted one.
    """
    start, end = episode_window(row, video_key)
    period = 1.0 / float(fps) if fps else 0.0
    return start, max(end - period, start)


def episode_data_path(root: Path, row: "dict[str, Any]") -> Path:
    """Path of the parquet file holding this episode's frame rows. Pure."""
    chunk = int(row["data/chunk_index"])
    index = int(row["data/file_index"])
    return Path(root) / "data" / f"chunk-{chunk:03d}" / f"file-{index:03d}.parquet"


def video_keys(root: Path) -> "list[str]":
    """The dataset's camera stream keys, in recorded order.

    Read from the dataset's own feature declaration rather than through the
    dataset API, so listing the cameras never depends on the episode metadata
    being complete. Depth maps are excluded: they are stored as images, not as a
    playable stream.
    """
    import json

    try:
        with open(Path(root) / "meta" / "info.json", "r") as f:
            features = json.load(f).get("features", {})
    except (OSError, ValueError, TypeError, AttributeError):
        return []
    keys = []
    for key, feature in features.items():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        if (feature.get("info") or {}).get("is_depth_map"):
            continue
        keys.append(key)
    return keys


def camera_label(video_key: str) -> str:
    """The stream's short name, without the feature-namespace prefix. Pure."""
    return (
        video_key[len(_IMAGE_PREFIX) :]
        if video_key.startswith(_IMAGE_PREFIX)
        else video_key
    )


def read_episode_row(root: Path, episode: int) -> "dict[str, Any] | None":
    """Episode ``episode``'s metadata row, or ``None`` if it cannot be read.

    Scans the episode metadata files and returns the matching row. Each file is
    read on its own so a damaged or still-open file costs only the episodes it
    holds -- the same fault isolation ``read_episode_lengths`` relies on, and the
    reason this does not go through the dataset API, which concatenates every
    file and fails as a whole.
    """
    meta_dir = Path(root) / "meta" / "episodes"
    if not meta_dir.is_dir():
        return None
    import pyarrow.parquet as pq

    for path in sorted(meta_dir.rglob("*.parquet")):
        try:
            table = pq.read_table(path)
            if "episode_index" not in table.column_names:
                continue
            indices = table.column("episode_index").to_pylist()
            if episode not in indices:
                continue
            # Stats columns are large and never wanted here.
            wanted = [c for c in table.column_names if not c.startswith("stats/")]
            return table.select(wanted).to_pylist()[indices.index(episode)]
        except Exception:
            continue
    return None


def read_episode_joints(root: Path, row: "dict[str, Any]") -> "tuple[Any, Any]":
    """``(state, action)`` arrays for the episode, straight from its data file.

    Only the two joint columns are read, and only the rows belonging to this
    episode, so the cost is independent of how many episodes share the file.
    """
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(
        episode_data_path(root, row),
        columns=["episode_index", "observation.state", "action"],
    )
    episode = int(row["episode_index"])
    table = table.filter(pc.equal(table.column("episode_index"), episode))
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=float)
    action = np.asarray(table.column("action").to_pylist(), dtype=float)
    return state, action


def decode_episode_frames(
    root: Path,
    row: "dict[str, Any]",
    video_key: str,
    fps: float,
    limit: "int | None" = None,
) -> "list[Any]":
    """This episode's BGR frames for one camera, by a single sequential pass.

    Seeks once to the start of the episode's window and decodes forward to its
    last frame. Frames before the window are skipped rather than sought past,
    because a seek lands on the preceding keyframe by definition and the frames
    between are decoded anyway.

    Stopping is bounded two ways, because the window's end is exclusive and
    landing one frame past it yields the NEXT episode's opening frame: by the
    frame count the episode declares, which is authoritative, and by a time half
    a frame period inside the exclusive end, so float noise cannot admit that
    frame either. ``limit`` stops earlier still, which the caller uses to keep
    the cameras aligned when one stream delivered a frame more than another.
    """
    import av  # type: ignore[import]

    start, last = playable_window(row, video_key, fps)
    cutoff = last + (0.5 / float(fps) if fps else 0.0)
    wanted = int(row.get("length") or 0) or None
    if limit is not None:
        wanted = limit if wanted is None else min(wanted, limit)
    frames: list[Any] = []
    container = av.open(str(episode_video_path(root, row, video_key)))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(start / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time < start - 0.001:
                continue
            if frame.time > cutoff:
                break
            frames.append(frame.to_ndarray(format="bgr24"))
            if wanted is not None and len(frames) >= wanted:
                break
    finally:
        container.close()
    return frames
