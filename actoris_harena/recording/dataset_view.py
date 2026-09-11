"""Camera-ablation VIEWS of a collected dataset.

An ablation asks what a policy loses when a camera is taken away. The honest
way to answer it is to train on the same episodes, the same frames and the same
actions, changing only which cameras the policy can see -- so the datasets that
differ must differ in nothing else.

Copying a dataset per camera set would do that and is wasteful: the videos are
99% of the bytes (525 MB of 543 MB for ``cube-pnp-new``) and they are exactly
the part that is shared. A **view** is a directory that LeRobot opens as an
ordinary dataset while owning almost nothing: the metadata is rewritten to name
only the cameras being kept, and each kept camera's video directory is a
symlink back to the source. A three-camera dataset therefore yields three views
for a few megabytes.

Rewriting the metadata rather than overriding the policy's inputs is what makes
this cheap at training time too. ``LeRobotDatasetMetadata.video_keys`` reads the
feature table in ``meta/info.json``, so a camera that is not named there is
never decoded -- and ``dataset_to_policy_features`` then derives exactly the
inputs the policy should see, with no ``--policy.input_features`` override to
keep in step with the data. MEASURED on this laptop, batch 8 with 4 workers:
127 ms/batch for three cameras, 56 ms for one.

Views also normalise the task string, which matters for one policy and not the
others. ``cube-pnp-new`` carries two spellings of one instruction -- "place it
on the plate" and "place it one the plate" -- because the string was retyped
partway through collection, leaving a minority of episodes labelled with the
typo. ACT and Diffusion ignore language entirely, but pi0.5 is conditioned on
it, and that minority would otherwise teach it a second instruction for one
task. The source dataset is never modified: the correction lives in the view,
where it is one rewritten column.

What a view holds, and why:

  * ``videos/<kept camera>/`` -- a symlink to the source. The bytes are shared.
  * ``meta/`` and ``data/`` -- REAL files, rewritten. They are small (under
    4 MB together here), and both carry per-camera and per-task columns that
    have to agree with ``info.json`` or the dataset describes itself wrongly.
  * ``extra/`` -- absent. The sidecar streams are a recording artefact; nothing
    in training reads them.

Building is atomic and idempotent, because the cluster builds views from
several array tasks at once: each build lands in a temporary directory and is
renamed into place, and a task that loses the race keeps the winner's view.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from actoris_harena.recording import camera_profile

CAMERA_PREFIX = "observation.images."
_VIDEO_DTYPES = ("video", "image")

# Camera names are long because they say which arm they ride on; a directory
# name repeating them in full is unreadable. Only the redundant middle word is
# dropped, so `wrist_camera_left` -> `wrist_left` stays unambiguous.
# pi0.5 was pretrained with three fixed camera slots under openpi's names, and a
# finetune reaches them by renaming rather than by re-deriving features: each
# slot carries what it learned about that viewpoint, so a rig camera should land
# on the slot that means the same thing. A slot left unmapped is not an error --
# pi0.5 fills it with a padded image and a zero mask, which is exactly how an
# ablated camera should read to the model.
# The slots themselves, in the order an unnamed camera is given one. The
# overhead slot first because it is the one whose viewpoint is least
# arm-specific, then the two wrists left-to-right.
PI05_SLOT_ORDER = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)

# Short names accepted in a run matrix's `slots` column, so a row can say
# `central=base,left_arm_left_gripper=left_wrist` instead of repeating the
# openpi feature keys.
PI05_SLOT_ALIASES = {
    "base": "observation.images.base_0_rgb",
    "left_wrist": "observation.images.left_wrist_0_rgb",
    "right_wrist": "observation.images.right_wrist_0_rgb",
}

# 224x224 because that is what the policy this exists for takes: FastWAM's
# default image_size is 224x448, which is exactly two square features side by
# side -- the overhead view and this. Each fingertip therefore gets 112x112,
# which is the price of showing the model all four rather than one. Not a rig
# fact: it is the model's input size, so it stays here.
COMPOSITE_SIZE = (224, 224)


def _composites() -> "dict[str, tuple[str, ...]]":
    """This rig's composite image features. See ``recording.camera_profile``."""
    return camera_profile.profile().composites


class ViewError(Exception):
    """A view was asked for that the source dataset cannot provide."""


# --------------------------------------------------------------------------
# pure: what the metadata says, and what it should say
# --------------------------------------------------------------------------
def camera_keys(info: dict) -> list[str]:
    """The dataset's camera feature keys, in the order ``info.json`` lists them."""
    return [
        key
        for key, ft in (info.get("features") or {}).items()
        if ft.get("dtype") in _VIDEO_DTYPES
    ]


def short_name(key: str) -> str:
    """``observation.images.central`` -> ``central``."""
    return key[len(CAMERA_PREFIX) :] if key.startswith(CAMERA_PREFIX) else key


def available_camera_names(info: dict) -> list[str]:
    """The dataset's camera SHORT names, plus every composite it could build.

    A composite is offered only when the dataset records all of its parts, so a
    run matrix naming one is judged against what this dataset can actually
    produce rather than against a list of names that exist somewhere.
    """
    have = [short_name(k) for k in camera_keys(info)]
    known = set(have)
    return have + [
        name for name, parts in _composites().items() if known.issuperset(parts)
    ]


def resolve_cameras(info: dict, names: Iterable[str]) -> list[str]:
    """Camera names (short or full, any order) -> full keys in dataset order.

    ``all`` selects every camera. Resolving through the dataset rather than
    trusting the caller is what turns a typo in a run matrix into a message
    naming the cameras that do exist, instead of a view that silently trains on
    fewer inputs than the experiment intended.
    """
    available = camera_keys(info)
    if not available:
        raise ViewError("dataset has no camera features")

    wanted: list[str] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name == "all":
            wanted.extend(available)
            continue
        key = name if name.startswith(CAMERA_PREFIX) else CAMERA_PREFIX + name
        if key not in available:
            raise ViewError(
                f"no camera {name!r} in this dataset; it has: "
                f"{', '.join(short_name(k) for k in available)}"
            )
        wanted.append(key)

    if not wanted:
        raise ViewError("no cameras selected")
    # Dataset order, duplicates dropped: the same set asked for in two orders is
    # one view, not two.
    return [key for key in available if key in set(wanted)]


def split_selection(info: dict, names: Iterable[str]) -> "tuple[list[str], list[str]]":
    """``central,tactile_quad`` -> the real camera keys, and the composites.

    ``all`` is the dataset's own cameras and never a composite: spending one
    view on four cameras is a deliberate choice about what a policy sees, so it
    has to be asked for by name.
    """
    wanted = [n.strip() for n in names if str(n).strip()]
    composites = [n for n in wanted if n in _composites()]
    for name in composites:
        missing = [
            part
            for part in _composites()[name]
            if CAMERA_PREFIX + part not in camera_keys(info)
        ]
        if missing:
            raise ViewError(
                f"composite {name!r} needs {', '.join(_composites()[name])}, and "
                f"this dataset has no {', '.join(missing)}"
            )
    plain = [n for n in wanted if n not in _composites()]
    return (resolve_cameras(info, plain) if plain else []), composites


def view_slug(info: dict, keep: Sequence[str]) -> str:
    """A short, stable directory name for a camera set (``all``, ``central+wrist_left``)."""
    available = camera_keys(info)
    if list(keep) == available:
        return "all"
    parts = []
    for key in keep:
        name = short_name(key)
        for prefix, replacement in camera_profile.profile().slug_elisions:
            if name.startswith(prefix):
                name = replacement + name[len(prefix) :]
        parts.append(name)
    return "+".join(parts)


def filtered_info(
    info: dict,
    keep: Sequence[str],
    composites: Sequence[str] = (),
    size: "tuple[int, int]" = COMPOSITE_SIZE,
) -> dict:
    """``info.json`` with every camera feature except ``keep`` removed.

    A composite is ADDED as a camera feature of its own, so LeRobot opens it as
    an ordinary camera and the policy is never told it is a tiling.
    """
    drop = set(camera_keys(info)) - set(keep)
    out = dict(info)
    out["features"] = {k: v for k, v in info["features"].items() if k not in drop}
    for name in composites:
        parts = [CAMERA_PREFIX + p for p in _composites()[name]]
        out["features"][CAMERA_PREFIX + name] = composite_feature(info, parts, size)
    return out


def filtered_stats(stats: dict, keep: Sequence[str], cameras: Sequence[str]) -> dict:
    """``stats.json`` with the dropped cameras' entries removed."""
    drop = set(cameras) - set(keep)
    return {k: v for k, v in stats.items() if k not in drop}


def dropped_meta_columns(
    columns: Iterable[str], keep: Sequence[str], cameras: Sequence[str]
) -> list[str]:
    """Episode-metadata columns belonging to cameras that this view drops.

    The per-episode table carries a block of columns per camera (``videos/<key>/…``
    for where its frames live, ``stats/<key>/…`` for its pixel statistics). They
    must go with the feature, or the view claims statistics for a camera it does
    not have.
    """
    drop = set(cameras) - set(keep)
    return [
        col
        for col in columns
        if any(
            col.startswith(f"videos/{k}/") or col.startswith(f"stats/{k}/")
            for k in drop
        )
    ]


def task_remap(
    tasks: Sequence[str],
    counts: dict[int, int] | None = None,
    canonical: str | None = None,
) -> tuple[str, dict[int, int]]:
    """Collapse a dataset's task strings onto one, keeping positions valid.

    Returns the canonical string and a mapping from each old ``task_index`` to
    its new one. The reader resolves a task POSITIONALLY
    (``meta.tasks.iloc[task_index].name``), so the rewritten table and the
    remapped column have to move together -- which is exactly why this returns
    both and the caller applies them in one pass.

    With no ``canonical`` given, the spelling covering the most FRAMES wins.
    Position is no guide: task indices are handed out in first-seen order, and
    on ``cube-pnp-new`` index 0 is the typo that 9 episodes carry while index 1
    is the spelling the other 109 use. Weight of evidence is what identifies the
    instruction the operator meant; order identifies only which was typed first.
    """
    if not tasks:
        raise ViewError("dataset has no tasks")
    if canonical is not None:
        chosen = canonical
    elif counts:
        weights = counts
        chosen = tasks[max(range(len(tasks)), key=lambda i: weights.get(i, 0))]
    else:
        chosen = tasks[0]
    return chosen, {old: 0 for old in range(len(tasks))}


def pi05_rename_map(
    keep: Sequence[str], slots: "dict[str, str] | None" = None
) -> dict[str, str]:
    """This view's cameras -> the pi0.5 slots they should be fed into.

    Only cameras the view actually has are mapped. The slots left over stay
    empty on purpose: that is how a camera is ablated for a policy whose
    architecture has a fixed number of views.

    A camera whose viewpoint matches one of pi0.5's own -- an overhead view, a
    wrist -- takes that slot by NAME, so it inherits what the base learned about
    it. A camera with no counterpart (this rig's fingertip cameras have none:
    pi0.5 was never pretrained on a gel image) takes the next slot still free,
    in ``PI05_SLOT_ORDER``. That is a weaker claim than the named case and is
    worth saying out loud, but it is the right one: the alternative was refusing
    to train pi0.5 on a tactile dataset at all.

    ``slots`` overrides both, and is how a run matrix pins the assignment for an
    ablation. Its values may be the openpi feature keys or the short aliases in
    ``PI05_SLOT_ALIASES``.
    """
    out: "dict[str, str]" = {}
    if len(keep) > len(PI05_SLOT_ORDER):
        raise ViewError(
            f"pi0.5 has {len(PI05_SLOT_ORDER)} image slots but this view has "
            f"{len(keep)} cameras ({', '.join(short_name(k) for k in keep)}); "
            "build a view with fewer, or map them in the slots column"
        )

    if slots:
        for camera, slot in slots.items():
            key = camera if camera.startswith(CAMERA_PREFIX) else CAMERA_PREFIX + camera
            if key not in list(keep):
                raise ViewError(
                    f"slots names camera {short_name(key)!r}, which this view "
                    f"does not have: {', '.join(short_name(k) for k in keep)}"
                )
            full = PI05_SLOT_ALIASES.get(slot, slot)
            if full not in PI05_SLOT_ORDER:
                raise ViewError(
                    f"{slot!r} is not a pi0.5 slot; want one of "
                    f"{', '.join(sorted(PI05_SLOT_ALIASES))}"
                )
            out[key] = full
        return out

    taken: "set[str]" = set()
    unnamed: "list[str]" = []
    for key in keep:
        named = camera_profile.profile().pi05_slots.get(short_name(key))
        if named is None:
            unnamed.append(key)
            continue
        out[key] = named
        taken.add(named)
    free = [s for s in PI05_SLOT_ORDER if s not in taken]
    for key, slot in zip(unnamed, free):
        out[key] = slot
    # zip stops at the shorter list, so a camera left without a slot means the
    # named ones already filled them -- which the length check above cannot
    # catch, because it counts cameras rather than collisions.
    if len(out) < len(keep):
        raise ViewError(
            "pi0.5's slots are already taken by the named cameras; map them "
            f"explicitly in the slots column ({', '.join(sorted(PI05_SLOT_ALIASES))})"
        )
    return out


# --------------------------------------------------------------------------
# building a view on disk
# --------------------------------------------------------------------------
def _rewrite_parquet_dir(
    src_dir: Path,
    dst_dir: Path,
    drop_columns: Sequence[str] = (),
    task_index_map: dict[int, int] | None = None,
    task_strings: dict[int, str] | None = None,
    copy_column_prefixes: "dict[str, str] | None" = None,
) -> None:
    """Copy a tree of parquet files, dropping columns and remapping task indices.

    Types are preserved by editing the arrow table in place rather than taking a
    pandas round trip, which would silently widen ints and turn the fixed-size
    lists the metadata uses into something LeRobot reads back differently.

    ``copy_column_prefixes`` duplicates a block of per-camera columns under a
    new camera's name, which is how a composite gets its own. It runs BEFORE the
    drop, so a composite can be built out of cameras the view itself does not
    keep.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    for src in sorted(src_dir.rglob("*.parquet")):
        table = pq.read_table(src)
        for old, new in (copy_column_prefixes or {}).items():
            for name in list(table.column_names):
                if name.startswith(old):
                    table = table.append_column(
                        new + name[len(old) :], table.column(name)
                    )
        drop = [c for c in drop_columns if c in table.column_names]
        if drop:
            table = table.drop(drop)
        if task_index_map and "task_index" in table.column_names:
            col = table.column("task_index")
            mapped = pa.array(
                [
                    None if v is None else task_index_map.get(v, v)
                    for v in col.to_pylist()
                ],
                type=col.type,
            )
            table = table.set_column(
                table.column_names.index("task_index"), "task_index", mapped
            )
        if task_strings is not None and "tasks" in table.column_names:
            col = table.column("tasks")
            canonical = task_strings[0]
            mapped = pa.array(
                [
                    None if v is None else [canonical for _ in v]
                    for v in col.to_pylist()
                ],
                type=col.type,
            )
            table = table.set_column(table.column_names.index("tasks"), "tasks", mapped)
        out = dst_dir / src.relative_to(src_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)


# --------------------------------------------------------------------------
# composites: several cameras tiled into one image feature
# --------------------------------------------------------------------------
#
# A policy whose architecture fixes how many views it takes cannot be given
# more of them. FastWAM concatenates its image features into a single frame of
# `policy.image_size`, so a five-camera rig has to choose three cameras to throw
# away -- unless four of them are tiled into one feature first, which is what
# this does. The four fingertip cameras become one square frame, and the policy
# sees every tactile signal for the price of one view.
#
# The tiles are SQUASHED to fill their quadrant rather than letterboxed. That is
# what LeRobot's own resize does to these frames anyway, and a letterbox would
# spend a fifth of the pixels on black.


def composite_grid(count: int) -> "tuple[int, int]":
    """Rows and columns for ``count`` tiles. Raises for a count that cannot tile."""
    if count == 4:
        return 2, 2
    if count == 2:
        return 1, 2
    raise ViewError(f"no tile layout for {count} cameras; a composite takes 2 or 4")


def composite_feature(
    info: dict, parts: Sequence[str], size: "tuple[int, int]"
) -> dict:
    """The ``info.json`` feature entry for a composite of ``parts``.

    Everything about the encoding is taken from the first part, so the composite
    is written the way the recorder writes a camera; only the frame size, which
    is the point of the composite, differs.
    """
    height, width = size
    first = info["features"][parts[0]]
    video = dict(first.get("info") or {})
    video.update({"video.height": height, "video.width": width})
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": video,
    }


def composite_stats(stats: dict, parts: Sequence[str]) -> "dict | None":
    """Pixel statistics for the tiled frame, derived from its parts'.

    Equal-area tiles, so the mean is the mean of the means and the second moment
    is the mean of the second moments; min and max carry straight over. Derived
    rather than measured because measuring means decoding every frame a second
    time for a number that only normalises an image -- and the resize, which is
    the one thing this ignores, moves it far less than the tiling does.
    """
    import math

    maybe = [stats.get(p) for p in parts]
    if any(s is None for s in maybe):
        return None
    have: "list[dict]" = [s for s in maybe if s is not None]
    try:
        n = len(have)
        # One row per part, one column per channel.
        means = [_flat(s["mean"]) for s in have]
        stds = [_flat(s["std"]) for s in have]
        mins = [_flat(s["min"]) for s in have]
        maxs = [_flat(s["max"]) for s in have]

        mean = [sum(col) / n for col in zip(*means)]
        # E[x^2] over equal-area tiles is the mean of each tile's E[x^2].
        second = [
            sum(m * m + sd * sd for m, sd in zip(mcol, scol)) / n
            for mcol, scol in zip(zip(*means), zip(*stds))
        ]
        std = [math.sqrt(max(e - m * m, 0.0)) for e, m in zip(second, mean)]
        return {
            "mean": _nest(mean),
            "std": _nest(std),
            "min": _nest([min(col) for col in zip(*mins)]),
            "max": _nest([max(col) for col in zip(*maxs)]),
            "count": [int(sum(float(_flat(s.get("count", [1]))[0]) for s in have) / n)],
        }
    except (KeyError, TypeError, ValueError, IndexError):
        # Statistics are a convenience for normalisation, not a correctness
        # requirement: a dataset whose stats are shaped differently should
        # produce a composite without them rather than no composite.
        return None


def _nest(channels: "Sequence[float]") -> list:
    """``[a, b, c]`` -> ``[[[a]], [[b]], [[c]]]``, the shape LeRobot writes."""
    return [[[float(v)]] for v in channels]


def _flat(value) -> list:
    """``[[[a]],[[b]],[[c]]]`` -> ``[a, b, c]``; a flat list is left alone."""
    out = []
    stack = [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = list(item) + stack
        else:
            out.append(float(item))
    return out


def build_composite_video(
    sources: "Sequence[Path]", dst: Path, size: "tuple[int, int]"
) -> int:
    """Tile ``sources`` frame by frame into one video at ``dst``. Returns frames.

    Decoded and encoded with PyAV rather than the ffmpeg command line, because
    this runs where the training runs: a CREATE compute node has no ffmpeg and
    no way to install one, while PyAV's wheel carries its own.

    The sources are read in LOCKSTEP and must have the same number of frames.
    They do by construction -- the recorder writes one frame per camera per
    dataset frame -- and if they ever do not, tiling them would silently pair
    frame k of one camera with frame k+1 of another, which is a dataset that
    looks fine and teaches the wrong thing. So it is checked.
    """
    import av
    import numpy as np

    rows, cols = composite_grid(len(sources))
    height, width = size
    tile_h, tile_w = height // rows, width // cols

    # Annotated because PyAV's stubs overload av.open on its mode argument and
    # do not narrow to a container when the mode is not a literal at the call.
    containers: "list[Any]" = [av.open(str(p)) for p in sources]
    out: "Any" = av.open(str(dst), mode="w")
    try:
        rate = containers[0].streams.video[0].average_rate
        stream = out.add_stream("libsvtav1", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "30", "preset": "8"}

        streams = [
            c.decode(video=0) for c in containers
        ]  # generators, advanced together
        written = 0
        while True:
            tiles = []
            ended = 0
            for gen in streams:
                try:
                    tiles.append(next(gen))
                except StopIteration:
                    ended += 1
            if ended == len(streams):
                break
            if ended:
                raise ViewError(
                    f"the cameras of this composite do not have the same number "
                    f"of frames ({written} in the shortest); they cannot be "
                    "tiled without pairing one camera's frame with another's "
                    "neighbour"
                )
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for i, frame in enumerate(tiles):
                row, col = divmod(i, cols)
                canvas[
                    row * tile_h : (row + 1) * tile_h,
                    col * tile_w : (col + 1) * tile_w,
                ] = frame.reformat(
                    width=tile_w, height=tile_h, format="rgb24"
                ).to_ndarray()
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, "rgb24")):
                out.mux(packet)
            written += 1
        for packet in stream.encode():
            out.mux(packet)
        return written
    finally:
        out.close()
        for container in containers:
            container.close()


def _write_view(
    src: Path,
    work: Path,
    keep: Sequence[str],
    canonical_task: str | None,
    composites: Sequence[str] = (),
    composite_size: "tuple[int, int]" = COMPOSITE_SIZE,
) -> None:
    import pandas as pd

    info = json.loads((src / "meta" / "info.json").read_text())
    cameras = camera_keys(info)

    (work / "meta").mkdir(parents=True)
    info_out = filtered_info(info, keep, composites, composite_size)

    stats_path = src / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        out_stats = filtered_stats(stats, keep, cameras)
        for name in composites:
            parts = [CAMERA_PREFIX + p for p in _composites()[name]]
            derived = composite_stats(stats, parts)
            if derived is not None:
                out_stats[CAMERA_PREFIX + name] = derived
        (work / "meta" / "stats.json").write_text(json.dumps(out_stats, indent=4))

    # Tasks: one canonical string, and the index map that keeps the data honest.
    tasks_df = pd.read_parquet(src / "meta" / "tasks.parquet")
    task_list = list(tasks_df.index)
    index_map: dict[int, int] | None = None
    task_strings: dict[int, str] | None = None
    if canonical_task is not None or len(task_list) > 1:
        chosen, index_map = task_remap(
            task_list, _task_frame_counts(src / "data"), canonical_task
        )
        task_strings = {0: chosen}
        new_tasks = pd.DataFrame(
            {"task_index": [0]}, index=pd.Index([chosen], name="task")
        )
        new_tasks.to_parquet(work / "meta" / "tasks.parquet")
        info_out["total_tasks"] = 1
    else:
        shutil.copy2(src / "meta" / "tasks.parquet", work / "meta" / "tasks.parquet")

    (work / "meta" / "info.json").write_text(json.dumps(info_out, indent=4))

    # A composite's per-episode columns are its FIRST PART's. For `videos/…`
    # that is exact: those columns say which file an episode's frames are in and
    # between which timestamps, and the tiled video is written frame for frame
    # against that same timeline in the same layout. For `stats/…` it is an
    # approximation -- one tile's pixel statistics standing for four. The
    # aggregate in stats.json, which is what LeRobot normalises with, is derived
    # from all four properly; these per-episode ones exist so the metadata is
    # shaped like every other camera's.
    renames = {
        f"{prefix}/{CAMERA_PREFIX}{_composites()[name][0]}/": (
            f"{prefix}/{CAMERA_PREFIX}{name}/"
        )
        for name in composites
        for prefix in ("videos", "stats")
    }
    _rewrite_parquet_dir(
        src / "meta" / "episodes",
        work / "meta" / "episodes",
        drop_columns=dropped_meta_columns(
            _all_columns(src / "meta" / "episodes"), keep, cameras
        ),
        task_index_map=index_map,
        task_strings=task_strings,
        copy_column_prefixes=renames,
    )
    _rewrite_parquet_dir(src / "data", work / "data", task_index_map=index_map)

    # The bytes that matter are shared, not copied.
    (work / "videos").mkdir()
    for key in keep:
        (work / "videos" / key).symlink_to((src / "videos" / key).resolve())
    for name in composites:
        _build_composite(src, work, name, composite_size)


def _build_composite(src: Path, work: Path, name: str, size: "tuple[int, int]") -> None:
    """Tile one composite's parts into a new video tree under ``work``.

    This is the one thing a view cannot share by symlink: the frames do not
    exist until they are made. It mirrors the source's chunk/file layout so the
    per-episode timestamps copied above still name the right file.
    """
    parts = [CAMERA_PREFIX + p for p in _composites()[name]]
    first = src / "videos" / parts[0]
    for rel in sorted(
        p.relative_to(first) for p in first.rglob("*.mp4") if p.is_file()
    ):
        sources = [src / "videos" / part / rel for part in parts]
        missing = [s for s in sources if not s.is_file()]
        if missing:
            raise ViewError(
                f"composite {name!r} is missing {missing[0]}; its cameras must "
                "share the source's chunk layout"
            )
        dst = work / "videos" / (CAMERA_PREFIX + name) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        build_composite_video(sources, dst, size)


def _task_frame_counts(data_dir: Path) -> dict[int, int]:
    """How many frames each ``task_index`` covers -- only that column is read."""
    import pyarrow.parquet as pq

    counts: dict[int, int] = {}
    for path in sorted(data_dir.rglob("*.parquet")):
        for value in (
            pq.read_table(path, columns=["task_index"]).column("task_index").to_pylist()
        ):
            if value is not None:
                counts[value] = counts.get(value, 0) + 1
    return counts


def _all_columns(parquet_dir: Path) -> list[str]:
    import pyarrow.parquet as pq

    for path in sorted(parquet_dir.rglob("*.parquet")):
        return list(pq.read_schema(path).names)
    return []


def build_view(
    src: Path | str,
    dst: Path | str,
    cameras: Iterable[str],
    canonical_task: str | None = None,
    force: bool = False,
    composites: Sequence[str] = (),
    composite_size: "tuple[int, int]" = COMPOSITE_SIZE,
) -> Path:
    """Create (or reuse) a camera view of ``src`` at ``dst``.

    Idempotent: an existing view is kept unless ``force``. The build is atomic,
    so a reader never sees a half-written dataset and concurrent builders cannot
    corrupt each other -- the loser of a race discards its own work.
    """
    src, dst = Path(src), Path(dst)
    if not (src / "meta" / "info.json").is_file():
        raise ViewError(f"no dataset at {src}")

    if dst.exists() and not force:
        return dst
    if dst.exists() and force:
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text())
    # The selection may name composites as well as cameras; splitting it here
    # means a caller passes what the run matrix said and nothing else.
    keep, named = split_selection(info, cameras)
    composites = list(composites) + [c for c in named if c not in composites]
    if not keep and not composites:
        raise ViewError("no cameras selected")

    dst.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
    try:
        _write_view(
            src, work / "view", keep, canonical_task, composites, composite_size
        )
        try:
            os.replace(work / "view", dst)
        except OSError:
            # Another task finished this same view first. Its copy is as good as
            # ours by construction, so keep it.
            if not dst.exists():
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return dst
