"""Whole-dataset operations for the console: create, rename, delete, merge.

The episode-level curation in ``actoris_harena.recording.dataset_edit`` works INSIDE one
dataset; this module works on the collection directory itself, which until now
was done by hand with ``mv`` and ``rm -rf`` (and is how a drive ends up holding
both ``fold_short`` and ``short_fold``).

Nothing here imports aiohttp: the rules are the interesting part and they are
unit-tested directly. The route layer in ``lifecycle_api`` turns the return
values into responses.

Two conventions are shared with ``dataset_edit`` because they are what makes
these operations safe on the collection drive:

* **A dataset's name is its directory name.** ``meta/info.json`` records no
  repo id, so renaming is a directory rename and nothing inside needs rewriting.
* **Trash, then swap.** A destructive step first moves the old tree aside to a
  sibling ``<name>.trash-<stamp>`` (atomic within the filesystem) and only then
  removes it, so a failure partway never leaves a half-deleted dataset.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from actoris_harena.recording.dataset_check import ensure_loadable
from actoris_harena.recording.dataset_edit import (
    ReadOnlyDatasetError,
    episode_uid_rel,
    read_soft_deleted,
    repair_episode_metadata,
    writability_problem,
)

# Names the console will accept for a dataset. Deliberately narrower than the
# filesystem allows: a dataset name is also a LeRobot repo id and appears in
# shell commands, so spaces, slashes and leading dots are all refused rather
# than quoted around forever.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Suffixes this module and dataset_edit use for their own working directories.
# A dataset must never be given one, or the next sweep would treat it as debris.
_RESERVED_SUFFIX_RE = re.compile(r"\.(trash|tmp)-\d{8}-\d{6}$")

_MAX_NAME = 64


def valid_dataset_name(name: str) -> "str | None":
    """Why ``name`` is not usable as a dataset name, or ``None`` if it is. Pure."""
    if not isinstance(name, str) or not name:
        return "give the dataset a name"
    if len(name) > _MAX_NAME:
        return f"name is longer than {_MAX_NAME} characters"
    if not _NAME_RE.match(name):
        return (
            "use letters, digits, dot, dash or underscore only, starting with a "
            "letter or digit (no spaces, slashes or leading dots)"
        )
    if _RESERVED_SUFFIX_RE.search(name):
        return "that suffix is reserved for the console's own working directories"
    return None


def is_working_dir(name: str) -> bool:
    """Whether ``name`` is one of the console's leftovers, not a dataset. Pure."""
    return bool(_RESERVED_SUFFIX_RE.search(name))


# ── Reading a dataset's shape (cheap: the metadata files, not the dataset) ────


def feature_signature(features: "dict[str, Any]") -> "dict[str, Any]":
    """``{key: (dtype, shape)}`` -- what two datasets must share to merge. Pure.

    A reduction of LeRobot's ``features_equal_for_merge`` to the parts an
    operator can act on: which streams a dataset has, and how wide each one is.
    LeRobot's own stricter check still runs inside the merge; this one exists so
    the console can say "these two do not have the same cameras" before starting
    an hour of re-encoding.
    """
    out: dict[str, Any] = {}
    for key, feature in sorted((features or {}).items()):
        if not isinstance(feature, dict):
            continue
        shape = feature.get("shape")
        out[key] = (feature.get("dtype"), tuple(shape) if shape else ())
    return out


def read_dataset_meta(path: Path) -> "dict[str, Any]":
    """The console's view of one dataset on disk. Never raises.

    Reads ``meta/info.json`` and ``meta/tasks.parquet`` directly instead of
    loading the dataset, so a partially written or damaged dataset still lists
    (with whatever it does have) rather than breaking the whole page.
    """
    path = Path(path)
    info: dict[str, Any] = {}
    try:
        with open(path / "meta" / "info.json", "r") as f:
            info = json.load(f)
    except (OSError, ValueError, TypeError):
        info = {}
    features = info.get("features") or {}
    streams = [
        k
        for k, v in features.items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    ]
    return {
        "name": path.name,
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "streams": sorted(streams),
        "features": feature_signature(features),
        "tasks": read_tasks(path),
        "pending": len(read_soft_deleted(path)),
    }


def read_tasks(path: Path) -> "list[str]":
    """The language instructions stored with the dataset's frames. Never raises."""
    try:
        import pandas as pd

        tasks = pd.read_parquet(Path(path) / "meta" / "tasks.parquet")
        return [str(t) for t in tasks.index.tolist()]
    except Exception:
        return []


def directory_size(path: Path) -> int:
    """Bytes used by ``path``, following no symlinks. Never raises."""
    total = 0
    for parent, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(parent, name)).st_size
            except OSError:
                continue
    return total


# ── Rename and delete ────────────────────────────────────────────────────────


def rename_dataset(root: Path, old: str, new: str) -> Path:
    """Rename a dataset directory. Returns the new path.

    Nothing inside the dataset records its name, so this is exactly a directory
    rename -- and, being a rename within one directory, it is atomic and cannot
    half-happen.
    """
    root = Path(root)
    problem = valid_dataset_name(new)
    if problem:
        raise ValueError(problem)
    src = root / old
    dst = root / new
    if valid_dataset_name(old) or not src.is_dir():
        raise FileNotFoundError(f"no dataset {old!r} in {root}")
    if dst.exists():
        raise FileExistsError(f"{new!r} already exists")
    problem = writability_problem(root)
    if problem:
        raise ReadOnlyDatasetError(f"cannot rename: {problem}")
    os.replace(src, dst)
    return dst


def delete_dataset(root: Path, name: str) -> Path:
    """Move a dataset to a sibling trash directory and return that path.

    The rename is what the operator waits for; removing the bytes afterwards is
    the caller's job (on a worker thread), because deleting a hundred gigabytes
    of video takes far longer than a click should.
    """
    root = Path(root)
    src = root / name
    if valid_dataset_name(name) or not src.is_dir():
        raise FileNotFoundError(f"no dataset {name!r} in {root}")
    problem = writability_problem(root)
    if problem:
        raise ReadOnlyDatasetError(f"cannot delete: {problem}")
    trash = root / f"{name}.trash-{time.strftime('%Y%m%d-%H%M%S')}"
    os.replace(src, trash)
    return trash


def purge(path: Path) -> None:
    """Remove a trashed dataset. Never raises -- the dataset is already gone."""
    shutil.rmtree(path, ignore_errors=True)


# ── Merge ────────────────────────────────────────────────────────────────────


def merge_compatibility(
    metas: "list[dict[str, Any]]",
    out_name: str,
    free_bytes: int,
    taken_names: "tuple[str, ...]" = (),
) -> "list[str]":
    """Every reason this merge is refused, in the operator's terms. Pure.

    An empty list means it may go ahead. All reasons are collected rather than
    raising on the first, so the form can show what to fix in one pass.
    """
    reasons: list[str] = []

    problem = valid_dataset_name(out_name)
    if problem:
        reasons.append(problem)
    elif out_name in taken_names:
        reasons.append(f"{out_name!r} already exists — choose another name")
    if any(m["name"] == out_name for m in metas):
        reasons.append("the merged dataset cannot have the same name as a source")

    if len(metas) < 2:
        reasons.append("choose at least two datasets to merge")
        return reasons

    first = metas[0]
    for meta in metas[1:]:
        if meta.get("fps") != first.get("fps"):
            reasons.append(
                f"{meta['name']} records at {meta.get('fps')} fps, "
                f"{first['name']} at {first.get('fps')} fps"
            )
        if meta.get("robot_type") != first.get("robot_type"):
            reasons.append(
                f"{meta['name']} is a {meta.get('robot_type')} dataset, "
                f"{first['name']} a {first.get('robot_type')} one"
            )
        differing = _feature_differences(first, meta)
        if differing:
            reasons.append(
                f"{meta['name']} and {first['name']} do not record the same "
                f"streams ({', '.join(differing)})"
            )

    for meta in metas:
        if meta.get("pending"):
            reasons.append(
                f"{meta['name']} has {meta['pending']} episode(s) marked for "
                "deletion — remove them for good, or restore them, before merging"
            )

    needed = sum(int(m.get("size_bytes") or 0) for m in metas)
    if free_bytes and needed > free_bytes:
        reasons.append(
            f"the merged dataset needs about {needed / 1e9:.1f} GB but only "
            f"{free_bytes / 1e9:.1f} GB is free"
        )
    return reasons


def _feature_differences(a: "dict[str, Any]", b: "dict[str, Any]") -> "list[str]":
    """Feature keys that stop ``a`` and ``b`` merging, named for a human. Pure."""
    fa = a.get("features") or {}
    fb = b.get("features") or {}
    out = [f"only in {a['name']}: {k}" for k in sorted(set(fa) - set(fb))]
    out += [f"only in {b['name']}: {k}" for k in sorted(set(fb) - set(fa))]
    out += [
        f"{k} differs in shape"
        for k in sorted(set(fa) & set(fb))
        if tuple(fa[k]) != tuple(fb[k])
    ]
    return out


def merge_extra_ops(
    source_counts: "list[int]", depth_names: "list[str] | None" = None
) -> "list[tuple[int, str, str]]":
    """``(source, src_rel, dst_rel)`` moves for the ``extra/`` side files. Pure.

    LeRobot's aggregation concatenates the sources in order and renumbers the
    episodes, so the k-th source's episodes start at the sum of the counts
    before it. This mirrors ``extra_reindex_ops`` for that offset: the sidecar
    and drift parquets, the identity file, and any per-episode depth directory.
    """
    ops: list[tuple[int, str, str]] = []
    offset = 0
    for src, count in enumerate(source_counts):
        for old in range(count):
            new = old + offset
            ops.append(
                (
                    src,
                    f"extra/drift_{old:06d}.parquet",
                    f"extra/drift_{new:06d}.parquet",
                )
            )
            ops.append(
                (
                    src,
                    f"extra/episode_{old:06d}.parquet",
                    f"extra/episode_{new:06d}.parquet",
                )
            )
            ops.append((src, episode_uid_rel(old), episode_uid_rel(new)))
            for dname in depth_names or []:
                ops.append(
                    (
                        src,
                        f"extra/depth/{dname}/episode_{old:06d}",
                        f"extra/depth/{dname}/episode_{new:06d}",
                    )
                )
        offset += count
    return ops


def _meta_json_choice(roots: "list[Path]", rel: str) -> "Path | None":
    """The one source copy of a dataset-level meta JSON, or ``None``.

    LeRobot rewrites ``meta/`` from scratch, so ``action_space.json`` and
    ``realsense.json`` have to be carried across by hand -- and only make sense
    on the merged dataset if every source that has one agrees.
    """
    found = [r / "meta" / rel for r in roots if (r / "meta" / rel).is_file()]
    if not found:
        return None
    first = json.loads(found[0].read_text())
    for other in found[1:]:
        if json.loads(other.read_text()) != first:
            raise ValueError(
                f"the sources disagree on meta/{rel}; merge only datasets "
                "collected with the same action space and camera setup"
            )
    return found[0]


def merge_datasets(
    root: Path,
    names: "list[str]",
    out_name: str,
    progress: "Callable[[str], None] | None" = None,
) -> Path:
    """Write ``names`` into one new dataset ``out_name``. The sources are kept.

    Builds into a sibling ``<out>.tmp-<stamp>`` and swaps it in at the end, so
    an interrupted merge leaves the collection directory as it was (apart from
    the temp directory, which names itself).
    """
    root = Path(root)
    say = progress or (lambda _m: None)
    roots = [root / n for n in names]
    for path in roots:
        if not path.is_dir():
            raise FileNotFoundError(f"no dataset {path.name!r} in {root}")
    dst = root / out_name
    if dst.exists():
        raise FileExistsError(f"{out_name!r} already exists")
    problem = writability_problem(root)
    if problem:
        raise ReadOnlyDatasetError(f"cannot merge: {problem}")

    from lerobot.datasets.aggregate import aggregate_datasets

    stamp = time.strftime("%Y%m%d-%H%M%S")
    temp = root / f"{out_name}.tmp-{stamp}"
    # A source that is itself the result of an earlier merge may name episode
    # metadata files that were never written; the aggregation opens exactly
    # those files, so it would fail here rather than at the end.
    for source in roots:
        for fixed in repair_episode_metadata(source):
            say(f"repaired {source.name}/{fixed}")
        # A source counting an episode nobody wrote takes the aggregation to the
        # Hub for a version tag, which offline blames on the network. Refuse
        # here, naming the source and the real fault.
        ensure_loadable(source)
    say(f"merging {', '.join(names)} → {out_name}")
    try:
        aggregate_datasets(list(names), out_name, roots=roots, aggr_root=temp)
        # The aggregation writes every source's rows into the destination's own
        # files but keeps the source's file indices, so the merged dataset would
        # be the next one that cannot be curated. Correct it before the swap.
        repair_episode_metadata(temp)

        for rel in ("action_space.json", "realsense.json"):
            src = _meta_json_choice(roots, rel)
            if src is not None:
                (temp / "meta").mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, temp / "meta" / rel)

        say("copying the per-episode side files")
        counts = [int(read_dataset_meta(r)["total_episodes"] or 0) for r in roots]
        depth_names = sorted(
            {
                d.name
                for r in roots
                for d in (r / "extra" / "depth").glob("*")
                if d.is_dir()
            }
        )
        for src_index, src_rel, dst_rel in merge_extra_ops(counts, depth_names):
            src = roots[src_index] / src_rel
            if not src.exists():
                continue
            out = temp / dst_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, out)
            else:
                shutil.copy2(src, out)

        os.replace(temp, dst)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    say(f"merged into {out_name}")
    return dst
