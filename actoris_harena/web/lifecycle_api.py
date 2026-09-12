"""Routes for the whole-dataset operations: create, rename, delete, merge.

The rules all live in ``actoris_harena.web.lifecycle``; this layer only reads the
request and runs the slow parts off the event loop.

A merge re-encodes every episode of every source, and freeing the bytes of a
deleted dataset takes about as long, so both outlive their request as JOBS
(``actoris_harena.web.jobs``): the POST returns a record and the browser's dock watches
it.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from aiohttp import web  # type: ignore[import]

from actoris_harena.recording.dataset_edit import ReadOnlyDatasetError
from actoris_harena.web.jobs import finish, new_job, refuse_while_busy
from actoris_harena.web.lifecycle import (
    delete_dataset,
    directory_size,
    is_working_dir,
    merge_compatibility,
    merge_datasets,
    purge,
    read_dataset_meta,
    rename_dataset,
    valid_dataset_name,
)
from actoris_harena.web.util import in_executor


def _body_name(body: "dict[str, Any]", key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise web.HTTPBadRequest(text=f"missing {key}")
    return value.strip()


async def handle_check_name(request: web.Request) -> web.Response:
    """Is this a usable name for a new dataset? Creation itself needs a session.

    A dataset is created by recording into it, so the console cannot make an
    empty one -- what it can do is tell the operator, while they type, that the
    name is fine and free.
    """
    app = request.app
    body = await request.json()
    name = _body_name(body, "name")
    problem = valid_dataset_name(name)
    taken = problem is None and (Path(app["root"]) / name).exists()
    return web.json_response(
        {
            "name": name,
            "ok": problem is None and not taken,
            "problem": problem or (f"{name!r} already exists" if taken else None),
            "exists": taken,
        }
    )


async def handle_rename(request: web.Request) -> web.Response:
    app = request.app
    name = request.match_info["name"]
    body = await request.json()
    new = _body_name(body, "new")
    if is_working_dir(name):
        raise web.HTTPNotFound(text=f"no dataset {name!r}")
    try:
        path = await in_executor(app, rename_dataset, app["root"], name, new)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc))
    except FileExistsError as exc:
        raise web.HTTPConflict(text=str(exc))
    except (ReadOnlyDatasetError, OSError) as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"name": path.name})


async def handle_remove(request: web.Request) -> web.Response:
    """Delete a whole dataset: rename it into the trash now, free the bytes after."""
    app = request.app
    name = request.match_info["name"]
    if is_working_dir(name):
        raise web.HTTPNotFound(text=f"no dataset {name!r}")
    try:
        trash = await in_executor(app, delete_dataset, app["root"], name)
    except FileNotFoundError as exc:
        raise web.HTTPNotFound(text=str(exc))
    except (ReadOnlyDatasetError, OSError) as exc:
        raise web.HTTPConflict(text=str(exc))
    # The dataset is already unreachable under its name, so the operator is not
    # kept waiting; removing a hundred gigabytes of video is a job.
    job = new_job(app, "delete", name, message=f"freeing {name}")

    def run() -> None:
        purge(trash)  # never raises: the dataset is already gone from the list
        finish(job, "done", f"deleted {name}")

    asyncio.get_running_loop().run_in_executor(app["job_executor"], run)
    return web.json_response({"deleted": name, "job": job["id"]})


def _merge_sources(root: Path, names: "list[str]") -> "list[dict[str, Any]]":
    """What ``merge_compatibility`` needs to judge each source. Blocking."""
    metas = []
    for name in names:
        path = root / name
        if not path.is_dir():
            raise FileNotFoundError(f"no dataset {name!r}")
        meta = read_dataset_meta(path)
        meta["size_bytes"] = directory_size(path)
        metas.append(meta)
    return metas


async def handle_merge_check(request: web.Request) -> web.Response:
    app = request.app
    body = await request.json()
    names = [str(n) for n in body.get("names") or []]
    out_name = _body_name(body, "name")
    reasons = await in_executor(app, _merge_reasons, app["root"], names, out_name)
    return web.json_response({"ok": not reasons, "reasons": reasons})


def _merge_reasons(root: Path, names: "list[str]", out_name: str) -> "list[str]":
    """Blocking: read every source, then apply the pure rules."""
    root = Path(root)
    try:
        metas = _merge_sources(root, names)
    except FileNotFoundError as exc:
        return [str(exc)]
    taken = tuple(p.name for p in root.iterdir() if p.is_dir())
    free = shutil.disk_usage(root).free
    return merge_compatibility(metas, out_name, free, taken)


async def handle_merge(request: web.Request) -> web.Response:
    app = request.app
    body = await request.json()
    names = [str(n) for n in body.get("names") or []]
    out_name = _body_name(body, "name")
    drop_sources = bool(body.get("delete_sources"))

    refuse_while_busy(app)

    reasons = await in_executor(app, _merge_reasons, app["root"], names, out_name)
    if reasons:
        raise web.HTTPBadRequest(text="; ".join(reasons))

    job = new_job(app, "merge", out_name, sources=names, delete_sources=drop_sources)
    root = Path(app["root"])

    def run() -> None:
        try:
            merge_datasets(
                root,
                names,
                out_name,
                progress=lambda m: job.__setitem__("message", m),
            )
        except BaseException as exc:  # noqa: B036 - reported, not swallowed
            finish(job, "failed", str(exc) or exc.__class__.__name__)
            return
        # Only now, with the merged dataset in place, may the sources go: the
        # copy the operator asked for exists, so this can never be the step
        # that loses the recordings.
        if drop_sources:
            for name in names:
                job["message"] = f"deleting source {name}"
                try:
                    purge(delete_dataset(root, name))
                except BaseException as exc:  # noqa: B036 - reported, not swallowed
                    finish(
                        job,
                        "failed",
                        f"merged into {out_name}, but {name} could not be "
                        f"removed: {exc or exc.__class__.__name__}",
                    )
                    return
        finish(job, "done", f"merged into {out_name}")

    asyncio.get_running_loop().run_in_executor(app["job_executor"], run)
    return web.json_response(job)


def add_lifecycle_routes(app: web.Application) -> None:
    """Register create/rename/delete/merge. ``jobs`` and ``job_executor`` must exist."""
    app.add_routes(
        [
            web.post("/api/datasets/check-name", handle_check_name),
            web.post("/api/datasets/merge", handle_merge),
            web.post("/api/datasets/merge-check", handle_merge_check),
            web.post("/api/datasets/{name}/rename", handle_rename),
            web.post("/api/datasets/{name}/remove", handle_remove),
        ]
    )
