"""The console's background work, and what the page's dock is shown of it.

Three operations outlive the request that starts them: a merge, which
re-encodes every episode of every source; a compaction, which rewrites a whole
dataset to really remove the episodes marked for deletion; and freeing the bytes
of a deleted dataset. Each runs on ONE worker thread, so two rewrites can never
touch the same collection directory at once, and each leaves a small record here
that the browser polls -- state, elapsed time and a message -- so the page stays
usable while it works and a failure is reported where the operator is looking
rather than as a dead request.

The records are kept in ``app["jobs"]`` and the worker is ``app["job_executor"]``;
both are created by ``tool/rig_web.py``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from aiohttp import web  # type: ignore[import]

# How much history the dock is given. A finished job is worth seeing for a
# while (it says what the merge was called, or why it failed) but not forever.
MAX_JOBS = 20
JOB_TTL_S = 3600.0


def prune_jobs(jobs: "dict[str, dict[str, Any]]", now: float) -> None:
    """Forget finished jobs that are old or surplus. Running jobs always stay."""
    finished = sorted(
        (j for j in jobs.values() if j["state"] != "running"),
        key=lambda j: j.get("finished") or j["started"],
    )
    for job in finished:
        stale = now - (job.get("finished") or job["started"]) > JOB_TTL_S
        if stale or len(jobs) > MAX_JOBS:
            jobs.pop(job["id"], None)


def new_job(app: web.Application, kind: str, name: str, **extra: Any) -> dict:
    job = {
        "id": uuid.uuid4().hex[:8],
        "kind": kind,
        "state": "running",
        "message": "starting",
        "name": name,
        "started": time.time(),
        **extra,
    }
    app["jobs"][job["id"]] = job
    prune_jobs(app["jobs"], job["started"])
    return job


def finish(job: dict, state: str, message: str) -> None:
    job["state"] = state
    job["message"] = message
    job["finished"] = time.time()


def running_job(app: web.Application) -> "dict[str, Any] | None":
    """The job in progress, if any. One worker thread means at most one."""
    for job in app["jobs"].values():
        if job["state"] == "running":
            return job
    return None


def refuse_while_busy(app: web.Application) -> None:
    """Refuse a second rewrite of the same drive, naming the one already going."""
    job = running_job(app)
    if job is not None:
        raise web.HTTPConflict(text=f"{job['kind']} already running ({job['id']})")


async def handle_jobs(request: web.Request) -> web.Response:
    """Every job the console remembers, newest first -- what the dock shows."""
    jobs = request.app["jobs"]
    prune_jobs(jobs, time.time())
    return web.json_response(sorted(jobs.values(), key=lambda j: -j["started"]))


async def handle_job(request: web.Request) -> web.Response:
    job = request.app["jobs"].get(request.match_info["id"])
    if job is None:
        raise web.HTTPNotFound(text="no such job")
    return web.json_response(job)


def add_job_routes(app: web.Application) -> None:
    """Register the dock's routes. ``jobs`` and ``job_executor`` must exist."""
    app.add_routes(
        [
            web.get("/api/jobs", handle_jobs),
            web.get("/api/datasets/jobs/{id}", handle_job),
        ]
    )
