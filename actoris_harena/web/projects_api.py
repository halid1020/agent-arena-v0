"""Routes for the Training tab's projects.

Thin, like ``training_api``: the rules are ``actoris_harena.web.projects`` and are
unit-tested directly. This layer reads a request, keeps the file work off the
event loop, and turns a :class:`ProjectError` into a 400 with the sentence it
carries.

Not in ``ROOT_REQUIRED_PATHS``, deliberately. A project groups runs on other
machines; it has nothing to do with the collection directory, and the reading
half of the Training tab has to work with none chosen -- which is how it is
used from a laptop watching a GPU box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiohttp import web  # type: ignore[import]

from actoris_harena.web import projects
from actoris_harena.web.projects import ProjectError
from actoris_harena.web.util import in_executor


def _file(app: web.Application) -> Path:
    return app["projects_file"]


async def _load(app: web.Application) -> "dict[str, Any]":
    return await in_executor(app, lambda: projects.load(_file(app)))


async def _mutate(app: web.Application, change) -> "dict[str, Any]":
    """Read, apply, write -- under the app's lock, so two tabs cannot interleave.

    The whole store is a few kilobytes, so this is a read-modify-write rather
    than anything cleverer; what it must not do is lose a project because two
    requests each wrote what they had read before the other's change landed.
    """
    lock = app.setdefault("projects_lock", __import__("asyncio").Lock())
    async with lock:

        def work() -> "dict[str, Any]":
            store = projects.load(_file(app))
            change(store)
            projects.save(_file(app), store)
            return store

        return await in_executor(app, work)


def _body_keys(body: "dict[str, Any]") -> "list[str]":
    keys = body.get("runs")
    if not isinstance(keys, list) or not keys:
        raise web.HTTPBadRequest(text="give the runs to move, as a list of run keys")
    return [str(k) for k in keys]


async def handle_list(request: web.Request) -> web.Response:
    store = await _load(request.app)
    return web.json_response(
        {
            "projects": store["projects"],
            # Named here rather than in the page so the two cannot disagree
            # about which pseudo-projects exist.
            "builtin": [
                {"name": name, "label": projects.BUILTIN_LABELS[name]}
                for name in projects.BUILTIN
            ],
        }
    )


async def handle_create(request: web.Request) -> web.Response:
    body = await request.json()
    name = str(body.get("name") or "")
    notes = str(body.get("notes") or "")
    created: "dict[str, Any]" = {}

    def change(store):
        created.update(projects.create(store, name, notes))

    try:
        await _mutate(request.app, change)
    except ProjectError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    return web.json_response(created)


async def handle_update(request: web.Request) -> web.Response:
    """Rename, and move runs in or out. One route, because the page does all
    three from the same selection and a rename that half-applied would be worse
    than one that did not start."""
    name = request.match_info["name"]
    body = await request.json()
    result: "dict[str, Any]" = {}

    def change(store):
        project = projects.find(store, name)
        if project is None:
            raise ProjectError(f"no project called {name!r}")
        if body.get("add"):
            projects.assign(store, name, [str(k) for k in body["add"]])
        if body.get("remove"):
            projects.unassign(store, name, [str(k) for k in body["remove"]])
        if body.get("notes") is not None:
            project["notes"] = str(body["notes"])
        if body.get("name") and str(body["name"]) != name:
            projects.rename(store, name, str(body["name"]))
        result.update(projects.find(store, str(body.get("name") or name)) or {})

    try:
        await _mutate(request.app, change)
    except ProjectError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    return web.json_response(result)


async def handle_delete(request: web.Request) -> web.Response:
    name = request.match_info["name"]

    def change(store):
        projects.delete(store, name)

    try:
        await _mutate(request.app, change)
    except ProjectError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    # Only the label is gone; nothing on any machine was touched, and the page
    # says so rather than letting "delete" read as "delete the runs".
    return web.json_response({"deleted": name, "runs_kept": True})


def add_project_routes(app: web.Application) -> None:
    app.router.add_get("/api/training/projects", handle_list)
    app.router.add_post("/api/training/projects", handle_create)
    app.router.add_patch("/api/training/projects/{name}", handle_update)
    app.router.add_delete("/api/training/projects/{name}", handle_delete)
