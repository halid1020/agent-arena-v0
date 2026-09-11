"""Grouping runs into experiments, because a run directory is not one.

A run is named ``<dataset>__<cameras>`` and lives on one machine. That names
what it trained on and nothing about *why*: the three finished thanos runs are
one comparison, the six camera-ablation rows on CREATE are another, and both
read as a flat list of directories. So the operator names a project and puts
runs in it; the dataset stays an attribute of the run, shown as a column,
because two datasets in one project is a normal experiment and one dataset
across two projects is normal too.

Two things follow from runs being DISCOVERED rather than launched here.

**A membership outlives the run it names.** Nineteen of the run directories on
these machines were started from a terminal and nine predate the launcher, so a
project cannot be a field on a launch record -- there is no record. It is a
list of keys, kept separately, and a key whose machine is unreachable today is
still a member: it renders as missing rather than disappearing, because a run
that vanished from a list reads as a run that was deleted.

**Nothing here is authoritative about what exists.** This module never decides
whether a run is real; ``common.training.runs`` does. A project holding a key
nothing answers for is a normal state, not a corrupt file.

Pure: dicts, strings and one YAML file. No aiohttp, no SSH. The routes are in
``projects_api``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

#: The key a run is remembered by. Machine, directory and policy -- the same
#: triple ``/api/training/progress`` is addressed with, so a membership can be
#: resolved without asking any machine anything.
KEY_PARTS = ("dest", "run", "policy")
KEY_SEPARATOR = "|"

MAX_NAME = 60
MAX_PROJECTS = 200

#: Deliberately permissive -- a project name is a label a person chose, not a
#: path or a shell word, and it is never interpolated into either. What is
#: refused is only what would make it unreadable or ambiguous in a list.
_NAME_RE = re.compile(r"^[^\x00-\x1f/\\]+$")

#: Two groupings every operator has before creating anything, so the view is
#: useful on a machine where no project exists yet. Not stored, and not
#: nameable: a real project called "All runs" would silently shadow one.
BUILTIN = ("__all__", "__unassigned__")
BUILTIN_LABELS = {"__all__": "All runs", "__unassigned__": "Unassigned"}


class ProjectError(ValueError):
    """A project operation that cannot be done. The message says why."""


# ── Keys ─────────────────────────────────────────────────────────────────────


def run_key(dest: str, run: str, policy: str) -> str:
    """``thanos|towel__all|act``. Pure.

    The separator is refused inside a part rather than escaped: none of the
    three can contain it (``common.training.runs.check_name`` allows letters,
    digits and ``. _ + -``), so a part that does is not one of ours.
    """
    parts = (str(dest), str(run), str(policy))
    for part in parts:
        if not part:
            raise ProjectError("a run key needs a machine, a directory and a policy")
        if KEY_SEPARATOR in part:
            raise ProjectError(
                f"{part!r} contains {KEY_SEPARATOR!r}, which separates the parts "
                "of a run key; this is not a run this rig created."
            )
    return KEY_SEPARATOR.join(parts)


def split_key(key: str) -> "dict[str, str]":
    """The inverse of :func:`run_key`. Raises on anything else."""
    parts = str(key).split(KEY_SEPARATOR)
    if len(parts) != len(KEY_PARTS):
        raise ProjectError(
            f"{key!r} is not a run key; want "
            f"{KEY_SEPARATOR.join(KEY_PARTS)} and nothing else."
        )
    return dict(zip(KEY_PARTS, parts))


def key_of(entry: "dict[str, Any]") -> str:
    """The key for a discovered run, as ``/api/training/discovered`` returns it."""
    return run_key(entry.get("dest", ""), entry.get("run", ""), entry.get("policy", ""))


# ── Names ────────────────────────────────────────────────────────────────────


def check_project_name(name: str) -> str:
    """Validate and normalise a project name, returning it."""
    text = str(name or "").strip()
    if not text:
        raise ProjectError("give the project a name")
    if len(text) > MAX_NAME:
        raise ProjectError(f"a project name is at most {MAX_NAME} characters")
    if not _NAME_RE.match(text):
        raise ProjectError(
            "a project name cannot contain a slash, a backslash or a control "
            "character -- it is a label, and those make it unreadable in a list."
        )
    if text in BUILTIN or text in BUILTIN_LABELS.values():
        raise ProjectError(
            f"{text!r} is one of the built-in groupings ("
            f"{', '.join(BUILTIN_LABELS.values())}), which every machine has "
            "already. Pick another name."
        )
    return text


# ── The file ─────────────────────────────────────────────────────────────────


def load(path: "Path | str") -> "dict[str, Any]":
    """Read the store. A missing or unreadable file is an empty one.

    Never raises: this is read on every page load, and a hand-edited file that
    no longer parses must cost the projects, not the whole Training tab.
    """
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {"projects": []}
    projects = raw.get("projects") if isinstance(raw, dict) else None
    if not isinstance(projects, list):
        return {"projects": []}
    out = []
    for item in projects:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        runs = item.get("runs")
        out.append(
            {
                "name": str(item["name"]),
                "created": item.get("created"),
                "notes": str(item.get("notes") or ""),
                "runs": [str(r) for r in runs] if isinstance(runs, list) else [],
            }
        )
    return {"projects": out}


def save(path: "Path | str", store: "dict[str, Any]") -> None:
    """Write the store, atomically -- the page reads it while jobs write it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(store, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    tmp.replace(target)


# ── Operations, all pure over a loaded store ─────────────────────────────────


def find(store: "dict[str, Any]", name: str) -> "dict[str, Any] | None":
    for project in store["projects"]:
        if project["name"] == name:
            return project
    return None


def create(store: "dict[str, Any]", name: str, notes: str = "") -> "dict[str, Any]":
    checked = check_project_name(name)
    if find(store, checked) is not None:
        raise ProjectError(f"there is already a project called {checked!r}")
    if len(store["projects"]) >= MAX_PROJECTS:
        raise ProjectError(f"at most {MAX_PROJECTS} projects")
    project = {
        "name": checked,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": str(notes or ""),
        "runs": [],
    }
    store["projects"].append(project)
    return project


def rename(store: "dict[str, Any]", name: str, new_name: str) -> "dict[str, Any]":
    project = find(store, name)
    if project is None:
        raise ProjectError(f"no project called {name!r}")
    checked = check_project_name(new_name)
    if checked != name and find(store, checked) is not None:
        raise ProjectError(f"there is already a project called {checked!r}")
    project["name"] = checked
    return project


def delete(store: "dict[str, Any]", name: str) -> None:
    """Forget a project. The runs themselves are untouched -- this is a label."""
    if find(store, name) is None:
        raise ProjectError(f"no project called {name!r}")
    store["projects"] = [p for p in store["projects"] if p["name"] != name]


def assign(store: "dict[str, Any]", name: str, keys: "list[str]") -> "dict[str, Any]":
    """Add runs to a project, keeping order and adding each at most once."""
    project = find(store, name)
    if project is None:
        raise ProjectError(f"no project called {name!r}")
    have = set(project["runs"])
    for key in keys:
        split_key(key)  # validate; raises on anything that is not one
        if key not in have:
            project["runs"].append(key)
            have.add(key)
    return project


def unassign(store: "dict[str, Any]", name: str, keys: "list[str]") -> "dict[str, Any]":
    project = find(store, name)
    if project is None:
        raise ProjectError(f"no project called {name!r}")
    drop = set(keys)
    project["runs"] = [k for k in project["runs"] if k not in drop]
    return project


def projects_of(store: "dict[str, Any]", key: str) -> "list[str]":
    """Every project a run belongs to. A run may be in more than one."""
    return [p["name"] for p in store["projects"] if key in p["runs"]]


def missing_runs(
    store: "dict[str, Any]", runs: "list[dict[str, Any]]", name: str
) -> "list[dict[str, str]]":
    """Members of a project that no machine answered for.

    Reported rather than dropped. A machine that is off the VPN this morning
    has not deleted anything, and a list that silently shrinks is the one way
    this view could mislead about what was run.
    """
    project = find(store, name)
    if project is None:
        return []
    present = set()
    for entry in runs:
        try:
            present.add(key_of(entry))
        except ProjectError:
            continue
    out = []
    for key in project["runs"]:
        if key in present:
            continue
        try:
            out.append(split_key(key))
        except ProjectError:
            continue
    return out
