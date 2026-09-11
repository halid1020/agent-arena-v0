"""Which collection directory the console works on, local or on another machine.

Everything the console does to a dataset is filesystem work: range reads from
the recorded video files, ``os.replace`` to rename or trash a directory, and
LeRobot's aggregation to merge. A drive on another machine is therefore reached
by MOUNTING it -- ``sshfs`` over the operator's existing SSH access -- and then
pointing the root at the mount, so every other part of the console goes on
working on ordinary paths and knows nothing about the network.

Authentication is the operator's key or agent (``BatchMode=yes``): no password
is ever typed into the browser. A host that would ask for one fails here with a
message saying what to do about it instead.

Nothing in this module imports aiohttp. The rules -- what a target may look
like, which command that becomes, what a failure means, which mounts belong to
the console -- are the interesting part and are unit-tested directly; the routes
are in ``roots_api``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Kept together because they are what makes an unattended mount behave: it
# reconnects after a dropped link, notices a dead one in about a minute, never
# prompts (a prompt in a web request would simply hang), and presents the remote
# files as the local user so a recording can be written back.
SSHFS_OPTIONS = (
    "reconnect",
    "ServerAliveInterval=15",
    "ServerAliveCountMax=3",
    "BatchMode=yes",
    "idmap=user",
    "follow_symlinks",
)
MOUNT_TIMEOUT_S = 30.0
MAX_RECENT = 8

_HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")
_USER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


# ── The target ───────────────────────────────────────────────────────────────


def parse_ssh_host(text: str) -> "tuple[str | None, str]":
    """``[user@]host`` -> ``(user, host)``. Raises ValueError. Pure.

    ``host`` may be an ``~/.ssh/config`` alias rather than a real name; what is
    checked is that it could not be read as an option or a second command. Both
    halves go on a command line unquoted, so a host like ``-oProxyCommand=…`` or
    ``box;rm -rf /`` is refused here rather than run.
    """
    target = str(text).strip()
    if not target:
        raise ValueError("give a machine, as host or user@host")
    if any(c.isspace() for c in target):
        raise ValueError("a remote machine cannot contain spaces")
    user: "str | None" = None
    if "@" in target:
        user, _, target = target.partition("@")
        if not _USER_RE.match(user):
            raise ValueError(f"{user!r} is not a usable user name")
    if not _HOST_RE.match(target):
        raise ValueError(f"{target!r} is not a usable host name")
    return user, target


def parse_ssh_target(text: str) -> "tuple[str | None, str, str]":
    """``user@host:/path`` -> ``(user, host, path)``. Raises ValueError. Pure.

    The user may be left out (SSH then uses its own config), and a path that is
    not absolute is taken from the remote home directory, exactly as ``scp``
    and ``sshfs`` read it.
    """
    target = str(text).strip()
    if not target:
        raise ValueError("give a remote directory, as user@host:/path")
    if any(c.isspace() for c in target):
        raise ValueError("a remote target cannot contain spaces")
    machine, sep, path = target.rpartition(":")
    if not sep:
        raise ValueError("missing the ':' before the remote path (user@host:/path)")
    user, host = parse_ssh_host(machine)
    if not path:
        raise ValueError("give the directory on the remote machine after the ':'")
    return user, host, path


def format_target(user: "str | None", host: str, path: str) -> str:
    """The canonical text form, as sshfs wants it."""
    return f"{user}@{host}:{path}" if user else f"{host}:{path}"


def mount_dir(base: Path, host: str, path: str) -> Path:
    """Where a remote directory is mounted. Deterministic, so a repeat reuses it."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") or "root"
    return Path(base) / f"{host}_{slug}"


def sshfs_argv(
    target: str,
    mountpoint: Path,
    port: "int | None" = None,
    identity: "str | None" = None,
) -> "list[str]":
    """The exact sshfs command line for this mount. Pure."""
    argv = ["sshfs", target, str(mountpoint), "-o", ",".join(SSHFS_OPTIONS)]
    if port:
        argv += ["-p", str(int(port))]
    if identity:
        argv += ["-o", f"IdentityFile={Path(identity).expanduser()}"]
    return argv


def sshfs_available() -> bool:
    return shutil.which("sshfs") is not None


def mount_message(returncode: int, stderr: str) -> "str | None":
    """``None`` if the mount worked, else what the operator should do. Pure.

    sshfs reports failures through ssh's own words, which say nothing about a
    console; these are the three that actually happen, and none of them can be
    fixed from the browser.
    """
    if returncode == 0:
        return None
    text = (stderr or "").strip()
    low = text.lower()
    if "permission denied" in low or "publickey" in low:
        return (
            "the remote machine did not accept your key. Authentication here is "
            "by key only — run 'ssh-copy-id <user>@<host>' from a terminal once, "
            "then try again. " + text
        )
    if "host key verification failed" in low or "known_hosts" in low:
        return (
            "the remote machine's host key is not known yet, and this cannot ask "
            "you to accept it — 'ssh <user>@<host>' once from a terminal, accept "
            "the key, then try again. " + text
        )
    if "could not resolve" in low or "name or service not known" in low:
        return f"the host name could not be resolved. {text}"
    return text or f"sshfs failed (exit {returncode})"


# ── Mounts ───────────────────────────────────────────────────────────────────


def _unescape(field: str) -> str:
    """/proc/mounts writes octal escapes for the awkward characters."""
    out, i = [], 0
    while i < len(field):
        if field[i] == "\\" and field[i + 1 : i + 4].isdigit():
            out.append(chr(int(field[i + 1 : i + 4], 8)))
            i += 4
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def parse_mounts(text: str) -> "list[dict[str, str]]":
    """Every fuse.sshfs line of ``/proc/mounts`` as ``{target, path}``. Pure."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[2] not in ("fuse.sshfs", "fuse.sshfs2"):
            continue
        out.append({"target": _unescape(parts[0]), "path": _unescape(parts[1])})
    return out


def sshfs_mounts() -> "list[dict[str, str]]":
    try:
        with open("/proc/mounts", "r") as f:
            return parse_mounts(f.read())
    except OSError:
        return []


def is_mounted(path: Path) -> bool:
    want = str(Path(path))
    return any(m["path"] == want for m in sshfs_mounts())


def mount_remote(
    target: str,
    mountpoint: Path,
    port: "int | None" = None,
    identity: "str | None" = None,
) -> "str | None":
    """Mount ``target`` at ``mountpoint``. ``None`` on success, else the message."""
    mountpoint = Path(mountpoint)
    if is_mounted(mountpoint):
        return None
    if not sshfs_available():
        return (
            "sshfs is not installed on this machine, and it is what mounts a "
            "remote collection directory: 'sudo apt install sshfs'."
        )
    try:
        mountpoint.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"cannot create the mount point {mountpoint}: {exc}"
    if any(mountpoint.iterdir()):
        return f"the mount point {mountpoint} is not empty"
    try:
        done = subprocess.run(
            sshfs_argv(target, mountpoint, port, identity),
            capture_output=True,
            text=True,
            timeout=MOUNT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"the remote machine did not answer within {int(MOUNT_TIMEOUT_S)}s"
    return mount_message(done.returncode, done.stderr)


def unmount(path: Path) -> "str | None":
    """Release a mount. ``None`` on success (including 'was not mounted')."""
    path = Path(path)
    if not is_mounted(path):
        return None
    for binary in ("fusermount3", "fusermount"):
        if shutil.which(binary) is None:
            continue
        done = subprocess.run(
            [binary, "-u", str(path)], capture_output=True, text=True, timeout=20
        )
        if done.returncode == 0:
            return None
        return (done.stderr or "").strip() or f"{binary} failed"
    return "neither fusermount3 nor fusermount is installed"


# ── The chosen directory ─────────────────────────────────────────────────────


def root_problem(path: "Path | None") -> "str | None":
    """Why this cannot be a collection directory, in the operator's terms."""
    if path is None or not str(path).strip():
        return "give a directory"
    path = Path(path).expanduser()
    if not path.is_absolute():
        return f"{path} is not an absolute path"
    if not path.exists():
        return f"{path} does not exist"
    if not path.is_dir():
        return f"{path} is not a directory"
    if not os.access(path, os.R_OK | os.X_OK):
        return f"{path} cannot be read"
    return None


def looks_like_collection(path: Path) -> bool:
    """Does this directory hold datasets? (a child with LeRobot metadata)"""
    try:
        for child in sorted(Path(path).iterdir())[:60]:
            if (child / "meta" / "info.json").is_file():
                return True
    except OSError:
        pass
    return False


# ── Remembering ──────────────────────────────────────────────────────────────


def load_state(path: Path) -> "dict[str, Any]":
    """The remembered directories. A missing or damaged file is simply empty."""
    try:
        with open(path, "r") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError
    except (OSError, ValueError):
        return {"last": None, "recent": []}
    state.setdefault("last", None)
    recent = [
        r for r in state.get("recent") or [] if isinstance(r, dict) and r.get("path")
    ]
    state["recent"] = recent
    return state


def remember(
    state: "dict[str, Any]", path: Path, kind: str, target: "str | None" = None
) -> "dict[str, Any]":
    """The state with this directory in front, most recent first. Pure."""
    entry = {"path": str(path), "kind": kind, "target": target}
    recent = [r for r in state.get("recent") or [] if r.get("path") != str(path)]
    return {"last": str(path), "recent": [entry, *recent][:MAX_RECENT]}


def save_state(path: Path, state: "dict[str, Any]") -> None:
    """Write the remembered directories. Never fatal -- this is a convenience."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass
