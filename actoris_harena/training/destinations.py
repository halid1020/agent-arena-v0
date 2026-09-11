"""The machines a training run can be sent to, and the commands that reach them.

A destination is a machine plus the little that differs about running training
on it: whether work is queued through Slurm or started directly, where its
scratch is, and what its GPU has been MEASURED to carry. Everything else -- the
dataset, the manifest, the camera views, ``test/system/long_vla_real.sh`` -- is
identical at both, which is the point: a run that was checked here behaves the
same on either machine.

Nothing here opens a connection or imports aiohttp. What a destination file may
say, which argv that becomes, and what an unreachable machine should tell the
operator are the interesting parts and are unit-tested directly; the routes are
in ``common.web.training_api`` and the CLI is ``tool/train_launch.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from actoris_harena.web.roots import parse_ssh_host

# Where the destinations file is. This was `Path(__file__).parents[3] / src /
# conf / train_destinations.yaml`, which was correct while this module lived
# inside the rig repo and silently wrong the instant it moved -- parents[3] of
# an installed package is wherever pip put it. The rig says where its own
# config is, exactly as it says what its cameras are called.
#
# Set it with set_destinations_path(); read it with destinations_path(). Every
# entry point can still pass an explicit path and skip this entirely.
_DESTINATIONS_PATH: "Path | None" = None
_REPO_ROOT: "Path | None" = None


def set_destinations_path(
    path: "Path | str", repo_root: "Path | str | None" = None
) -> None:
    """Declare where this rig's train_destinations.yaml lives.

    ``repo_root`` is what a destination whose ``repo`` is ``.`` resolves to --
    this rig's checkout, which the driver needs spelled out because it is
    started from wherever the console happened to be launched.
    """
    global _DESTINATIONS_PATH, _REPO_ROOT
    _DESTINATIONS_PATH = Path(path)
    if repo_root is not None:
        _REPO_ROOT = Path(repo_root)


def repo_root() -> Path:
    """This rig's checkout, for a destination whose ``repo`` is ``.``."""
    if _REPO_ROOT is None:
        raise RuntimeError(
            "a destination says repo: '.' but no rig checkout has been "
            "declared; pass repo_root= to set_destinations_path()"
        )
    return _REPO_ROOT


def destinations_path() -> Path:
    """The declared destinations file, or a clear refusal naming the fix."""
    if _DESTINATIONS_PATH is None:
        raise RuntimeError(
            "no train_destinations.yaml has been declared. A rig calls "
            "actoris_harena.training.destinations.set_destinations_path() "
            "once (so101_garment does it in common/rig_profile.py), or a "
            "caller passes an explicit path."
        )
    return _DESTINATIONS_PATH


KINDS = ("slurm", "ssh", "local")

# A remote path from this file reaches the destination's LOGIN SHELL unquoted,
# because that is the only way `~` and `$USER` can mean the remote home and the
# remote user -- which is the whole point of writing them rather than a literal
# path that exists on one machine. Unquoted means the characters below are all
# it may contain: everything that could start a second command, a redirect or a
# subshell is refused when the file is read, not when it is run.
_PATH_OK = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" "_-./~${}+:="
)

_REQUIRED = frozenset({"kind", "repo", "scratch", "stage"})
# `ssh` is required of every machine that is not this one; `outputs` names a
# second place run directories may be, which is what the local machine needs
# (a console started through setup.sh puts SO101_OUTPUT_DIR in the repo).
_OPTIONAL = frozenset({"ssh", "partition", "account", "limits", "outputs"})

# Long enough for a login node behind a jump host to answer, short enough that
# an operator watching a page does not think it has hung. A machine that needs
# longer than this is not reachable in any useful sense.
REACH_TIMEOUT_S = 20.0


# ── What the file may say ────────────────────────────────────────────────────


def _validate(name: str, entry: object, path: Path) -> "dict[str, Any]":
    """One destination, checked. Raises ValueError naming the offending key."""
    where = f"destination '{name}'"
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: {where} must be a mapping")
    keys = set(entry)
    missing = _REQUIRED - keys
    unknown = keys - _REQUIRED - _OPTIONAL
    if missing:
        raise ValueError(f"{path}: {where} is missing key(s): {sorted(missing)}")
    if unknown:
        raise ValueError(f"{path}: {where} has unknown key(s): {sorted(unknown)}")

    out = dict(entry)
    out["name"] = name
    if out["kind"] not in KINDS:
        raise ValueError(
            f"{path}: {where} has kind {out['kind']!r}; want one of {', '.join(KINDS)}"
        )
    if out["kind"] == "local":
        # The machine the console is running on. There is no host to name, and
        # naming one would be a lie the launcher would then try to reach.
        out.setdefault("ssh", "-")
        # `.` means "this checkout", which is the only local path that is the
        # same on every machine the console runs on -- and it must be resolved
        # here, because the driver is started from wherever the console
        # happened to be launched.
        if str(out.get("repo", "")).strip() in (".", "./"):
            out["repo"] = str(repo_root())
    else:
        if not out.get("ssh"):
            raise ValueError(f"{path}: {where} is not local and needs an 'ssh'")
        try:
            parse_ssh_host(str(out["ssh"]))
        except ValueError as exc:
            raise ValueError(f"{path}: {where} has an unusable 'ssh': {exc}") from exc
    if out["kind"] == "slurm" and not out.get("partition"):
        # Without one, sbatch falls back to whatever the cluster's default
        # partition is -- which on CREATE has no GPU, so the row would queue and
        # then train on a CPU. Naming it here is cheaper than discovering that.
        raise ValueError(f"{path}: {where} is 'slurm' and needs a 'partition'")
    for field in ("repo", "scratch", "stage", "outputs"):
        if field not in out:
            continue
        value = str(out[field]).strip()
        if not value:
            raise ValueError(f"{path}: {where} has an empty {field!r}")
        bad = sorted(set(value) - _PATH_OK)
        if bad:
            raise ValueError(
                f"{path}: {where} has {''.join(bad)!r} in {field!r}. These paths "
                "reach the remote shell unquoted so that '~' and '$USER' mean "
                "the REMOTE home and user; a path needing anything else cannot "
                "be sent safely."
            )
        out[field] = value
    out["limits"] = _validate_limits(name, out.get("limits") or {}, path)
    return out


def _validate_limits(name: str, limits: object, path: Path) -> "dict[str, dict]":
    if not isinstance(limits, dict):
        raise ValueError(f"{path}: destination '{name}' limits must be a mapping")
    out: "dict[str, dict]" = {}
    for policy, caps in limits.items():
        if not isinstance(caps, dict):
            raise ValueError(
                f"{path}: destination '{name}' limits.{policy} must be a mapping"
            )
        bad = set(caps) - {"batch", "steps"}
        if bad:
            raise ValueError(
                f"{path}: destination '{name}' limits.{policy} has unknown "
                f"key(s): {sorted(bad)}"
            )
        out[str(policy)] = {k: int(v) for k, v in caps.items()}
    return out


def load_destinations(path: "Path | str | None" = None) -> "dict[str, dict]":
    """Every destination in the file, validated. Raises ValueError / OSError."""
    p = Path(path) if path else destinations_path()
    if not p.exists():
        raise FileNotFoundError(f"no training destinations file at {p}")
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{p}: expected a non-empty top-level mapping")
    return {name: _validate(name, entry, p) for name, entry in data.items()}


def destination(name: str, path: "Path | str | None" = None) -> "dict[str, Any]":
    """One destination by name, or a ValueError listing the ones that exist."""
    known = load_destinations(path)
    if name not in known:
        raise ValueError(
            f"no training destination called {name!r}; "
            f"this file names {', '.join(sorted(known))}"
        )
    return known[name]


# ── Where things live on that machine ────────────────────────────────────────
#
# These paths are NOT expanded here. `~` and `$USER` are left exactly as
# written, because they mean the REMOTE home and the REMOTE user; expanding them
# against this machine would send a run to a directory that exists here and
# nowhere else. They reach the remote login shell, which is what resolves them.


def stage_dir(dest: "dict[str, Any]") -> str:
    """Where staged datasets go on that machine, unexpanded."""
    return str(dest["stage"]).format(scratch=str(dest["scratch"]))


def manifest_dir(dest: "dict[str, Any]") -> str:
    """Where the run matrices that were submitted are kept, on that machine."""
    return f"{dest['scratch']}/so101_manifests"


def dataset_dir(dest: "dict[str, Any]", dataset: str) -> str:
    return f"{stage_dir(dest)}/{dataset}"


def rsync_destination(dest: "dict[str, Any]", subpath: str = "") -> str:
    """The ``host:path`` rsync writes to. Trailing subpath is appended raw."""
    if dest.get("kind") == "local":
        return f"{stage_dir(dest)}{subpath}"
    return f"{dest['ssh']}:{stage_dir(dest)}{subpath}"


# ── The commands ─────────────────────────────────────────────────────────────


def ssh_argv(dest: "dict[str, Any]", remote_cmd: str) -> "list[str]":
    """Run one shell command on the destination. Pure.

    The local machine is not a special case anywhere else -- same driver, same
    lock, same run directories, same manifest -- so it is not one here either:
    the only difference is that there is no ssh in front of the command. Making
    that the ONE place the kind is consulted is what lets staging, the manifest,
    the dispatch, the status and the stop all work on it unchanged.

    ``BatchMode=yes`` because a password prompt in a web request simply hangs,
    and because an unattended launcher that stops to ask has already failed.
    """
    if dest.get("kind") == "local":
        return ["bash", "-lc", remote_cmd]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(REACH_TIMEOUT_S)}",
        str(dest["ssh"]),
        remote_cmd,
    ]


def rsync_argv(
    local: "Path | str", dest: "dict[str, Any]", dry_run: bool = False
) -> "list[str]":
    """Copy one dataset directory up to the destination's staging area. Pure.

    The source has NO trailing slash on purpose: rsync then creates
    ``<stage>/<dataset>`` rather than emptying the dataset into the stage.
    """
    argv = ["rsync", "-avP"]
    if dry_run:
        argv.append("--dry-run")
    argv += [str(Path(local)), rsync_destination(dest) + "/"]
    return argv


def mkdir_argv(dest: "dict[str, Any]", remote_dir: str) -> "list[str]":
    return ssh_argv(dest, f"mkdir -p {remote_dir}")


# ── Is it there? ─────────────────────────────────────────────────────────────


def reach_message(returncode: int, stderr: str, dest_name: str) -> "str | None":
    """``None`` if the machine answered, else what the operator should do. Pure.

    SSH reports these in its own words, which say nothing about training. Each
    of the four has a different fix and only one of them is a bug.
    """
    if returncode == 0:
        return None
    text = (stderr or "").strip()
    low = text.lower()
    if not text and returncode == 124:
        return (
            f"{dest_name} did not answer within {int(REACH_TIMEOUT_S)}s. If it is "
            "on a university network, are you on the VPN?"
        )
    if "operation timed out" in low or "connection timed out" in low:
        return (
            f"{dest_name} did not answer in time. If it is on a university "
            "network, are you on the VPN?"
        )
    if "permission denied" in low or "publickey" in low:
        return (
            f"{dest_name} did not accept your key. Authentication here is by key "
            f"only — run 'ssh-copy-id {dest_name}' from a terminal once. " + text
        )
    if "host key verification failed" in low or "known_hosts" in low:
        return (
            f"{dest_name}'s host key is not known yet, and this cannot ask you to "
            f"accept it — 'ssh {dest_name}' once from a terminal, accept the key, "
            "then try again. " + text
        )
    if "could not resolve" in low or "name or service not known" in low:
        return (
            f"{dest_name} could not be resolved. Is it an ~/.ssh/config alias on "
            "this machine? " + text
        )
    return text or f"ssh to {dest_name} failed (exit {returncode})"


def reachable(dest: "dict[str, Any]", timeout: float = REACH_TIMEOUT_S) -> "str | None":
    """``None`` if the machine answers, else the sentence to show. Never raises.

    A timeout is reported as exit 124, the same code ``timeout(1)`` uses, so
    ``reach_message`` can tell "did not answer" from "refused me".
    """
    # The local machine is reachable by definition; what can be missing is the
    # checkout the driver is started from, and saying "cannot reach localhost"
    # would be no help at all.
    probe = (
        f"test -f {dest['repo']}/hpc/gpu_box_run.sh"
        if dest.get("kind") == "local"
        else "true"
    )
    argv = ssh_argv(dest, probe)
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout + 5.0
        )
    except subprocess.TimeoutExpired:
        return reach_message(124, "", str(dest.get("name", dest["ssh"])))
    except OSError as exc:
        return f"could not run ssh: {exc}"
    if dest.get("kind") == "local":
        return (
            None
            if proc.returncode == 0
            else f"no so101_garment checkout at {dest['repo']} on this machine"
        )
    return reach_message(
        proc.returncode, proc.stderr, str(dest.get("name", dest["ssh"]))
    )


# ── Remembering what was launched ────────────────────────────────────────────
#
# Copied in shape from actoris_harena.web.roots: a corrupt or missing file reads empty
# and a failed write is not fatal, because this is a convenience -- the runs
# themselves live on the destination, and losing this record loses only the
# console's list of them, which `--status` can rebuild by asking the machine.

MAX_RUNS = 40


def load_runs(path: "Path | str") -> "dict[str, Any]":
    try:
        with open(path, "r") as f:
            state = yaml.safe_load(f)
        if not isinstance(state, dict):
            raise ValueError
    except (OSError, ValueError, yaml.YAMLError):
        return {"runs": []}
    runs = [r for r in state.get("runs") or [] if isinstance(r, dict) and r.get("id")]
    return {"runs": runs}


def remember_run(state: "dict[str, Any]", run: "dict[str, Any]") -> "dict[str, Any]":
    """The state with this run in front, most recent first. Pure."""
    rest = [r for r in state.get("runs") or [] if r.get("id") != run.get("id")]
    return {"runs": [dict(run), *rest][:MAX_RUNS]}


def save_runs(path: "Path | str", state: "dict[str, Any]") -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{path}.tmp")
        with open(tmp, "w") as f:
            yaml.safe_dump(state, f, sort_keys=False)
        os.replace(tmp, path)
    except OSError:
        pass
