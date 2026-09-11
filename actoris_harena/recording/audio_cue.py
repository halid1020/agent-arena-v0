"""Non-blocking audible cues for data collection (record start / stop).

A short sound plays when an episode starts recording and when it stops-saves, so
the operator gets an audible confirmation without watching the terminal. Playback
is best-effort and MUST never block the record loop or raise: it launches a
system audio player (``paplay`` / ``aplay`` / ``ffplay``) as a detached
subprocess and falls back to the terminal bell (``\\a``) when no player is on the
PATH or the sound file is missing. The two shipped cues live in ``sounds/`` next
to this module; a config value that is a bare filename resolves against that
directory, an absolute path is used as given.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_SOUNDS_DIR = Path(__file__).resolve().parent / "sounds"

# Player command prefixes, best first. Each plays a file given as the final argv
# element; all handle the shipped 16-bit PCM WAV. The flags keep them quiet and
# non-interactive. Order matters: modern desktops run PipeWire/PulseAudio, where
# raw ALSA (``aplay``) often plays to a dead default device and is silent, so the
# server-aware players (``pw-play``, ``paplay``, ``ffplay``) come first and
# ``aplay`` is only the last resort.
_PLAYERS: tuple[tuple[str, list[str]], ...] = (
    ("pw-play", []),
    ("paplay", []),
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ("aplay", ["-q"]),
)


def resolve_player() -> "list[str] | None":
    """First available system audio player as an argv prefix, or None. Pure-ish."""
    for exe, flags in _PLAYERS:
        path = shutil.which(exe)
        if path:
            return [path, *flags]
    return None


def resolve_sound(name: "str | None") -> "Path | None":
    """A sound name/path → an existing file Path, or None. Pure.

    A bare filename resolves against the packaged ``sounds/`` directory; an
    absolute (or relative-existing) path is honoured as given.
    """
    if not name:
        return None
    p = Path(name)
    if p.is_absolute():
        return p if p.is_file() else None
    packaged = _SOUNDS_DIR / name
    if packaged.is_file():
        return packaged
    return p if p.is_file() else None


class AudioCue:
    """Plays start/stop cues asynchronously; a safe no-op when disabled."""

    def __init__(
        self, enabled: bool = True, start_sound: str = "", stop_sound: str = ""
    ) -> None:
        self.enabled = bool(enabled)
        self._player = resolve_player() if self.enabled else None
        self._sounds = {
            "start": resolve_sound(start_sound),
            "stop": resolve_sound(stop_sound),
        }

    def play(self, event: str) -> None:
        """Play the cue for ``event`` ('start'|'stop'); never blocks or raises."""
        if not self.enabled:
            return
        sound = self._sounds.get(event)
        if self._player is None or sound is None:
            self._bell()
            return
        try:
            subprocess.Popen(
                [*self._player, str(sound)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:  # pragma: no cover - playback is best-effort
            self._bell()

    @staticmethod
    def _bell() -> None:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:  # pragma: no cover
            pass
