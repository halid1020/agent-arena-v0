"""What a capture thread needs from whatever it publishes into.

The camera threads used to annotate their argument as ``DualDataManager`` -- a
dual-arm blackboard carrying IK targets, controller state, leader mappings and
last-sent commands, none of which a camera has any use for. Three methods is the
whole of what they actually call, so three methods is the contract.

Any object with these satisfies it structurally; nothing has to inherit. The
dual SO-101's ``DualDataManager`` already does, and a single-arm rig writes
whatever suits it -- or uses :class:`FrameStore`.
"""

import threading
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FramePublisher(Protocol):
    """The publishing side of a frame store, as a capture thread uses it.

    Runtime-checkable so a rig can assert its own store fits before a session
    rather than finding out mid-episode. Note what that check does and does not
    cover: ``isinstance`` against a Protocol tests that the METHODS EXIST, not
    that their signatures match, so it catches a forgotten method and not a
    wrong keyword.
    """

    def set_rgb_image(
        self, rgb: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        """Publish one RGB frame under ``name``, stamped when it was READ.

        ``t_capture`` is a ``time.monotonic()`` reading taken as close to the
        device read as the driver allows. It is what every stream is later
        aligned through, so a frame stamped on arrival instead of on read moves
        the whole episode's alignment by the queueing delay.
        """

    def set_depth_image(
        self, depth16: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        """Publish one 16-bit depth frame under ``name``, same stamping rule."""

    def is_shutdown_requested(self) -> bool:
        """True once the session is ending, so a capture loop can stop."""


class FrameStore:
    """The last frame per name, with the stamp it was read at.

    A minimal :class:`FramePublisher` for a rig that has no blackboard of its
    own. Deliberately plain: one lock, two dicts, and no history -- anything
    that needs to look a stream up by time uses ``actoris_harena.sync``, which
    is where that reasoning belongs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb: "dict[str, tuple[np.ndarray, float | None]]" = {}
        self._depth: "dict[str, tuple[np.ndarray, float | None]]" = {}
        self._shutdown = threading.Event()

    def set_rgb_image(
        self, rgb: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        with self._lock:
            self._rgb[name] = (rgb, t_capture)

    def set_depth_image(
        self, depth16: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        with self._lock:
            self._depth[name] = (depth16, t_capture)

    def get_rgb_image(self, name: str) -> "np.ndarray | None":
        with self._lock:
            found = self._rgb.get(name)
        return None if found is None else found[0]

    def get_depth_image(self, name: str) -> "np.ndarray | None":
        with self._lock:
            found = self._depth.get(name)
        return None if found is None else found[0]

    def rgb_camera_names(self) -> "list[str]":
        with self._lock:
            return sorted(self._rgb)

    def is_shutdown_requested(self) -> bool:
        return self._shutdown.is_set()

    def request_shutdown(self) -> None:
        self._shutdown.set()
