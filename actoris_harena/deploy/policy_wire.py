"""The observation/action-chunk wire format shared by the rig and a policy host.

On-robot inference does not have to happen on the robot's computer. The rig
captures the cameras and drives the followers; a machine with a real GPU can run
the forward pass and hand back actions. What makes that cheap is the shape of the
policies themselves: an action-chunking policy consumes an observation only when
it re-plans and then emits many actions at once -- one hundred for the ACT
configuration used here, thirty-two for diffusion -- so a link carries one
observation window per chunk rather than one per control tick.

The format is therefore built around a *window*: the last ``n_obs_steps``
observations, oldest first, because that is the order the policy's own
observation queue expects. ACT asks for one; diffusion asks for two consecutive
frames and would otherwise be fed a pair captured a whole chunk apart, which
never happens during training.

A message is a four-byte big-endian header length, a JSON header, then the
binary payloads back to back. JSON alone would mean base64 for the images (a
third more bytes and a needless copy); a pure binary format would mean a schema
nobody can read while debugging. Images travel as JPEG because raw frames at the
recorded resolution are 2.7 MB per observation, where the three rig cameras
compress to about seventy kilobytes together -- with no effect a policy trained
on lossy video can notice.

Encoding and decoding live together here, and nothing in this module imports
aiohttp or torch, so the format is unit-testable without a GPU, a network or a
robot -- and the client and the server cannot drift apart, because neither owns
a private copy of it.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

_LEN = struct.Struct(">I")

#: Default JPEG quality. High enough that the compression is invisible next to
#: the AV1 the dataset itself stores; measured at ~72 kB for one observation of
#: this rig's three cameras, so even a two-step window is a small message.
DEFAULT_QUALITY = 90

#: Channels in a policy action / state vector, when a caller wants the width
#: pinned. There is no rig-wide default here on purpose: this package serves
#: more than one robot, and the dual SO-101's 12 is not a single UR3e's 7. A
#: caller that knows its own width passes ``action_dim=`` and gets the strong
#: check; a caller that does not still gets the checks that matter, because a
#: message must be self-consistent -- every step the same width, every step the
#: same cameras, and the chunk's declared ``dim`` matching the bytes that follow.


class WireError(ValueError):
    """A message could not be decoded, or does not describe what it claims."""


def _encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """One RGB frame -> JPEG bytes.

    OpenCV writes JPEGs from channel-order BGR, so the array is flipped on the
    way in and back on the way out. Round-tripping RGB as though it were BGR
    would also work -- the swap is its own inverse -- but the intermediate file
    would have its red and blue exchanged, and a wire format whose payloads open
    wrongly in an image viewer is a bad thing to debug through.
    """
    import cv2

    ok, buf = cv2.imencode(
        ".jpg",
        np.ascontiguousarray(rgb[:, :, ::-1]),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise WireError("JPEG encoding failed")
    return buf.tobytes()


def _decode_jpeg(payload: bytes) -> np.ndarray:
    """JPEG bytes -> one RGB uint8 frame."""
    import cv2

    arr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise WireError("JPEG decoding failed")
    return np.ascontiguousarray(arr[:, :, ::-1])


def _pack(header: "dict[str, Any]", payloads: "list[bytes]") -> bytes:
    blob = json.dumps(header).encode("utf-8")
    return b"".join([_LEN.pack(len(blob)), blob, *payloads])


def _unpack(message: bytes) -> "tuple[dict[str, Any], memoryview]":
    if len(message) < _LEN.size:
        raise WireError("message is too short to hold a header length")
    (n,) = _LEN.unpack_from(message, 0)
    start = _LEN.size + n
    if len(message) < start:
        raise WireError("message is truncated inside its header")
    try:
        header = json.loads(bytes(message[_LEN.size : start]))
    except ValueError as exc:
        raise WireError(f"header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise WireError("header is not an object")
    return header, memoryview(message)[start:]


def encode_request(
    steps: "list[tuple[np.ndarray, dict[str, np.ndarray]]]",
    task: str,
    *,
    session: str = "",
    seq: int = 0,
    actions: "int | None" = None,
    quality: int = DEFAULT_QUALITY,
    action_dim: "int | None" = None,
) -> bytes:
    """An observation window -> one request message.

    ``steps`` is ``[(state, {camera: rgb_frame}), ...]``, oldest first. Every
    step must carry the same cameras: a window whose steps disagree cannot be
    stacked along time, and finding that out on the GPU host -- after the arms
    are already under torque -- is far worse than finding it out here.

    ``actions`` asks the host for at most that many actions from the chunk;
    ``None`` means "whatever the policy plans for".
    """
    if not steps:
        raise WireError("an observation window needs at least one step")
    cameras = sorted(steps[0][1])
    if not cameras:
        raise WireError("an observation step needs at least one camera")

    payloads: "list[bytes]" = []
    sizes: "list[int]" = []
    states: "list[list[float]]" = []
    shape: "list[int]" = []
    width = action_dim
    for state, images in steps:
        vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise WireError("a state vector cannot be empty")
        if width is None:
            width = int(vec.size)
        if vec.size != width:
            raise WireError(f"state must have {width} channels, got {vec.size}")
        if sorted(images) != cameras:
            raise WireError(
                f"every step must carry the same cameras: {sorted(images)} != {cameras}"
            )
        states.append([float(v) for v in vec])
        for name in cameras:
            frame = np.asarray(images[name])
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise WireError(
                    f"camera '{name}' frame must be HxWx3, got {frame.shape}"
                )
            if frame.dtype != np.uint8:
                raise WireError(
                    f"camera '{name}' frame must be uint8, got {frame.dtype}"
                )
            if not shape:
                shape = [int(frame.shape[0]), int(frame.shape[1]), 3]
            elif list(frame.shape) != shape:
                raise WireError(
                    f"camera '{name}' frame is {list(frame.shape)}, expected {shape}"
                )
            blob = _encode_jpeg(frame, quality)
            payloads.append(blob)
            sizes.append(len(blob))

    header = {
        "kind": "request",
        "session": session,
        "seq": int(seq),
        "task": task,
        "cameras": cameras,
        "shape": shape,
        "states": states,
        "sizes": sizes,
        "actions": None if actions is None else int(actions),
    }
    return _pack(header, payloads)


def decode_request(message: bytes) -> "dict[str, Any]":
    """A request message -> ``{"steps": [(state, {camera: rgb}), ...], ...}``.

    Steps come back oldest first, exactly as they were sent.
    """
    header, body = _unpack(message)
    if header.get("kind") != "request":
        raise WireError(f"expected a request, got kind={header.get('kind')!r}")
    cameras = list(header.get("cameras") or [])
    states = list(header.get("states") or [])
    sizes = list(header.get("sizes") or [])
    if not cameras or not states:
        raise WireError("request carries no cameras or no states")
    if len(sizes) != len(states) * len(cameras):
        raise WireError(
            f"request declares {len(sizes)} images for "
            f"{len(states)} steps x {len(cameras)} cameras"
        )

    steps: "list[tuple[np.ndarray, dict[str, np.ndarray]]]" = []
    at = 0
    k = 0
    for state in states:
        images: "dict[str, np.ndarray]" = {}
        for name in cameras:
            size = int(sizes[k])
            if at + size > len(body):
                raise WireError("request is truncated inside its image payloads")
            images[name] = _decode_jpeg(bytes(body[at : at + size]))
            at += size
            k += 1
        steps.append((np.asarray(state, dtype=np.float32), images))

    return {
        "steps": steps,
        "task": header.get("task", ""),
        "session": header.get("session", ""),
        "seq": int(header.get("seq", 0)),
        "actions": header.get("actions"),
        "cameras": cameras,
    }


def encode_chunk(
    actions: np.ndarray,
    *,
    seq: int = 0,
    timings: "dict[str, float] | None" = None,
    action_dim: "int | None" = None,
) -> bytes:
    """A ``(K, channels)`` action chunk -> one response message.

    Float32 little-endian, so a hundred actions are under five kilobytes and the
    reply never becomes the slow half of the round trip.
    """
    arr = np.asarray(actions, dtype="<f4")
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise WireError(f"chunk must be (K, channels), got {arr.shape}")
    if action_dim is not None and arr.shape[1] != action_dim:
        raise WireError(f"chunk must be (K, {action_dim}), got {arr.shape}")
    header = {
        "kind": "chunk",
        "seq": int(seq),
        "count": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "timings": dict(timings or {}),
    }
    return _pack(header, [np.ascontiguousarray(arr).tobytes()])


def decode_chunk(message: bytes) -> "tuple[np.ndarray, dict[str, Any]]":
    """A response message -> ``(actions (K, channels) float32, header)``."""
    header, body = _unpack(message)
    if header.get("kind") != "chunk":
        raise WireError(f"expected a chunk, got kind={header.get('kind')!r}")
    count = int(header.get("count", 0))
    dim = int(header.get("dim", 0))
    want = count * dim * 4
    if len(body) < want:
        raise WireError(f"chunk is truncated: {len(body)} bytes for {want} expected")
    arr = np.frombuffer(bytes(body[:want]), dtype="<f4").reshape(count, dim)
    return np.array(arr, dtype=np.float32), header


class ObservationWindow:
    """The last ``n_obs_steps`` observations, oldest first.

    A policy that consumes several observation steps expects them *adjacent in
    time*: diffusion was trained on frames one control tick apart, so handing it
    the frame it last re-planned on plus the current one -- a chunk apart --
    would present it with a history it never saw. The rig captures at the control
    rate whether or not it is sending, so it always has adjacent frames to give;
    this keeps them.

    The camera set is fixed at construction from the host's handshake, so a rig
    configured with a different camera set is rejected on the first frame rather
    than by a shape error inside the policy.
    """

    def __init__(self, cameras: "list[str]", n_obs_steps: int) -> None:
        if n_obs_steps < 1:
            raise WireError(f"n_obs_steps must be >= 1, got {n_obs_steps}")
        self.cameras = sorted(cameras)
        self.n_obs_steps = int(n_obs_steps)
        self._steps: "list[tuple[np.ndarray, dict[str, np.ndarray]]]" = []

    def push(self, state: np.ndarray, images: "dict[str, np.ndarray]") -> None:
        """Add the newest observation, dropping the oldest once full."""
        if sorted(images) != self.cameras:
            raise WireError(
                f"observation carries cameras {sorted(images)}, "
                f"but the policy expects {self.cameras}"
            )
        self._steps.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                {k: v for k, v in images.items()},
            )
        )
        del self._steps[: max(0, len(self._steps) - self.n_obs_steps)]

    @property
    def ready(self) -> bool:
        """True once a full window has been seen."""
        return len(self._steps) >= self.n_obs_steps

    def steps(self) -> "list[tuple[np.ndarray, dict[str, np.ndarray]]]":
        """The window, oldest first.

        Before the first ``n_obs_steps`` frames have arrived the oldest is
        repeated to fill it -- the same thing the policy's own queue does on its
        first call, so a run does not have to stall waiting for history.
        """
        if not self._steps:
            raise WireError("the observation window is empty")
        pad = [self._steps[0]] * (self.n_obs_steps - len(self._steps))
        return [*pad, *self._steps]
