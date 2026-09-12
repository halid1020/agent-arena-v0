"""The wire a rig's monitor speaks, and the keys a remote caller may press.

One process opens the devices and another looks at them: the collection session
and the console, or a rig agent and the shared console. This is what passes
between them -- MJPEG framing, JPEG encoding, and the per-mode allow-list saying
which keystrokes a remote page is permitted to forward.

None of it knows what a robot is. The JOINT half of the old module -- rows of
measured-against-last-sent per limb -- stayed in the rig that has those limbs,
because a row is shaped by a schema and a server is not.

THE ALLOW-LIST IS A SAFETY BOUNDARY, not a convenience. A page can forward the
episode and quit keys, and ENABLE for a leader session whose keys are otherwise
read from a terminal a console-started session does not have. Park and home stay
physical in both modes: they move the arms a long way, and the person who should
decide that is the one standing next to them.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

# The live view is a monitor, not a recording: a low rate and a small frame keep
# it far below the cost of the capture threads it watches, and JPEG at this
# quality is indistinguishable at tile size.
DEFAULT_VIEW_FPS = 10.0
DEFAULT_QUALITY = 70
DEFAULT_MAX_WIDTH = 480

BOUNDARY = "frame"

# Only these reach the session's button callbacks. Both are decisions the
# operator can safely make from another room: start/stop this episode, and end
# the session. Everything that moves an arm is deliberately absent.
DEFAULT_ALLOWED_KEYS = frozenset({"a", "q"})

# A leader session is driven from a KEYBOARD, not from a headset -- and when the
# console starts it, that keyboard is the terminal the console itself was
# launched in, which is not where the operator is standing. So enabling the arms
# is added: without it a session started from the browser cannot be driven at
# all. Home and park stay physical, on the terminal or the desktop window.
LEADER_ALLOWED_KEYS = frozenset({"y", "a", "q"})


def mjpeg_part(jpeg: bytes, boundary: str = BOUNDARY) -> bytes:
    """One ``multipart/x-mixed-replace`` part. Pure — unit-tested."""
    return (
        (
            f"--{boundary}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        + jpeg
        + b"\r\n"
    )


def allowed_keys_for(input_mode: str) -> "frozenset[str]":
    """Which control keys a watcher may press, given how the rig is driven. Pure.

    The rule is the same in both modes -- a key that moves an arm belongs where
    the operator can see the arm -- and it lands differently only because the
    two modes put the operator in different places. With a headset on, every
    button is already to hand. With leader arms, the alternative surface is a
    keyboard the console-started session does not have.
    """
    return LEADER_ALLOWED_KEYS if input_mode == "leader" else DEFAULT_ALLOWED_KEYS


def key_refusal(key: str, allowed: "Iterable[str]", known: "Iterable[str]") -> str:
    """Why this key may not be pressed remotely, or "". Pure — unit-tested."""
    allowed = set(allowed)
    known = set(known)
    if not key:
        return "no key given"
    if key not in known:
        return f"no such control key {key!r}"
    if key not in allowed:
        return (
            f"{key!r} is not remotely controllable: it moves the arms, so it "
            "stays on the headset and the session keyboard"
        )
    return ""


def encode_frame_batch(
    frames: "dict[str, Any]",
    quality: int = DEFAULT_QUALITY,
    max_width: int = DEFAULT_MAX_WIDTH,
    encoder: "Callable | None" = None,
) -> "dict[str, str]":
    """Named frames as base64 JPEGs, for one batched response. Unit-tested.

    Why a batch exists at all: a ``multipart/x-mixed-replace`` stream never
    completes, so a page that gives each camera its own stream spends one of the
    browser's ~6 connections per origin on each tile, for as long as the tile is
    on screen. Add the page's own pollers and only four tiles ever load -- the
    rest queue behind the limit and stay black, however healthy the cameras are.
    One response carrying every frame costs one connection whatever the camera
    count, which is the only property that scales.

    A stream with nothing published yet is LEFT OUT rather than sent as null, so
    the viewer keeps whatever that tile last showed. A camera between frames
    should not make its tile flicker.
    """
    import base64

    enc = encoder or encode_jpeg
    out: "dict[str, str]" = {}
    for name, rgb in frames.items():
        if rgb is None:
            continue
        jpeg = enc(rgb, quality, max_width)
        if jpeg is not None:
            out[name] = base64.b64encode(jpeg).decode("ascii")
    return out


def encode_jpeg(
    rgb, quality: int = DEFAULT_QUALITY, max_width: int = DEFAULT_MAX_WIDTH
):
    """RGB array → JPEG bytes, downscaled to ``max_width``. ``None`` if it cannot."""
    import cv2  # type: ignore[import]

    if rgb is None:
        return None
    frame = rgb
    if max_width and frame.shape[1] > max_width:
        scale = max_width / float(frame.shape[1])
        frame = cv2.resize(
            frame, (max_width, max(1, int(round(frame.shape[0] * scale))))
        )
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
