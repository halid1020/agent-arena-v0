"""Generic UVC (OpenCV) camera capture thread for data collection.

Generalised from the retired single-camera thread (git 185df55^:
src/common/threads/camera_usb.py): one :class:`CameraCapture` per stream, each
opening its own ``/dev/videoN`` device, forcing the configured resolution
(warn-once resize if the device negotiates a different size), converting BGR to
RGB, optionally rotating 180 deg, and publishing into the shared
a ``FramePublisher`` via ``set_rgb_image(rgb, name)``. On device loss the loop
retries the open every 2 s so a bumped USB cable does not kill the session.

Freshness is a design constraint, not an accident: every stream is opened with a
compressed pixel format (``fourcc``, MJPG by default — several uncompressed
640x480@30 streams do not fit through the shared USB controllers) and with the
shortest driver queue, and the loop is paced by the blocking read alone so no
queue of stale frames can build up. See :meth:`_configure`.

The recorder polls :meth:`seconds_since_last_frame` to detect stale streams.
"""

from __future__ import annotations

import threading
import time
import traceback

import cv2  # type: ignore[import]
import numpy as np

from actoris_harena.recording.camera_controls import apply_controls
from actoris_harena.recording.frames import FramePublisher

# How long to wait before each successive FAILED attempt to reopen a device. A
# device that drops off the USB bus is usually back within a few hundred
# milliseconds, and every millisecond spent waiting is a millisecond of the
# episode that is not being recorded -- so the FIRST retry is immediate. What
# follows it has to back off, though: a stream that was refused USB bandwidth
# fails instantly and for ever, and retrying that at full speed hammers the very
# bus the other cameras are sharing. The last value is the ceiling.
_REOPEN_BACKOFF_S = (0.0, 0.5, 1.0, 2.0, 4.0, 5.0)

# Consecutive failures after which a stream that has NEVER delivered a frame is
# given up as starved. A camera that has worked at least once is a different
# case -- that is a real unplug, and it is retried for as long as the session
# lasts, because the operator can plug it back in.
_STARVE_GIVE_UP_ATTEMPTS = 8

# How long open() waits for the device to actually produce a frame before
# calling the open a failure. Generous: a healthy camera answers on the first
# read (tens of milliseconds), so this budget is only ever spent on a bad one.
_WARM_UP_S = 1.5
_WARM_UP_POLL_S = 0.02

# How long stop() waits for the capture thread to leave its blocking read.
_STOP_JOIN_S = 3.0

# Minimum time one capture iteration may take. A blocking V4L2 read paces the
# loop by itself; this only stops a device that returns frames instantly from
# spinning a core.
_BUSY_SPIN_S = 0.001

# V4L2 buffers per stream. The smallest value that still sustains the device's
# full frame rate (1 halves it — see _configure).
_CAPTURE_BUFFERS = 2


def reopen_delay(consecutive_failures: int) -> float:
    """Seconds to wait before the next reopen attempt. Pure -- unit-tested.

    ``consecutive_failures`` counts cycles since the last delivered frame, so 1
    is the first retry and gets no delay at all; the schedule then climbs to its
    ceiling. See ``_REOPEN_BACKOFF_S`` for why the first one is free.
    """
    if consecutive_failures <= 0:
        return 0.0
    return _REOPEN_BACKOFF_S[min(consecutive_failures - 1, len(_REOPEN_BACKOFF_S) - 1)]


class CameraCapture:
    """Capture thread for a single named UVC camera stream."""

    def __init__(
        self,
        name: str,
        device: "int | str",
        width: int,
        height: int,
        fps: int,
        rotate180: bool,
        fourcc: str = "",
        controls: "dict[str, float | None] | None" = None,
    ) -> None:
        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate180 = rotate180
        self.fourcc = fourcc
        # Per-camera image controls; None values are left to the camera. See
        # actoris_harena.recording.camera_controls -- exposure among them caps frame rate.
        self.controls = dict(controls or {})

        # How many times this device stopped delivering and had to be reopened.
        # Read at the end of the session by actoris_harena.recording.fault_report, which
        # weighs it against the other devices' counts: several failures on one hub
        # mean the hub, one device's failures mean that device.
        self.disconnects = 0

        # Set once this stream has been given up on: it opened but never once
        # delivered a frame, which on this rig means the USB bus refused it the
        # isochronous bandwidth. The monitor and the console show it as starved
        # rather than merely absent, because the two want different fixes -- an
        # absent camera wants replugging, a starved one wants fewer cameras.
        self.starved = False

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_frame_mono: float | None = None
        self._warned_resize = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _configure(self, cap: "cv2.VideoCapture") -> None:
        # Pixel format first: switching FOURCC can reset the mode.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Short driver queue, to bound how stale a delivered frame can be if this
        # thread is momentarily late. Two is deliberate and MEASURED: a single
        # buffer starves the driver (it has nowhere to put the next frame while
        # we hold the only one, so it drops every other frame and the stream
        # halves to ~15 fps), while 2 and above all sustain the device's full
        # rate. Do not "optimise" this to 1.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, _CAPTURE_BUFFERS)
        # Image controls last, since some cameras reset them when the format or
        # size changes. Exposure among them caps the frame rate, so this is not
        # only about how the picture looks -- see camera_controls.
        apply_controls(cap, self.controls)

    def _delivers(self, cap: "cv2.VideoCapture") -> bool:
        """Whether the device actually produces a frame, not merely opens.

        A camera denied its share of the USB bus's isochronous bandwidth opens
        perfectly and then returns nothing, for ever -- and on this rig that is
        the common failure, not an exotic one (see the USB budget note in
        src/conf/recording.yaml). Without this check such a stream looks healthy
        at startup and only reveals itself once recording has begun, as an
        endless reopen loop. Failing here instead makes it a plain open failure,
        which every caller already handles: the recorder refuses to start and
        names the camera, the console's preview skips it and shows the rest.
        """
        deadline = time.monotonic() + _WARM_UP_S
        while time.monotonic() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                return True
            time.sleep(_WARM_UP_POLL_S)
        return False

    def open(self) -> bool:
        """Attempt to open the device once. Returns whether it opened.

        Used at startup for a fail-fast check BEFORE the dataset is created. An
        open that cannot deliver frames counts as a failure -- see _delivers.
        """
        # Pin the V4L2 backend: the default fallback chain can silently
        # open a DIFFERENT physical camera when given a metadata node or a
        # /dev/v4l/by-path alias (verified with the tactile cameras).
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        self._configure(cap)
        if not self._delivers(cap):
            cap.release()
            return False
        self._cap = cap
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"  📷 camera '{self.name}' opened on device {self.device}: "
            f"{actual_w}x{actual_h} (forced to {self.width}x{self.height})"
        )
        return True

    def start(self, data_manager: FramePublisher) -> None:
        """Spawn the capture thread (opens the device if not already open)."""
        self._thread = threading.Thread(
            target=self._loop, args=(data_manager,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread and let it release its own handle.

        The handle is released by whoever owns it and by nobody else. A starved
        or wedged thread can still be inside a blocking V4L2 read when the join
        times out, and releasing a VideoCapture from here while that read is in
        flight aborts the whole process -- which is what turned Ctrl+C on the
        console into a core dump. The thread is a daemon, so leaving a wedged
        one to the interpreter's exit costs nothing.
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            # No capture thread ever ran: open() was used on its own as a
            # readiness check, so this side owns the handle after all.
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            return
        thread.join(timeout=_STOP_JOIN_S)
        if thread.is_alive():
            print(
                f"⚠️  camera '{self.name}' did not stop within {_STOP_JOIN_S:.0f}s "
                "— leaving its handle to the capture thread"
            )
            return
        self._cap = None

    # ── Freshness ────────────────────────────────────────────────────────────

    def seconds_since_last_frame(self, now_mono: float) -> float | None:
        """Age (seconds) of the most recent published frame, or None if never."""
        with self._lock:
            if self._last_frame_mono is None:
                return None
            return now_mono - self._last_frame_mono

    # ── Capture loop ─────────────────────────────────────────────────────────

    def _force_size(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            if not self._warned_resize:
                print(
                    f"⚠️  camera '{self.name}' delivered "
                    f"{rgb.shape[1]}x{rgb.shape[0]}, resizing to "
                    f"{self.width}x{self.height} (warned once)"
                )
                self._warned_resize = True
            rgb = cv2.resize(rgb, (self.width, self.height))
        return rgb

    def _give_up_if_starved(self, failures: int) -> None:
        """Stop retrying a stream that has never once delivered a frame.

        Only a stream that has NEVER worked is given up on. One that delivered
        frames and then stopped is a cable someone can plug back in, and it is
        retried (with backoff) for the whole session, which is the behaviour a
        bumped USB lead needs.
        """
        if self.starved or failures < _STARVE_GIVE_UP_ATTEMPTS:
            return
        with self._lock:
            if self._last_frame_mono is not None:
                return
        self.starved = True
        print(
            f"❌ camera '{self.name}' ({self.device}) opened but never delivered a "
            f"frame in {failures} attempts — giving up on it for this session.\n"
            "   The usual cause is the USB bandwidth budget, not the camera: "
            "these are USB 2.0 devices and one 480 Mbit/s bus carries only about "
            "three 640x480 MJPG streams.\n"
            "   Record fewer cameras at once, or try the uvcvideo FIX_BANDWIDTH "
            "quirk — see the USB BUDGET note in src/conf/recording.yaml."
        )

    def _loop(self, data_manager: FramePublisher) -> None:
        dt = 1.0 / float(self.fps)
        # Cycles since the last delivered frame. Drives the reopen backoff, and
        # is reset by a frame arriving -- never by an open() that merely
        # succeeded, so a device that flaps open/closed still backs off.
        failures = 0
        next_attempt = 0.0
        try:
            while not self._stop.is_set() and not data_manager.is_shutdown_requested():
                iteration_start = time.time()

                if self.starved:
                    # Given up on. Idle until stop() rather than keep hammering a
                    # bus that has already refused this stream many times over --
                    # the other cameras are sharing it.
                    time.sleep(dt)
                    continue

                if self._cap is None:
                    now = time.monotonic()
                    if now < next_attempt:
                        time.sleep(min(dt, next_attempt - now))
                        continue
                    if self.open():
                        print(f"  ✓ camera '{self.name}' reopened")
                    else:
                        failures += 1
                        next_attempt = time.monotonic() + reopen_delay(failures)
                        self._give_up_if_starved(failures)
                    continue

                ret, frame = self._cap.read()
                # Stamp the capture instant immediately, before colour convert /
                # resize, so downstream alignment reflects when the sensor was
                # actually read rather than when the processed frame is published.
                t_capture = time.monotonic()
                if not ret or frame is None:
                    self.disconnects += 1
                    failures += 1
                    print(
                        f"⚠️  camera '{self.name}' read failed "
                        f"({self.disconnects} so far); reopening"
                    )
                    self._cap.release()
                    self._cap = None
                    next_attempt = time.monotonic() + reopen_delay(failures)
                    self._give_up_if_starved(failures)
                    continue

                failures = 0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._force_size(rgb)
                if self.rotate180:
                    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
                data_manager.set_rgb_image(rgb, self.name, t_capture=t_capture)
                with self._lock:
                    self._last_frame_mono = t_capture

                # No fps sleep here: the V4L2 ``read`` above already blocks until
                # the next frame, so it paces this loop at the device rate. An
                # extra sleep on top would hold the loop below that rate and let
                # the driver queue build up — the very drift this avoids. Only
                # guard against a device that returns instantly (busy-spin).
                if time.time() - iteration_start < _BUSY_SPIN_S:
                    time.sleep(_BUSY_SPIN_S)
        except Exception as e:  # pragma: no cover - hardware failure path
            print(f"❌ camera '{self.name}' thread error: {e}")
            traceback.print_exc()
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            print(f"📷 camera '{self.name}' thread stopped")
