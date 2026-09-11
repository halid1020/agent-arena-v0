"""What a rollout drives: the arms on the bench, or the same arms in the twin.

A rollout loop needs remarkably little from a robot -- an observation to send,
somewhere to put an action, and a way to stop safely. Naming that little makes
the digital twin a drop-in for the rig, so a chunking strategy can be measured a
hundred times in simulation and then run on hardware through the same client,
the same wire and the same splice.

    ``observe() -> (state, {camera: HWC uint8})``
    ``command(action)``
    ``enable(first_action)``  -- authorise motion and reach the first goal
    ``release()``   -- take torque off, but keep everything else running
    ``shutdown()``
    ``cameras``  -- the names ``observe`` will produce
    ``frames``   -- ``get_rgb_image(name)``, the one thing the live view asks

Two more are OPTIONAL, and a caller must ask whether they are there rather than
assume: ``hold()``, which spends a tick on a rig whose clock only moves when
commanded, and ``reset(scenario)``, which puts a scene back for the next
attempt. Neither exists on the bench -- a bench tick passes whether anyone
commands it, and no button can tidy a real table -- so they are not in the
protocol, and the loop reaches them through ``getattr``.

A DEPLOYMENT needs three things a benchmark does not, so they are in the
protocol rather than in one implementation: ``enable``, which is where torque is
turned on and the arms are ramped to the first action; ``release``, which is
where it comes off again between attempts, leaving the cameras, the buses and
the page exactly where they were; and ``frames``, which is where the live view
reads its pictures. The twin answers all three -- with a ramp that takes the
same three seconds, a release that has nothing to release, and a cache of its
last render -- so a rehearsal in simulation walks the same path, in the same
order, as the run it rehearses.

The hardware implementation stays in ``tool/run_policy.py``, where the
buses, the torque and the confirmation prompt already live and where they
belong: this module deliberately imports nothing that can move a real motor.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np


class Rig(Protocol):
    """The robot, as much of it as a rollout loop needs to know.

    Only what every rig can answer is declared here. ``hold()`` and
    ``reset(scenario)`` are optional -- see the module docstring -- and a caller
    must look for them before it calls them.
    """

    cameras: "list[str]"

    def observe(self) -> "tuple[np.ndarray, dict]":
        """This tick's state vector and one frame per camera.

        The width is the rig's own -- twelve for a dual five-joint arm with a
        gripper each, seven for a single six-joint arm with one. Nothing in
        this module or in the rollout loop depends on which.
        """

    def command(self, action: np.ndarray) -> None:
        """Execute one action, in the units the dataset recorded."""

    def enable(self, first_action: np.ndarray) -> None:
        """Authorise motion and reach the first action. May enable torque."""

    def release(self) -> None:
        """End a trial: take torque off, and leave the run standing.

        Distinct from :meth:`shutdown` in what SURVIVES it. The cameras, the
        buses, the page and the host session all stay, because the operator is
        handling the rig between attempts and will very likely want another one.
        """

    def shutdown(self) -> None:
        """Stop safely. Called on every exit path, including a crash."""


#: Seconds spent reaching the policy's first action. Slow on purpose: the arms
#: may be a long way from wherever the policy decided to start.
RAMP_S = 3.0


def parse_camera_map(
    text: "str | None", default: "dict[str, str] | None" = None
) -> "dict[str, str]":
    """``scene=central,wrist_camera_left=left`` -> a rename map. Pure.

    A checkpoint asks for the camera names its dataset was collected under, and
    those are not always the names the rig produces now -- a stream gets renamed
    and every earlier checkpoint still wants the old one. Renaming on the way
    out is the whole fix; retraining is not.
    """
    if text is None:
        return dict(default or {})
    if text.strip() in ("", "none"):
        return {}
    mapping = {}
    for pair in text.split(","):
        if "=" not in pair:
            raise ValueError(f"--camera-map wants name=name pairs, got {pair!r}")
        old, new = pair.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


class FrameCache:
    """The last frame per camera, for a view that has no camera thread to read.

    On the bench the live view reads a ``DualDataManager`` the capture threads
    are already publishing into. The twin has no threads: it renders inside
    ``observe``, on the control thread. So the loop hands each tick's frames
    here, and the view reads them exactly as it reads the real thing --
    ``get_rgb_image(name)`` is the entire interface it uses.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: "dict[str, np.ndarray]" = {}

    def publish(self, images: "dict[str, np.ndarray]") -> None:
        with self._lock:
            self._frames.update(images)

    def get_rgb_image(self, name: str) -> "np.ndarray | None":
        with self._lock:
            return self._frames.get(name)

    def request_shutdown(self) -> None:
        """Nothing runs in the background here; the name matches the bench."""


class TwinRig:
    """The MuJoCo twin behind the same three methods as the bench.

    One ``command`` is exactly one control period, ``1/fps``, of simulated time,
    so a tick here and a recorded frame mean the same thing; how many physics
    substeps that takes is the environment's business, not this class's.
    Nothing reads the wall clock: the world advances only when ``command`` is
    called, which is what lets a rollout inject a chosen inference delay and get
    the same answer every time.

    Rendering happens on whichever thread calls ``observe``, and must therefore
    be the control thread only -- ``mujoco.Renderer`` is not thread-safe, and
    the remote client fetches its chunks on a background thread.
    """

    def __init__(
        self,
        env,
        cameras: "list[str]",
        camera_wh: "tuple[int, int]" = (640, 480),
        camera_map: "dict[str, str] | None" = None,
        fps: float = 30.0,
        decode_action=None,
    ) -> None:
        if decode_action is None:
            raise TypeError(
                "TwinRig needs decode_action: the function that turns one "
                "action vector into whatever the twin's env.tick wants. That "
                "mapping is the rig's, not this package's -- it is where the "
                "gripper channel and the joint units are decided."
            )

        self.env = env
        self.camera_wh = (int(camera_wh[0]), int(camera_wh[1]))
        #: The control rate the caller is ticking at. Only the ramp reads it --
        #: everything else here counts ticks, not seconds -- but a ramp measured
        #: in the wrong rate lasts the wrong number of seconds.
        self.fps = float(fps)
        self._decode = decode_action
        #: Applied to the names this rig REPORTS, so a caller that asks what
        #: cameras exist is told what the policy will be sent. The client does
        #: the same rename on the frames themselves.
        self.camera_map = dict(camera_map or {})
        self.cameras = sorted(self.camera_map.get(name, name) for name in cameras)
        self.ticks = 0
        self.last_action: "np.ndarray | None" = None
        #: What the live view reads. Filled by whoever calls :meth:`observe`.
        self.frames = FrameCache()

    def observe(self) -> "tuple[np.ndarray, dict]":
        state, images = self.env.observe(self.camera_wh)
        self.frames.publish(self.rename(images))
        return state, images

    def rename(self, images: "dict[str, np.ndarray]") -> "dict[str, np.ndarray]":
        """The frames under the names :attr:`cameras` reports. Pure."""
        return {self.camera_map.get(k, k): v for k, v in images.items()}

    def command(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=float)
        q_rad, grip = self._decode(action)
        self.env.tick(q_rad, grip)
        self.last_action = action
        self.ticks += 1

    def hold(self) -> None:
        """Advance time with no new goal -- the servos keep their last target.

        The twin's actuators hold their commanded position when nothing new is
        written, so a starved queue costs the same here as it does on the bench:
        the arms stop where they are, and the world keeps moving.
        """
        if self.last_action is None:
            state, _ = self.env.observe(self.camera_wh)
            self.last_action = np.asarray(state, dtype=float)
        self.command(self.last_action)

    def enable(self, first_action: np.ndarray) -> None:
        """Reach the first action over :data:`RAMP_S`, as the bench does.

        There is no torque to enable, but there IS a ramp: the twin starts at
        its neutral pose and the policy's first action may be nowhere near it,
        and a rehearsal that teleported there would hide the one part of a real
        rollout most likely to surprise somebody.
        """
        target = np.asarray(first_action, dtype=float)
        state, _ = self.env.observe(self.camera_wh)
        start = np.asarray(state, dtype=float)
        steps = max(1, int(RAMP_S * self.fps))
        for i in range(1, steps + 1):
            self.command(start + (target - start) * (i / steps))

    def reset(self, scenario) -> None:
        """Put the scene back for another attempt, and forget the last one.

        Clearing :attr:`last_action` is the part that is easy to miss: ``hold``
        re-commands it, so a rollout that reset and then held would drive the
        arms to a pose chosen for the episode that has just ended, into a scene
        that no longer contains what it was reaching for. Zeroing the tick count
        keeps the twin's clock and the new attempt's log agreeing about when it
        began.
        """
        self.env.reset(scenario)
        self.last_action = None
        self.ticks = 0

    def release(self) -> None:
        """Nothing to let go of: there is no torque in a twin.

        Deliberately NOT a reset. The simulated arms keep the pose the trial
        left them in, and :attr:`last_action` with it, so a rehearsal of "stop,
        then look at where it ended up" shows what the bench would show. Only
        :meth:`reset` forgets an attempt.
        """

    def shutdown(self) -> None:
        """Nothing to release: no torque, no bus, no camera thread."""
