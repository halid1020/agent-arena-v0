"""What a rollout is doing, and the one switch that gates its motion.

``tool/run_policy.py`` drives the arms from a policy's action chunks. Two
parties need to agree about that run: the control loop, which writes goals at
the control rate, and a watcher (``common.web.policy_view``), which reads what
is happening and may ask for less motion. This module is what they share, and
it is deliberately pure -- no camera, no bus, no HTTP -- so the arithmetic can
be tested without either of them.

THE THROTTLE. A rollout is in one of three modes:

  * ``run``     -- serve an action every tick, which is what an evaluation does.
  * ``hold``    -- serve nothing; the servos keep their last commanded goal.
  * ``preview`` -- serve nothing, but KEEP what is queued, so the plan on the
    screen is the plan that will execute when it is asked for.
  * ``step``    -- serve exactly one chunk, then fall back to where it came from.

``step`` is how a failure is examined: one plan is executed, the arms stop, and
the plan that produced the motion is still on the screen beside it.

``preview`` and ``step`` together are how a plan is inspected BEFORE it moves
anything: watch the queued chunk in the twin, then execute that same chunk on
the arms, then return to ``preview`` for the next one. That cycle only works
because a step taken from ``preview`` returns to ``preview``, and because
previewing does not throw the plan away.

Leaving ``hold`` DROPS whatever is queued (``queue_stale``), because a chunk was
planned from an observation taken before the pause -- possibly minutes before --
and executing it afterwards would drive the arms from a stale picture of the
world. The cost is one round trip on resume; the alternative is a surprise.
Nothing else drops it: a plan examined in ``preview`` is worth exactly as much a
moment later, and dropping it there would make the inspection meaningless.

ARMING. By default the modes only ever gate motion the operator already
authorised at the terminal, where torque is enabled after a confirmation, and
nothing here can enable torque.

A run may instead DELEGATE that consent to the view (``--arm-from-view``), and
then :meth:`RunControl.arm` is what the terminal prompt was: the question asked
once, before any torque. Consent lasts as long as the trial does. Ending one
takes the arms off torque, so :meth:`RunControl.disarm` withdraws it and the
next trial has to ask again -- a rig somebody has just walked up to and moved by
hand is exactly the rig whose next motion deserves a second look.

Delegating it is a real change in who can start the arms: the view binds
loopback and is unauthenticated, so anyone who can reach that port can begin the
motion. That is why it is a flag and not the default.

ENDING A TRIAL. A rollout is usually attempted more than once, and between the
attempts somebody has to handle the rig -- put the cube back, straighten a
garment, take hold of an arm that has folded itself into a corner. None of that
is possible while the followers are stiff, so both requests that end a trial
RELEASE them, and the loop that reads the request is what takes torque off:

  * ``stop``  -- this trial is over, and no other is intended. The arms go free
    and the run stays alive: the page, the run log, the host session and the
    cameras are all still there, because the operator may well change their
    mind. Nothing here ends the process; Ctrl+C at the terminal does that.
  * ``reset`` -- this trial is over and the NEXT one begins. The same release,
    plus the scene goes back where the scene is a data structure, and the log
    starts counting a new trial.

Both are one-shot flags the loop picks up, exactly as it picks up
``queue_stale``, and both drop the throttle to ``hold``: an arm that has just
been let go must not be sitting in ``run``, waiting to serve the instant torque
comes back.
"""

from __future__ import annotations

import math
import threading
import time

#: What a rollout may be doing. ``stop`` and ``reset`` are requests, not modes:
#: they end a trial and release the arms, the second one beginning another --
#: see ``ENDING A TRIAL`` above. ``arm`` is not a mode either: it is the consent
#: that lets torque be enabled at all, and only a run started with that consent
#: DELEGATED to the view will wait for it -- see ``ARMING``.
MODES = ("run", "step", "hold", "preview")
REQUESTS = MODES + ("stop", "arm", "reset")


def gate(mode: str, depth: int, budget: int) -> "tuple[str, int]":
    """What this tick may do, and what is left of a step's budget. Pure.

    Returns ``serve`` (take an action and write it), ``hold`` (write nothing) or
    ``wait`` (write nothing, but a chunk has been asked for and is on its way).
    ``wait`` differs from ``hold`` only in what an operator is told.

    In ``step`` the budget is set from the queue the moment a chunk lands, so
    one press executes one chunk however long that policy's chunk happens to be.
    """
    if mode == "run":
        return ("serve" if depth > 0 else "wait"), 0
    if mode in ("hold", "preview"):
        return "hold", 0
    if budget > 0:
        return ("serve", budget - 1) if depth > 0 else ("wait", budget)
    if depth > 0:
        return "serve", depth - 1
    return "wait", 0


def prefetch_threshold(
    static: int,
    explicit: "int | None",
    round_trip_s: float,
    hz: float,
    chunk: int,
    margin: float = 1.5,
) -> int:
    """How many actions may remain when the next chunk is requested. Pure.

    The queue drains at the control rate while a request is in flight, so the
    threshold has to cover *round trip x hz* or the arms run dry every chunk --
    which is exactly what a 32-action diffusion chunk does at 30 Hz against half
    a second of inference: a third of a chunk is 0.33 s of runway against a
    0.6 s round trip. The margin is for the jitter around that measurement.

    An operator who passes ``--prefetch`` means it, so ``explicit`` wins. The
    threshold never reaches the chunk length, or a request would be in flight
    permanently.
    """
    if explicit is not None:
        return max(0, int(explicit))
    if round_trip_s <= 0.0 or hz <= 0.0:
        return static
    need = int(math.ceil(round_trip_s * hz * margin))
    return max(static, min(need, max(1, chunk - 1)))


class RunControl:
    """The mode, the step budget and the latest snapshot, behind one lock.

    The control loop calls :meth:`decide` once per tick and :meth:`publish` with
    what it did; the view calls :meth:`request` and :meth:`snapshot`. Neither
    blocks the other for longer than a dictionary swap.
    """

    def __init__(self, mode: str = "run") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        self._lock = threading.Lock()
        self._mode = mode
        self._budget = 0
        self._step_from = "hold"
        self._stop = False
        self._armed = False
        self._queue_stale = False
        self._reset = False
        self._snapshot: "dict" = {
            "mode": mode,
            "started": time.time(),
            "armed": False,
        }

    # -- the watcher's side --------------------------------------------
    def request(self, mode: str) -> str:
        """Ask for a mode (or ``stop``). Returns the mode now in force."""
        if mode not in REQUESTS:
            raise ValueError(
                f"unknown mode: {mode} (want one of {', '.join(REQUESTS)})"
            )
        with self._lock:
            if mode in ("stop", "reset"):
                # Both end the trial and free the arms; only the second asks for
                # the scene to be put back. Holding is part of the request
                # rather than a courtesy the loop adds afterwards: between the
                # flag being raised and the loop reading it there are ticks, and
                # none of them may serve an action to a rig about to go limp.
                if mode == "stop":
                    self._stop = True
                else:
                    self._reset = True
                self._budget = 0
                self._step_from = "hold"
                self._mode = "hold"
                return self._mode
            if mode == "arm":
                # Consent for THIS trial; ending one withdraws it (see disarm).
                self._armed = True
                return self._mode
            if mode != self._mode:
                if self._mode == "hold":
                    # Only a pause invalidates a plan: it may have lasted
                    # minutes, and the world in that observation is gone.
                    self._queue_stale = True
                if mode == "step":
                    # So that inspecting a plan and executing it is a cycle
                    # rather than a one-way trip into hold.
                    self._step_from = "preview" if self._mode == "preview" else "hold"
                self._budget = 0
                self._mode = mode
            return self._mode

    def snapshot(self) -> "dict":
        with self._lock:
            return dict(self._snapshot)

    # -- the control loop's side ---------------------------------------
    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def armed(self) -> bool:
        """True while consent to enable torque stands, for this trial."""
        with self._lock:
            return self._armed

    def disarm(self) -> None:
        """Withdraw that consent, because the arms have been let go.

        Only a run that took its consent ON the page may be disarmed here: a run
        armed at the terminal has no button to ask again with, and inventing one
        would hand the unauthenticated view a door the flag exists to keep shut.
        The caller knows which kind of run it is; this method does not.
        """
        with self._lock:
            self._armed = False

    def queue_stale(self) -> bool:
        """True once, when the queue must be dropped before the next tick."""
        with self._lock:
            stale, self._queue_stale = self._queue_stale, False
            return stale

    def reset_requested(self) -> bool:
        """True once, when the next attempt on this scene has been asked for."""
        with self._lock:
            wanted, self._reset = self._reset, False
            return wanted

    def stop_requested(self) -> bool:
        """True once, when this trial has been ended with no next one intended.

        A one-shot like the others, and NOT a latch: the loop releases the arms
        and carries on serving the page, so a second press some minutes later
        has to be heard as a second request rather than as the same one still
        being true.
        """
        with self._lock:
            wanted, self._stop = self._stop, False
            return wanted

    def decide(self, depth: int) -> str:
        """``serve`` / ``hold`` / ``wait`` for this tick, advancing a step."""
        with self._lock:
            what, budget = gate(self._mode, depth, self._budget)
            self._budget = budget
            if self._mode == "step" and what == "serve" and budget == 0:
                # That was the last action of the chunk.
                self._mode = self._step_from
            return what

    def publish(self, **fields) -> None:
        with self._lock:
            self._snapshot.update(fields, mode=self._mode, armed=self._armed)
