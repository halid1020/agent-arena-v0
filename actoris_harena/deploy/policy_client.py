"""Where a rollout's actions come from: this process, or another machine.

Both sources answer the same small protocol, so a control loop never learns
which one it has:

    ``offer(state, images)``  -- here is this tick's observation
    ``take() -> action | None`` -- give me something to execute, or nothing
    ``drain()`` / ``set_paused(bool)`` / ``depth`` / ``last_sent()``

:class:`LocalActionSource` runs the forward pass inline, exactly as it always
did. :class:`RemoteActionSource` sends the observation window to a GPU host and
splices the returned chunk into what is already queued, following one of the
strategies in :mod:`actoris_harena.deploy.chunking`.

``reset()`` is on the remote source only, and a caller looks for it rather than
assuming it: a second attempt on the same scene needs the host to forget the
session it has been accumulating, and inference in this process has no such
session to begin again.

Neither touches a camera or a servo bus, which is what lets the same client
drive the rig and the digital twin.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from actoris_harena.deploy.chunking import (
    DEFAULT_BLEND_WINDOW,
    DEFAULT_ENSEMBLE_WEIGHT,
    DEFAULT_EXECUTE_RATIO,
    GUIDED,
    STRATEGIES,
    ChunkingError,
    delay_ticks,
    splice,
)

#: Strategies that block in ``offer`` until the reply lands. They ask only when
#: the queue is empty, so nothing is ever executed from a stale plan -- and the
#: arms hold still for the whole round trip, which is the price.
BLOCKING = ("sync", "receding")
from actoris_harena.deploy.policy_run import prefetch_threshold


def _infer(
    policy, preprocessor, postprocessor, build_batch, state, images, task, device, torch
) -> np.ndarray:
    """One policy step: observation -> 12-D action (numpy). Mirrors eval_sim_policy."""
    batch = build_batch(state, images, task, device)
    with torch.no_grad():
        batch = preprocessor(batch)
        action = policy.select_action(batch)
        action = postprocessor(action)
    return np.asarray(action.squeeze(0).to("cpu")).astype(float)


class LocalActionSource:
    """Inference in this process: one ``select_action`` per tick, as before.

    The policy's own action queue means only every ``n_action_steps``-th call
    touches the GPU; the rest are dequeues. Kept exactly as it was so that
    running without ``--server`` is unchanged.
    """

    def __init__(self, checkpoint: str, device: str, task: str) -> None:
        import torch

        from tool.eval_sim_policy import build_batch, load_policy

        self.torch = torch
        self.build_batch = build_batch
        self.policy, self.pre, self.post, self.type = load_policy(checkpoint, device)
        self.policy.reset()
        #: Which weights answered. Both sources carry it under the same name so
        #: a run log can say WHICH checkpoint a rollout scored, rather than only
        #: that one did -- the whole point of comparing several.
        self.checkpoint = str(checkpoint)
        self.device = device
        self.task = task
        self.cameras: "list[str] | None" = None  # whatever the rig is configured with
        self.fatal: "str | None" = None
        self.last_error: "str | None" = None
        self.round_trip_s = 0.0  # nothing travels; kept so both sources report alike
        self._latest: "tuple[np.ndarray, dict] | None" = None

    def describe(self) -> str:
        return f"local '{self.type}' policy on {self.device}"

    #: Inference here has no chunk to step through: the policy's own queue is
    #: internal, so a tick either infers or dequeues and the operator sees one
    #: action at a time.
    chunked = False
    last_chunk: "np.ndarray | None" = None
    last_chunk_seq = 0
    last_chunk_at = 0.0

    def set_task(self, task: str) -> str:
        """Change the language task. Takes effect on the next inference."""
        self.task = str(task)
        return self.task

    def offer(self, state: np.ndarray, images: dict) -> None:
        self._latest = (state, images)

    def drain(self) -> None:
        """Nothing is queued here; the next take() infers from what is offered."""

    def set_paused(self, paused: bool) -> None:
        """No background fetching to pause."""

    def last_sent(self) -> "tuple[np.ndarray, dict] | None":
        return self._latest

    def take(self) -> "np.ndarray | None":
        if self._latest is None:
            return None
        state, images = self._latest
        return _infer(
            self.policy,
            self.pre,
            self.post,
            self.build_batch,
            state,
            images,
            self.task,
            self.device,
            self.torch,
        )

    @property
    def depth(self) -> int:
        return 0


class RemoteActionSource:
    """Inference on another machine; action chunks arrive ahead of being needed.

    The caller keeps capturing at the control rate and feeds every observation
    into a rolling window, because a policy with more than one observation step
    was trained on adjacent frames. When the local action queue runs low, the
    window is sent and the next chunk requested on a background thread, so the
    round trip overlaps motion already being executed and the network never sits
    inside the control loop.

    THE SPLICE. A chunk lands after the world has moved on, and ``strategy``
    decides what to do about it -- see :mod:`actoris_harena.deploy.chunking`. The default is
    ``receding``: half of each chunk is executed and then the next is asked for,
    so the plan being followed is never much older than one round trip, and the
    seams are as far apart as that allows. ``append`` is what this class used to
    do unconditionally, and is now one choice among several; every strategy but
    that one discards the rows whose moment has passed, which is measured from
    this link's own round trip rather than assumed.

    ``sync`` is the exception to all of the above: it blocks in ``offer`` until
    the reply arrives, because its whole point is that nothing is executed from a
    stale plan. The arms hold still for the round trip.

    A request that the server rejects outright (4xx) is fatal: the observation
    or the session is wrong and repeating it cannot help. A dropped connection
    or a timeout is not -- the next tick tries again, and the caller's stall
    policy decides how long that is allowed to go on.
    """

    def __init__(
        self,
        url: str,
        task: str,
        actions_per_chunk: "int | None" = None,
        prefetch: "int | None" = None,
        timeout_s: float = 20.0,
        hz: float = 30.0,
        strategy: str = "receding",
        blend_window: int = DEFAULT_BLEND_WINDOW,
        ramp_kind: str = "linear",
        new_weight: float = DEFAULT_ENSEMBLE_WEIGHT,
        camera_map: "dict[str, str] | None" = None,
        virtual_delay_ticks: "int | None" = None,
        execute_ratio: float = DEFAULT_EXECUTE_RATIO,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ChunkingError(
                f"unknown strategy: {strategy} (want one of {', '.join(STRATEGIES)})"
            )
        self.url = url.rstrip("/")
        self.task = task
        self.timeout_s = float(timeout_s)
        self.strategy = strategy
        self.blend_window = int(blend_window)
        self.ramp_kind = ramp_kind
        self.new_weight = float(new_weight)
        self.execute_ratio = float(execute_ratio)
        #: Rename a camera on the way out, for a checkpoint that was trained
        #: under a different name for the same viewpoint (the twin renders
        #: ``scene`` where the rig records ``central``). Renaming here rather
        #: than in the environment keeps the wire honest about what the
        #: checkpoint asked for.
        self.camera_map = dict(camera_map or {})
        #: Deterministic pacing. When set, the request is made inline and the
        #: reply is WITHHELD for exactly this many ticks however long it really
        #: took, so a strategy can be swept against a delay the network will not
        #: oblige by producing, and the same sweep answers the same way twice.
        self.virtual_delay_ticks = (
            None if virtual_delay_ticks is None else max(0, int(virtual_delay_ticks))
        )
        self._pending: "tuple[np.ndarray, int, int] | None" = None
        #: Which attempt the source is on. A request is built against the
        #: session of the attempt that asked for it, so a reply that arrives
        #: after :meth:`reset` has moved on answers a question about a scene
        #: that is gone. Bumped by reset, captured by every fetch, and checked
        #: before that fetch is allowed to touch anything.
        self._generation = 0

        self._handshake(actions_per_chunk)
        # The static default covers a fast policy; the threshold below grows it
        # to cover whatever round trip this link turns out to have.
        self.prefetch = max(1, self.actions // 3)
        self.explicit_prefetch = None if prefetch is None else int(prefetch)
        self.hz = float(hz)

        self._queue: "deque[np.ndarray]" = deque()
        self._lock = threading.Lock()
        self._inflight = False
        self._seq = 0
        self.fatal: "str | None" = None
        self.last_error: "str | None" = None
        self.round_trip_s = 0.0
        self.server_infer_s = 0.0
        #: True when the last splice discarded the whole chunk it was handed.
        self._stale_on_arrival = False
        #: Ticks of round trip the last splice compensated for, and how many
        #: actions it threw away because their moment had passed. Reported so a
        #: strategy can be judged without re-deriving it from the log.
        self.last_delay = 0
        self.dropped_stale = 0
        self.boundary_offset = 0
        self._paused = False
        self._last_sent: "tuple[np.ndarray, dict] | None" = None
        self.last_chunk: "np.ndarray | None" = None
        self.last_chunk_seq = 0
        self.last_chunk_at = 0.0

    #: A chunk is a visible unit here, so a rollout can be stepped one at a time.
    chunked = True

    @property
    def guided(self) -> bool:
        """True when the smoothing is the policy's job, not the splice's."""
        return self.strategy in GUIDED

    @property
    def threshold(self) -> int:
        """Queue depth at which the next chunk is requested. See policy_run.

        A blocking strategy asks only once the queue is empty: it is defined by
        not having a plan in flight while one is executing.
        """
        if self.strategy in BLOCKING:
            return 0
        return prefetch_threshold(
            self.prefetch,
            self.explicit_prefetch,
            self.round_trip_s,
            self.hz,
            self.actions,
        )

    def _handshake(self, actions_per_chunk: "int | None") -> None:
        """Ask the host to start a session, and size the window and the chunk.

        Also the whole of :meth:`reset`, which is why it is not inline in the
        constructor: a second attempt needs a session the host has forgotten the
        history of, and re-deriving the sizes from the same answer is what keeps
        the two paths from drifting apart.
        """
        import json

        from actoris_harena.deploy.policy_wire import ObservationWindow

        meta = json.loads(self._rpc("/reset", b""))
        self.type = meta["policy_type"]
        #: See LocalActionSource.checkpoint. The server reports it in /meta, so
        #: this costs nothing but was being thrown away.
        self.checkpoint = str(meta.get("checkpoint") or "")
        self.session = meta["session"]
        self.cameras = sorted(meta["cameras"])
        self.n_obs_steps = int(meta["n_obs_steps"])
        self.server_actions = int(meta["n_action_steps"])
        self.actions = min(
            int(actions_per_chunk or self.server_actions), self.server_actions
        )
        self.host_guides = bool(meta.get("rtc"))
        if self.guided and not self.host_guides:
            # Refused rather than quietly downgraded: RTC's smoothing happens
            # inside the denoiser on the host, so a host that does not do it
            # gives exactly 'replace' -- and a run labelled 'rtc' that was
            # really 'replace' is a result nobody can trust afterwards.
            raise ChunkingError(
                f"the host at {self.url} does not do RTC guidance "
                f"(policy '{self.type}'); its answer would be plain 'replace'"
            )
        #: Rebuilt, not kept: the frames in it are of the scene as it was, and a
        #: policy with several observation steps would plan the next attempt
        #: partly from the last one.
        self.window = ObservationWindow(self.cameras, self.n_obs_steps)

    def reset(self, scenario=None) -> None:
        """Begin a new attempt: a fresh session, and nothing carried into it.

        The host keeps per-session state -- a diffusion policy's own action
        queue, an RTC guide's previous plan -- and none of it describes the
        scene that is about to be attempted. So the session is started again,
        and everything this end had planned goes with it: a chunk queued for the
        old episode would drive the arms at whatever is no longer there.

        ``scenario`` is accepted and ignored, so a caller can hand the same
        argument to the rig and the source without knowing which one uses it.
        """
        with self._lock:
            self._generation += 1
            self._queue.clear()
            self._pending = None
            self.last_chunk = None
            self._stale_on_arrival = False
            # A fetch for the previous attempt may still be on the socket. It is
            # abandoned here rather than waited for -- its answer is void either
            # way -- so the slot it holds is released now, or the first tick of
            # the new attempt would find the source already busy.
            self._inflight = False
        self._handshake(self.actions)
        # A handshake that returned is a session this host will answer, so
        # whatever it refused before no longer describes the client. Clearing
        # the latch here is what lets a caller recover from another client
        # taking the server's single session slot out from under it: without
        # it, `offer` refuses forever and the only cure is a new process.
        with self._lock:
            self.fatal = None

    def drain(self) -> None:
        """Forget what is queued: it was planned from an older observation."""
        with self._lock:
            self._queue.clear()

    def set_strategy(self, strategy: "str | None" = None, **params) -> dict:
        """Change the splice, and the numbers it uses, while the run continues.

        The queue is DROPPED. What is in it was spliced under the old rule --
        under ``append`` it may be two plans deep, under ``blend`` its leading
        rows are a cross-fade into a plan the new rule would not have chosen --
        and carrying that into a different strategy would make the first chunk
        of every switch belong to neither. The cost is one round trip, which is
        the same cost as resuming from a pause.

        Returns the settings now in force, so a caller need not guess what was
        accepted. Unknown parameters are ignored rather than refused: the page
        sends whatever its controls hold, and only some apply to any strategy.
        """
        if strategy is not None and strategy not in STRATEGIES:
            raise ChunkingError(
                f"unknown strategy: {strategy} (want one of {', '.join(STRATEGIES)})"
            )
        if strategy in GUIDED and not self.host_guides:
            raise ChunkingError(
                f"the host at {self.url} does not do RTC guidance "
                f"(policy '{self.type}'); its answer would be plain 'replace'"
            )
        ratio = params.get("execute_ratio")
        if ratio is not None and not 0.0 < float(ratio) <= 1.0:
            raise ChunkingError(f"execute_ratio must be in (0, 1], got {ratio}")
        with self._lock:
            if strategy is not None:
                self.strategy = strategy
            if ratio is not None:
                self.execute_ratio = float(ratio)
            if params.get("blend_window") is not None:
                self.blend_window = max(1, int(params["blend_window"]))
            if params.get("new_weight") is not None:
                weight = float(params["new_weight"])
                if not 0.0 <= weight <= 1.0:
                    raise ChunkingError(f"new_weight must be in [0, 1], got {weight}")
                self.new_weight = weight
            if params.get("ramp_kind") is not None:
                self.ramp_kind = str(params["ramp_kind"])
            self._queue.clear()
            self._pending = None
        return self.settings()

    def settings(self) -> dict:
        """The splice in force and the numbers it uses."""
        return {
            "strategy": self.strategy,
            "execute_ratio": round(self.execute_ratio, 3),
            "blend_window": int(self.blend_window),
            "new_weight": round(self.new_weight, 3),
            "ramp_kind": self.ramp_kind,
            "blocking": self.strategy in BLOCKING,
        }

    def set_task(self, task: str) -> str:
        """Change the language task the requests carry.

        The task travels with every ``/act`` message rather than being fixed at
        the handshake, so this needs no reset: the next chunk simply answers a
        different question. A run may therefore be STARTED without a task and
        given one from the live view.
        """
        with self._lock:
            self.task = str(task)
            return self.task

    def set_paused(self, paused: bool) -> None:
        """While paused the window keeps filling but no chunk is requested."""
        with self._lock:
            self._paused = bool(paused)

    def last_sent(self) -> "tuple[np.ndarray, dict] | None":
        """The observation the last request carried -- what the policy was shown."""
        return self._last_sent

    def describe(self) -> str:
        return (
            f"remote '{self.type}' policy at {self.url} "
            f"({self.n_obs_steps} obs step(s) per request, "
            f"{self.actions} actions per chunk, {self.strategy} splice)"
        )

    def _rpc(self, path: str, data: "bytes | None" = None) -> bytes:
        import urllib.request

        request = urllib.request.Request(
            self.url + path,
            data=data,
            method="GET" if data is None else "POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    def _rename(self, images: dict) -> dict:
        if not self.camera_map:
            return images
        return {
            self.camera_map.get(name, name): frame for name, frame in images.items()
        }

    def offer(self, state: np.ndarray, images: dict) -> None:
        """Record this tick's observation and, if the queue is low, ask for more."""
        images = self._rename(images)
        self.window.push(state, images)
        with self._lock:
            if (
                self._inflight
                or self._pending is not None
                or self._paused
                or self.fatal is not None
                or len(self._queue) > self.threshold
            ):
                return
            self._inflight = True
            self._seq += 1
            seq = self._seq
        self._last_sent = (state, images)
        steps = self.window.steps()
        if self.virtual_delay_ticks is not None:
            # Inline, so the wall clock plays no part in when this lands.
            self._fetch(steps, seq)
            return
        if self.strategy in BLOCKING:
            # Blocking on purpose: nothing may be executed from a stale plan, so
            # the caller waits here rather than running the queue down.
            self._fetch(steps, seq)
            return
        threading.Thread(target=self._fetch, args=(steps, seq), daemon=True).start()

    def _splice_in(self, chunk: np.ndarray, delay: int) -> None:
        """Merge an arrived chunk into the queue. Caller holds the lock."""
        leftover = (
            np.asarray(self._queue, dtype=float)
            if self._queue
            else np.zeros((0, chunk.shape[1]), dtype=float)
        )
        merged = splice(
            self.strategy,
            leftover,
            chunk,
            delay,
            window=self.blend_window,
            ramp_kind=self.ramp_kind,
            new_weight=self.new_weight,
            execute_ratio=self.execute_ratio,
        )
        self.last_delay = delay
        self.dropped_stale = 0 if self.strategy == "append" else min(delay, len(chunk))
        #: Actions still to be taken before the FIRST action of the chunk just
        #: spliced. Zero for every strategy that replaces the queue; for
        #: ``append`` it is the whole leftover, because that is exactly how long
        #: the new plan waits. A caller measuring the join needs this: the
        #: boundary is not where the chunk landed, it is where it starts.
        self.boundary_offset = len(leftover) if self.strategy == "append" else 0
        self._queue = deque(np.asarray(row, dtype=float) for row in merged)
        self._stale_on_arrival = bool(len(merged) == 0 and len(chunk))
        if self._stale_on_arrival:
            # An ALIGNING splice throws away the rows that were planned for
            # ticks already executed; when the round trip is longer than the
            # chunk itself, that is every row, and the arms get nothing at all
            # -- for ever, since the next chunk will be just as late. Silence
            # here reads as "the server is not answering", which it is not.
            self.last_error = (
                f"the whole chunk was stale on arrival: {self.strategy} aligns a "
                f"plan to the present, and {delay} ticks of round trip is longer "
                f"than its {len(chunk)} actions. Use 'append', or shorten the "
                f"round trip"
            )

    def _fetch(self, steps, seq: int) -> None:
        import urllib.error

        from actoris_harena.deploy.policy_wire import decode_chunk, encode_request

        started = time.perf_counter()
        # The attempt this request belongs to. Everything below is conditional
        # on it still being the current one: see _obsolete.
        generation = self._generation
        try:
            message = encode_request(
                steps, self.task, session=self.session, seq=seq, actions=self.actions
            )
            chunk, header = decode_chunk(self._rpc("/act", message))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace").strip()[:200]
            if self._obsolete(generation):
                return
            self.last_error = f"HTTP {exc.code}: {body}"
            if 400 <= exc.code < 500:
                self.fatal = self.last_error
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a transport hiccup must not kill the loop
            if self._obsolete(generation):
                return
            self.last_error = f"{type(exc).__name__}: {exc}"
        else:
            if self._obsolete(generation):
                return
            round_trip = time.perf_counter() - started
            arrived = np.asarray(chunk, dtype=float)
            with self._lock:
                if self.virtual_delay_ticks is not None:
                    # Withheld: released by take() after the chosen number of
                    # ticks, so the queue drains for exactly as long as the
                    # experiment says it should.
                    self._pending = (arrived, self.virtual_delay_ticks, seq)
                else:
                    self._splice_in(arrived, delay_ticks(round_trip, self.hz))
                    self.last_chunk = arrived
                    self.last_chunk_seq = seq
                    self.last_chunk_at = time.time()
                self.round_trip_s = round_trip
                self.server_infer_s = float(
                    (header.get("timings") or {}).get("infer_s", 0.0)
                )
                if not self._stale_on_arrival:
                    # The request succeeded; clear whatever the last one failed
                    # with -- unless the splice just threw the whole chunk away,
                    # which is the one thing the transport being fine does not
                    # fix, and which nothing else would ever say out loud.
                    self.last_error = None
        finally:
            with self._lock:
                # Only if this fetch is still the current one: reset() already
                # released the slot, and a newer request may hold it by now.
                if generation == self._generation:
                    self._inflight = False

    def _obsolete(self, generation: int) -> bool:
        """Has :meth:`reset` moved on since this fetch asked its question?

        A fetch runs on its own thread for every non-blocking strategy, so one
        outlives the episode that started it whenever an attempt ends while a
        chunk is in the air. Its answer describes a scene that has since been
        rebuilt, and its session is one the host has replaced -- so neither of
        its outcomes may be acted on.

        In practice it is the REFUSAL that arrives: tool/policy_server.py
        answers a superseded session with 409, and a 4xx is otherwise fatal, so
        a sweep ended at an episode boundary it was right to cross. That check
        is also what stops the other outcome, a chunk planned for the previous
        scene being spliced into this one and driving the arms at objects that
        have since been put back. Dropping both here makes that true of the
        client rather than only of the one host, which matters for a host with
        no session of its own to check.
        """
        with self._lock:
            return generation != self._generation

    def _release_due(self) -> None:
        """Count a withheld chunk down one tick, splicing it when due.

        Caller holds the lock. The delay is spent whether or not the queue has
        anything left, because that is what a real round trip does.
        """
        if self._pending is None:
            return
        chunk, remaining, seq = self._pending
        if remaining > 0:
            self._pending = (chunk, remaining - 1, seq)
            return
        self._pending = None
        self._splice_in(chunk, self.virtual_delay_ticks or 0)
        self.last_chunk = chunk
        self.last_chunk_seq = seq
        self.last_chunk_at = time.time()

    def pending(self) -> "np.ndarray | None":
        """The actions that will be executed next, in order.

        NOT the chunk the policy returned: under every strategy but ``append``
        the queue is the spliced result, and under ``blend`` its leading rows
        exist in no chunk at all. A preview that showed the raw chunk would be
        showing motion the arms are not going to make.
        """
        with self._lock:
            if not self._queue:
                return None
            return np.asarray(self._queue, dtype=float)

    def take(self) -> "np.ndarray | None":
        with self._lock:
            self._release_due()
            return self._queue.popleft() if self._queue else None

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def in_flight(self) -> bool:
        """True while a request is out, or a reply is being withheld."""
        with self._lock:
            return self._inflight or self._pending is not None
