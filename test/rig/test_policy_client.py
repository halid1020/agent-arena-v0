"""Unit tests for the splice strategies as the remote client applies them.

``test_run_policy`` already covers the client's transport behaviour -- the
handshake, the window, what is fatal. These are about the one thing that changed
when the strategies arrived: what the queue holds after a chunk lands, and that
an unflagged run still behaves exactly as it did.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from actoris_harena.deploy.chunking import ChunkingError
from actoris_harena.deploy.policy_client import RemoteActionSource
from actoris_harena.deploy.policy_wire import decode_request, encode_chunk

CHUNK = 8


class _StubHandler(BaseHTTPRequestHandler):
    """A host whose chunk rows are numbered, so a dropped row is visible."""

    def log_message(self, *_args):  # keep the test output clean
        pass

    def _send(self, payload: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        state = self.server.state
        if self.path == "/reset":
            state["resets"] += 1
            state["session"] = f"session-{state['resets']}"
            self._send(
                json.dumps(
                    {
                        "session": state["session"],
                        "policy_type": "act",
                        "cameras": ["central"],
                        "n_obs_steps": 1,
                        "n_action_steps": CHUNK,
                        "action_dim": 12,
                    }
                ).encode(),
                "application/json",
            )
            return

        request = decode_request(body)
        gate = state.get("gate")
        if gate is not None:
            # Held open so a test can reset the source while this request is
            # still on the socket -- the race, made repeatable.
            gate.wait(5)
        if request.get("session") != state["session"]:
            # What tool/policy_server.py answers a session it has replaced.
            self.send_error(409, "session is not the one that reset this server")
            return
        state["requests"].append(request)
        served = len(state["requests"])
        # Row k of request n is (n * 100 + k), in every channel.
        rows = np.arange(CHUNK, dtype=np.float32)[:, None] + served * 100
        self._send(encode_chunk(np.tile(rows, (1, 12))), "application/octet-stream")


class RemoteSourceCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.state = {"requests": [], "resets": 0, "session": ""}
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    @staticmethod
    def observation(seed=1):
        return np.full(12, float(seed), dtype=np.float32), {
            "central": np.full((8, 8, 3), seed % 255, dtype=np.uint8)
        }

    def source(self, **kwargs):
        return RemoteActionSource(self.url, "task", hz=30.0, **kwargs)

    def fill(self, source):
        """Offer until the first chunk has landed."""
        for _ in range(400):
            source.offer(*self.observation())
            if source.depth:
                return
            time.sleep(0.01)
        self.fail("no chunk arrived")


class TestTheDefault(RemoteSourceCase):
    def test_an_unflagged_client_executes_part_of_each_chunk(self):
        # 'receding' is what a deployment runs: it keeps the plan being followed
        # younger than a whole chunk without a seam on every tick.
        source = self.source()
        self.assertEqual(source.strategy, "receding")
        self.fill(source)
        self.assertEqual(source.depth, CHUNK // 2)


class TestAppending(RemoteSourceCase):
    """What this client did unconditionally before there were strategies."""

    def test_nothing_is_discarded_so_the_whole_chunk_is_queued(self):
        source = self.source(strategy="append")
        self.fill(source)
        self.assertEqual(source.depth, CHUNK)
        self.assertEqual(source.dropped_stale, 0)

    def test_a_second_chunk_lands_behind_the_first(self):
        source = self.source(strategy="append")
        self.fill(source)
        first = source.take()
        source.drain()
        source._splice_in(np.full((CHUNK, 12), 7.0), delay=3)
        self.assertEqual(source.depth, CHUNK)
        self.assertIsNotNone(first)


class TestAligningStrategies(RemoteSourceCase):
    def test_replace_discards_the_rows_whose_moment_has_passed(self):
        source = self.source(strategy="replace")
        self.fill(source)
        # Whatever the measured round trip was, the queue is short by exactly
        # the ticks it consumed, and the leading row is that far into the chunk.
        self.assertEqual(source.depth, CHUNK - source.last_delay)
        self.assertEqual(source.dropped_stale, source.last_delay)
        self.assertAlmostEqual(float(source.take()[0]) % 100, source.last_delay)

    def test_blend_joins_continuously_to_what_was_queued(self):
        source = self.source(strategy="blend", blend_window=4)
        source.drain()
        source._queue.extend(np.full(12, 5.0) for _ in range(6))
        source._splice_in(np.full((CHUNK, 12), 9.0), delay=0)
        # The first action is the one that was already going to be executed.
        np.testing.assert_allclose(source.take(), np.full(12, 5.0))

    def test_ensemble_averages_the_overlap(self):
        source = self.source(strategy="ensemble", new_weight=0.75)
        source.drain()
        source._queue.extend(np.zeros(12) for _ in range(CHUNK))
        source._splice_in(np.ones((CHUNK, 12)), delay=0)
        np.testing.assert_allclose(source.take(), np.full(12, 0.75))

    def test_a_stale_round_trip_can_empty_the_queue(self):
        # Longer in flight than the chunk is long: everything planned was for a
        # tick that has gone by, and the caller must see a stall, not a crash.
        source = self.source(strategy="replace")
        source.drain()
        source._splice_in(np.ones((CHUNK, 12)), delay=CHUNK + 5)
        self.assertEqual(source.depth, 0)
        self.assertIsNone(source.take())

    def test_and_says_why_rather_than_looking_like_a_dead_server(self):
        # MEASURED on a real link: 72 ticks of round trip against a 32-action
        # diffusion chunk. Every aligning splice then discards every row of
        # every chunk, for ever, and the rollout hangs for a minute before
        # dying of "no action" -- which blames the wrong thing entirely.
        source = self.source(strategy="blend")
        source.drain()
        source._splice_in(np.ones((CHUNK, 12)), delay=CHUNK + 5)

        self.assertIn("stale on arrival", source.last_error or "")
        self.assertIn("blend", source.last_error or "")
        self.assertIn(str(CHUNK), source.last_error or "")

    def test_a_splice_that_kept_something_says_nothing(self):
        source = self.source(strategy="replace")
        source.drain()
        source._splice_in(np.ones((CHUNK, 12)), delay=1)

        self.assertEqual(source.depth, CHUNK - 1)
        self.assertIsNone(source.last_error)


class TestBoundaryOffset(RemoteSourceCase):
    """Where the new plan actually STARTS, which is not where it landed."""

    def test_appending_puts_the_join_behind_every_leftover(self):
        source = self.source(strategy="append")
        source.drain()
        source._queue.extend(np.zeros(12) for _ in range(7))
        source._splice_in(np.ones((CHUNK, 12)), delay=2)
        self.assertEqual(source.boundary_offset, 7)

    def test_replacing_puts_the_join_at_the_next_tick(self):
        # 'rtc' is absent because this stub host cannot guide, and asking it
        # for RTC is refused; its splice is covered in test_chunking.
        for strategy in ("replace", "blend", "ensemble"):
            source = self.source(strategy=strategy)
            source.drain()
            source._queue.extend(np.zeros(12) for _ in range(7))
            source._splice_in(np.ones((CHUNK, 12)), delay=2)
            self.assertEqual(source.boundary_offset, 0, msg=strategy)


class TestSync(RemoteSourceCase):
    def test_it_asks_only_once_the_queue_is_empty(self):
        self.assertEqual(self.source(strategy="sync").threshold, 0)

    def test_offer_blocks_so_the_chunk_is_there_when_it_returns(self):
        source = self.source(strategy="sync")
        source.offer(*self.observation())
        # No polling: a synchronous client has its plan by the time offer ends.
        self.assertEqual(source.depth, CHUNK)


class TestReceding(RemoteSourceCase):
    """Execute a fraction of each chunk, then go and look again."""

    def test_it_asks_only_once_the_queue_is_empty(self):
        self.assertEqual(self.source(strategy="receding").threshold, 0)

    def test_offer_blocks_and_leaves_the_ratio_queued(self):
        source = self.source(strategy="receding", execute_ratio=0.5)
        source.offer(*self.observation())
        self.assertEqual(source.depth, CHUNK // 2)

    def test_the_ratio_applies_to_the_chunk_that_actually_came_back(self):
        source = self.source(strategy="receding", execute_ratio=0.25)
        source.offer(*self.observation())
        self.assertEqual(source.depth, CHUNK // 4)

    def test_the_next_request_starts_from_the_new_observation(self):
        # Drain what was queued; the next offer must go and ask again rather
        # than serving the part of the last plan it threw away.
        source = self.source(strategy="receding", execute_ratio=0.5)
        source.offer(*self.observation())
        while source.take() is not None:
            pass
        source.offer(*self.observation())
        self.assertEqual(len(self.server.state["requests"]), 2)
        self.assertEqual(source.depth, CHUNK // 2)


class TestSwitchingStrategy(RemoteSourceCase):
    """Changing the splice while the run continues."""

    def test_the_queue_is_dropped_because_it_was_spliced_by_another_rule(self):
        source = self.source(strategy="append")
        self.fill(source)
        self.assertGreater(source.depth, 0)
        source.set_strategy("replace")
        self.assertEqual(source.depth, 0)
        self.assertEqual(source.strategy, "replace")

    def test_the_settings_it_returns_are_the_ones_in_force(self):
        source = self.source()
        settings = source.set_strategy("receding", execute_ratio=0.25)
        self.assertEqual(settings["strategy"], "receding")
        self.assertEqual(settings["execute_ratio"], 0.25)
        self.assertTrue(settings["blocking"])
        self.assertEqual(settings, source.settings())

    def test_a_number_can_be_changed_without_changing_the_strategy(self):
        source = self.source(strategy="blend")
        source.set_strategy(None, blend_window=9)
        self.assertEqual(source.strategy, "blend")
        self.assertEqual(source.blend_window, 9)

    def test_an_unknown_strategy_is_refused_and_changes_nothing(self):
        source = self.source(strategy="append")
        with self.assertRaises(ChunkingError):
            source.set_strategy("clever")
        self.assertEqual(source.strategy, "append")

    def test_switching_to_rtc_is_refused_by_a_host_that_cannot_guide(self):
        source = self.source(strategy="append")
        with self.assertRaises(ChunkingError):
            source.set_strategy("rtc")
        self.assertEqual(source.strategy, "append")

    def test_a_ratio_outside_the_unit_interval_is_refused(self):
        source = self.source()
        for bad in (0.0, 1.5, -1):
            with self.assertRaises(ChunkingError, msg=str(bad)):
                source.set_strategy("receding", execute_ratio=bad)


class TestStartingAnotherAttempt(RemoteSourceCase):
    """A reset from the page has to reach the HOST, not only the local queue."""

    def test_the_session_is_started_again(self):
        # The host keeps per-session state -- a diffusion policy's own action
        # queue, an RTC guide's previous plan -- and none of it describes the
        # scene about to be attempted.
        source = self.source()
        self.assertEqual(self.server.state["resets"], 1)

        source.reset()

        self.assertEqual(self.server.state["resets"], 2)

    def test_and_nothing_planned_for_the_old_episode_survives_it(self):
        source = self.source()
        self.fill(source)
        self.assertGreater(source.depth, 0)

        source.reset()

        self.assertEqual(source.depth, 0)
        self.assertIsNone(source.last_chunk)

    def test_the_window_is_emptied_so_the_next_plan_sees_only_the_new_scene(self):
        source = self.source()
        self.fill(source)

        source.reset()

        self.assertFalse(source.window.ready)

    def test_the_chunk_length_it_negotiated_is_kept(self):
        # Re-deriving it from the fresh handshake must not quietly widen a run
        # that was started with --actions-per-chunk.
        source = self.source(actions_per_chunk=3)

        source.reset()

        self.assertEqual(source.actions, 3)


class TestCameraMap(RemoteSourceCase):
    def test_a_camera_is_renamed_on_the_way_out(self):
        # The twin renders 'scene'; this checkpoint was trained on 'central'.
        source = self.source(camera_map={"scene": "central"})
        state = np.zeros(12, dtype=np.float32)
        source.offer(state, {"scene": np.zeros((8, 8, 3), dtype=np.uint8)})
        for _ in range(400):
            if self.server.state["requests"]:
                break
            source.offer(state, {"scene": np.zeros((8, 8, 3), dtype=np.uint8)})
            time.sleep(0.01)
        self.assertEqual(self.server.state["requests"][0]["cameras"], ["central"])

    def test_what_the_policy_was_shown_is_recorded_under_the_sent_name(self):
        source = self.source(camera_map={"scene": "central"})
        source.offer(
            np.zeros(12, dtype=np.float32),
            {"scene": np.zeros((8, 8, 3), dtype=np.uint8)},
        )
        self.assertEqual(sorted(source.last_sent()[1]), ["central"])


class TestRefusals(RemoteSourceCase):
    def test_an_unknown_strategy_is_refused_before_the_handshake(self):
        with self.assertRaises(ChunkingError):
            self.source(strategy="clever")

    def test_rtc_against_a_host_that_cannot_guide_is_refused(self):
        # Silently giving 'replace' under the name 'rtc' would produce a
        # result nobody could trust afterwards.
        with self.assertRaises(ChunkingError) as caught:
            self.source(strategy="rtc")
        self.assertIn("RTC", str(caught.exception))


class TestTaskCanBeSetLater(RemoteSourceCase):
    """A run may be started with no task and given one from the live view."""

    def test_the_next_request_carries_the_new_task(self):
        source = self.source()
        self.assertEqual(source.set_task("fold the towel"), "fold the towel")
        self.fill(source)
        sent = self.server.state["requests"][-1]
        self.assertEqual(sent["task"], "fold the towel")

    def test_a_source_may_start_with_no_task_at_all(self):
        source = RemoteActionSource(self.url, "", hz=30.0)
        self.assertEqual(source.task, "")
        source.set_task("pick up the cube")
        self.fill(source)
        self.assertEqual(self.server.state["requests"][-1]["task"], "pick up the cube")


class TestAChunkThatOUTLIVESItsAttempt(RemoteSourceCase):
    """Every non-blocking strategy fetches on its own thread, so a request can
    still be on the socket when the episode that asked for it ends. Both of its
    outcomes belong to a scene that no longer exists.
    """

    def gated(self):
        """A source with one request held open on the host."""
        gate = threading.Event()
        self.server.state["gate"] = gate
        self.addCleanup(gate.set)
        source = self.source(strategy="append")
        source.offer(*self.observation())
        for _ in range(200):  # let the fetch thread reach the socket
            if self.server.state.get("session"):
                break
            time.sleep(0.005)
        return source, gate

    def test_its_chunk_is_not_spliced_into_the_next_one(self):
        # Held by the client, not only by the host: our own server refuses the
        # superseded session before it plans anything, but a host without one
        # would answer, and those actions were planned for objects that have
        # since been put back.
        source, gate = self.gated()

        source.reset()
        gate.set()
        time.sleep(0.3)

        self.assertEqual(source.depth, 0)
        self.assertIsNone(source.last_chunk)

    def test_and_its_refusal_does_not_end_the_run(self):
        # The visible half: the host answers a superseded session with 409, and
        # a 4xx is otherwise fatal -- which ends a sweep at an episode boundary
        # it was right to cross.
        source, gate = self.gated()

        source.reset()
        gate.set()
        time.sleep(0.3)

        self.assertIsNone(source.fatal)

    def test_so_the_next_attempt_can_still_ask_for_a_plan(self):
        # The abandoned fetch held the one request slot; reset frees it, or the
        # new attempt would never send anything.
        source, gate = self.gated()

        source.reset()
        gate.set()
        self.server.state["gate"] = None
        self.fill(source)

        self.assertGreater(source.depth, 0)
        self.assertIsNone(source.fatal)


if __name__ == "__main__":
    unittest.main()


class TestRecoveringAStolenSession(RemoteSourceCase):
    """Another client resetting the host must not end this one permanently.

    The host keeps one session slot. A second client -- a rollout page, a stray
    probe, another sweep -- claims it with a reset, and from then on every
    request of ours is answered 409. That used to latch: nothing cleared the
    flag, so the only cure was a new process, and a multi-hour grid died in its
    second cell. A fresh handshake is a valid session again, so it clears.
    """

    def steal_the_session(self):
        """What a second client does to us, without a second client."""
        import urllib.request

        urllib.request.urlopen(
            urllib.request.Request(
                f"{self.url}/reset",
                data=json.dumps({"task": "someone else"}).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=5,
        ).read()

    def test_the_host_refuses_once_the_slot_is_taken(self):
        source = self.source()
        self.fill(source)
        self.steal_the_session()
        for _ in range(200):
            source.drain()
            source.offer(*self.observation())
            if source.fatal:
                break
            time.sleep(0.01)
        self.assertIsNotNone(source.fatal)
        self.assertIn("409", source.fatal)

    def test_and_a_new_handshake_clears_the_refusal(self):
        source = self.source()
        self.fill(source)
        self.steal_the_session()
        for _ in range(200):
            source.drain()
            source.offer(*self.observation())
            if source.fatal:
                break
            time.sleep(0.01)
        self.assertIsNotNone(source.fatal)

        source.reset()

        self.assertIsNone(source.fatal)

    def test_and_chunks_flow_again_afterwards(self):
        source = self.source()
        self.fill(source)
        self.steal_the_session()
        for _ in range(200):
            source.drain()
            source.offer(*self.observation())
            if source.fatal:
                break
            time.sleep(0.01)
        source.reset()

        self.fill(source)  # fails the test if nothing arrives
        self.assertIsNotNone(source.take())
