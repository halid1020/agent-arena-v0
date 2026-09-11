"""Unit tests for the rig seam and deterministic pacing.

``TwinRig`` is checked against a stubbed environment rather than MuJoCo, so
these stay in the fast tier; the twin itself is exercised by the integration
tier. The pacing tests are the important half: an experiment that cannot repeat
itself is not measuring a strategy, it is measuring the network.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from actoris_harena.deploy.policy_client import RemoteActionSource
from actoris_harena.deploy.policy_rig import TwinRig
from actoris_harena.deploy.policy_wire import decode_request, encode_chunk

CHUNK = 10

#: The dual SO-101's action layout, as the recorder writes it:
#: ``[left5 deg, left_grip, right5 deg, right_grip]``. TwinRig takes this as an
#: argument rather than importing it, because the mapping from one action vector
#: to whatever an env.tick wants is the RIG's -- it is where the gripper channel
#: and the joint units are decided, and a single six-joint arm decides them
#: differently. Passing it here is what keeps this test about the rig protocol
#: and not about one robot's units.
GRIPPER_OPEN_MAX_FRAC = 0.6


def decode_action(action):
    a = np.asarray(action, dtype=float)
    q_rad = np.radians(np.concatenate([a[0:5], a[6:11]]))
    grip = {
        side: float(np.clip(g, 0.0, GRIPPER_OPEN_MAX_FRAC) / GRIPPER_OPEN_MAX_FRAC)
        for side, g in (("left", a[5]), ("right", a[11]))
    }
    return q_rad, grip


class StubEnv:
    """Records what was commanded; renders a frame per camera name."""

    def __init__(self, cameras=("scene", "wrist_camera_left")):
        self.cameras = list(cameras)
        self.commands = []
        self.state = np.zeros(12, dtype=np.float32)

    def observe(self, camera_wh):
        w, h = camera_wh
        return self.state, {
            name: np.zeros((h, w, 3), dtype=np.uint8) for name in self.cameras
        }

    def tick(self, q_rad_10, grip_frac):
        self.commands.append((np.asarray(q_rad_10, dtype=float), dict(grip_frac)))

    def reset(self, scenario):
        self.commands.clear()


class TestTwinRig(unittest.TestCase):
    def rig(self, **kwargs):
        env = StubEnv()
        kwargs.setdefault("decode_action", decode_action)
        return env, TwinRig(env, cameras=env.cameras, camera_wh=(32, 24), **kwargs)

    def test_an_action_reaches_the_environment_as_joints_and_grippers(self):
        env, rig = self.rig()
        rig.command(np.zeros(12))
        self.assertEqual(len(env.commands), 1)
        q_rad, grip = env.commands[0]
        self.assertEqual(q_rad.shape, (10,))
        self.assertEqual(sorted(grip), ["left", "right"])

    def test_time_advances_only_when_something_is_commanded(self):
        env, rig = self.rig()
        for _ in range(3):
            rig.observe()
        self.assertEqual(env.commands, [])
        self.assertEqual(rig.ticks, 0)

    def test_a_hold_repeats_the_last_goal_rather_than_freezing_the_world(self):
        # A starved queue must cost the same here as on the bench: the arms stop
        # where they are, and the world keeps moving.
        env, rig = self.rig()
        rig.command(np.full(12, 0.2))
        rig.hold()
        self.assertEqual(len(env.commands), 2)
        np.testing.assert_allclose(env.commands[0][0], env.commands[1][0])

    def test_a_hold_before_any_action_holds_the_measured_state(self):
        env, rig = self.rig()
        rig.hold()
        self.assertEqual(len(env.commands), 1)

    def test_the_reported_cameras_are_the_names_the_policy_will_be_sent(self):
        _, rig = self.rig(camera_map={"scene": "central"})
        self.assertEqual(rig.cameras, ["central", "wrist_camera_left"])

    def test_a_release_leaves_the_pose_the_trial_ended_in(self):
        # The bench's followers keep their measured pose when torque comes off,
        # and a rehearsal of "stop, then look at where it ended up" has to show
        # the same thing. Only a reset forgets an attempt.
        _env, rig = self.rig()
        rig.command(np.arange(12.0))
        rig.release()

        self.assertIsNotNone(rig.last_action)
        self.assertEqual(rig.ticks, 1)

    def test_and_a_reset_is_the_one_that_forgets_it(self):
        _env, rig = self.rig()
        rig.command(np.arange(12.0))
        rig.reset(None)

        self.assertIsNone(rig.last_action)
        self.assertEqual(rig.ticks, 0)

    def test_shutdown_releases_nothing_and_raises_nothing(self):
        _, rig = self.rig()
        rig.shutdown()


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _send(self, payload, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/reset":
            self._send(
                json.dumps(
                    {
                        "session": "s1",
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
        self.server.state["requests"].append(decode_request(body))
        served = len(self.server.state["requests"])
        rows = np.arange(CHUNK, dtype=np.float32)[:, None] + served * 100
        self._send(encode_chunk(np.tile(rows, (1, 12))), "application/octet-stream")


class TestVirtualDelay(unittest.TestCase):
    """A chosen delay, honoured exactly, whatever the link really did."""

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.state = {"requests": []}
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def source(self, delay, strategy="replace"):
        return RemoteActionSource(
            self.url,
            "task",
            hz=30.0,
            strategy=strategy,
            virtual_delay_ticks=delay,
        )

    @staticmethod
    def observation():
        return np.zeros(12, dtype=np.float32), {
            "central": np.zeros((8, 8, 3), dtype=np.uint8)
        }

    def test_the_reply_is_withheld_for_exactly_the_chosen_ticks(self):
        source = self.source(delay=4)
        source.offer(*self.observation())
        # The request has already been answered, but nothing may be taken yet.
        for tick in range(4):
            self.assertIsNone(source.take(), msg=f"tick {tick}")
        self.assertIsNotNone(source.take())

    def test_no_second_request_goes_out_while_a_reply_is_withheld(self):
        source = self.source(delay=5)
        for _ in range(4):
            source.offer(*self.observation())
        self.assertEqual(len(self.server.state["requests"]), 1)

    def test_the_splice_compensates_for_the_delay_it_was_told(self):
        source = self.source(delay=3, strategy="replace")
        source.offer(*self.observation())
        for _ in range(3):
            source.take()
        # Rows 0..2 were planned for ticks that have gone by, so row 3 leads.
        self.assertAlmostEqual(float(source.take()[0]) % 100, 3.0)
        self.assertEqual(source.dropped_stale, 3)

    def test_the_wall_clock_plays_no_part(self):
        # Two sources, the same delay, the same answer -- which is the whole
        # point of virtual pacing.
        depths = []
        for _ in range(2):
            source = self.source(delay=2)
            source.offer(*self.observation())
            time.sleep(0.05)
            source.take()
            source.take()
            source.take()
            depths.append(source.depth)
        self.assertEqual(depths[0], depths[1])

    def test_a_delay_of_zero_lands_immediately(self):
        source = self.source(delay=0)
        source.offer(*self.observation())
        self.assertIsNotNone(source.take())

    def test_a_live_link_is_unaffected_by_any_of_this(self):
        source = RemoteActionSource(self.url, "task", hz=30.0)
        self.assertIsNone(source.virtual_delay_ticks)
        self.assertFalse(source.in_flight)


if __name__ == "__main__":
    unittest.main()
