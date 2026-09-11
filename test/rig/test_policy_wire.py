"""The observation/action wire format, without a network, a GPU or a robot.

The format exists so the rig and a policy host agree on what an observation is.
These tests pin the parts that would fail silently or expensively otherwise: the
ORDER of a multi-step window (a policy trained on adjacent frames is wrecked by a
reversed pair, and nothing would raise), the camera-set agreement (a mismatch
otherwise surfaces as a shape error inside inference, with arms already under
torque), and the fact that a chunk survives the round trip exactly -- these
numbers are joint targets in degrees.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_policy_wire
"""

import unittest

import numpy as np

from actoris_harena.deploy.policy_wire import (
    ObservationWindow,
    WireError,
    decode_chunk,
    decode_request,
    encode_chunk,
    encode_request,
)

_CAMERAS = ("central", "wrist_camera_left", "wrist_camera_right")


def _frame(seed: int, h: int = 32, w: int = 48) -> np.ndarray:
    """A smooth gradient, so JPEG is near-lossless and comparisons stay tight."""
    y = np.linspace(0, 200, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 200, w, dtype=np.float32)[None, :]
    base = (y + x) / 2 + seed
    return np.clip(np.stack([base, base * 0.7, base * 0.4], axis=-1), 0, 255).astype(
        np.uint8
    )


def _step(seed: int) -> "tuple[np.ndarray, dict]":
    state = np.arange(12, dtype=np.float32) + seed
    return state, {name: _frame(seed + i) for i, name in enumerate(_CAMERAS)}


class TestRequestRoundTrip(unittest.TestCase):
    def test_carries_task_session_and_sequence(self):
        got = decode_request(
            encode_request(
                [_step(0)], "fold the towel", session="abc", seq=7, actions=32
            )
        )
        self.assertEqual(got["task"], "fold the towel")
        self.assertEqual(got["session"], "abc")
        self.assertEqual(got["seq"], 7)
        self.assertEqual(got["actions"], 32)

    def test_steps_keep_their_order(self):
        # Oldest first, both ways. A window handed over backwards presents the
        # policy with a history that runs the wrong way and raises nothing.
        got = decode_request(encode_request([_step(0), _step(50)], "t"))
        self.assertEqual(len(got["steps"]), 2)
        self.assertAlmostEqual(float(got["steps"][0][0][0]), 0.0)
        self.assertAlmostEqual(float(got["steps"][1][0][0]), 50.0)

    def test_frames_survive_as_rgb_uint8(self):
        state, images = _step(3)
        got = decode_request(encode_request([(state, images)], "t"))
        back = got["steps"][0][1]
        self.assertEqual(sorted(back), sorted(_CAMERAS))
        for name in _CAMERAS:
            self.assertEqual(back[name].shape, images[name].shape)
            self.assertEqual(back[name].dtype, np.uint8)
            error = np.abs(back[name].astype(float) - images[name].astype(float)).mean()
            self.assertLess(error, 2.0, f"{name} came back too far from what was sent")

    def test_channels_are_not_swapped(self):
        # JPEG is written from BGR, so a missing flip on either side would return
        # a picture whose red and blue are exchanged -- lossy compression hides
        # that from a tolerance check unless the channels differ.
        red = np.zeros((16, 16, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        got = decode_request(encode_request([(np.zeros(12), {"central": red})], "t"))
        back = got["steps"][0][1]["central"]
        self.assertGreater(int(back[:, :, 0].mean()), 200)
        self.assertLess(int(back[:, :, 2].mean()), 60)


class TestRequestRefusals(unittest.TestCase):
    def test_a_state_of_any_width_is_accepted_unless_the_width_is_pinned(self):
        message = encode_request([(np.zeros(7), {"central": _frame(0)})], "t")
        state, _images = decode_request(message)["steps"][0]
        self.assertEqual(state.shape, (7,))

    def test_a_state_that_contradicts_a_pinned_width_is_refused(self):
        with self.assertRaises(WireError):
            encode_request([(np.zeros(6), {"central": _frame(0)})], "t", action_dim=12)

    def test_steps_must_agree_on_the_state_width(self):
        # The check that survives without a pin: one message cannot describe
        # two different robots.
        with self.assertRaises(WireError):
            encode_request(
                [
                    (np.zeros(7), {"central": _frame(0)}),
                    (np.zeros(12), {"central": _frame(1)}),
                ],
                "t",
            )

    def test_an_empty_state_is_refused(self):
        with self.assertRaises(WireError):
            encode_request([(np.zeros(0), {"central": _frame(0)})], "t")

    def test_steps_must_agree_on_the_camera_set(self):
        first = (np.zeros(12), {"central": _frame(0), "wrist_camera_left": _frame(1)})
        second = (np.zeros(12), {"central": _frame(2)})
        with self.assertRaises(WireError):
            encode_request([first, second], "t")

    def test_float_frames_are_refused(self):
        with self.assertRaises(WireError):
            encode_request(
                [(np.zeros(12), {"central": _frame(0).astype(np.float32)})], "t"
            )

    def test_a_truncated_message_is_refused_not_misread(self):
        message = encode_request([_step(0)], "t")
        with self.assertRaises(WireError):
            decode_request(message[: len(message) // 2])


class TestChunkRoundTrip(unittest.TestCase):
    def test_actions_survive_exactly(self):
        actions = (np.arange(100 * 12, dtype=np.float32) / 7.0).reshape(100, 12)
        back, header = decode_chunk(
            encode_chunk(actions, seq=4, timings={"infer_s": 0.1})
        )
        np.testing.assert_array_equal(back, actions)
        self.assertEqual(header["seq"], 4)
        self.assertEqual(header["count"], 100)
        self.assertAlmostEqual(header["timings"]["infer_s"], 0.1)

    def test_a_chunk_of_any_width_is_accepted_unless_the_width_is_pinned(self):
        # This package serves more than one robot, so seven channels is a
        # single six-joint arm with a gripper and not an error. The header
        # carries the width, so the decoder never has to be told.
        actions, header = decode_chunk(encode_chunk(np.zeros((10, 7), np.float32)))
        self.assertEqual(actions.shape, (10, 7))
        self.assertEqual(header["dim"], 7)

    def test_a_chunk_that_contradicts_a_pinned_width_is_refused(self):
        with self.assertRaises(WireError):
            encode_chunk(np.zeros((10, 7), dtype=np.float32), action_dim=12)

    def test_a_chunk_with_no_channels_is_refused(self):
        with self.assertRaises(WireError):
            encode_chunk(np.zeros((10, 0), dtype=np.float32))

    def test_a_request_is_not_mistaken_for_a_chunk(self):
        with self.assertRaises(WireError):
            decode_chunk(encode_request([_step(0)], "t"))


class TestObservationWindow(unittest.TestCase):
    def test_keeps_the_last_n_oldest_first(self):
        window = ObservationWindow(list(_CAMERAS), 2)
        for seed in (0, 10, 20):
            window.push(*_step(seed))
        steps = window.steps()
        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(float(steps[0][0][0]), 10.0)
        self.assertAlmostEqual(float(steps[1][0][0]), 20.0)

    def test_pads_with_the_oldest_until_full(self):
        # The policy's own queue does exactly this on its first call, so a run
        # does not have to stall waiting for history it will have in a tick.
        window = ObservationWindow(list(_CAMERAS), 2)
        window.push(*_step(5))
        self.assertFalse(window.ready)
        steps = window.steps()
        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(float(steps[0][0][0]), 5.0)
        self.assertAlmostEqual(float(steps[1][0][0]), 5.0)

    def test_a_different_camera_set_is_refused_on_the_first_frame(self):
        window = ObservationWindow(list(_CAMERAS), 1)
        with self.assertRaises(WireError):
            window.push(np.zeros(12), {"central": _frame(0)})

    def test_ready_once_the_window_is_full(self):
        window = ObservationWindow(["central"], 2)
        window.push(np.zeros(12), {"central": _frame(0)})
        self.assertFalse(window.ready)
        window.push(np.zeros(12), {"central": _frame(1)})
        self.assertTrue(window.ready)


if __name__ == "__main__":
    unittest.main()
