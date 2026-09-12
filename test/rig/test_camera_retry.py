"""Unit tests for how a camera stream retries, and when it stops retrying.

The behaviour under test is what turned a bandwidth-starved camera into an
endless ``read failed (N so far); reopening`` loop that hammered the USB bus the
other cameras were sharing: the first retry is free (a bumped cable must not cost
half a second of episode), everything after it backs off, and a stream that has
NEVER delivered a frame is eventually given up on rather than retried for ever.
"""

import unittest

from actoris_harena.recording.cameras import (
    _REOPEN_BACKOFF_S,
    _STARVE_GIVE_UP_ATTEMPTS,
    CameraCapture,
    reopen_delay,
)


class TestReopenDelay(unittest.TestCase):
    def test_the_first_retry_is_immediate(self):
        # A device that dropped off the bus is usually back within a few hundred
        # milliseconds, and every millisecond waited is episode not recorded.
        self.assertEqual(reopen_delay(1), 0.0)

    def test_it_backs_off_after_that(self):
        delays = [reopen_delay(n) for n in range(1, len(_REOPEN_BACKOFF_S) + 1)]
        self.assertEqual(delays, list(_REOPEN_BACKOFF_S))
        self.assertEqual(sorted(delays), delays)

    def test_it_stops_growing_at_the_ceiling(self):
        ceiling = _REOPEN_BACKOFF_S[-1]
        for n in (len(_REOPEN_BACKOFF_S), 50, 5000):
            self.assertEqual(reopen_delay(n), ceiling)

    def test_no_failures_means_no_delay(self):
        self.assertEqual(reopen_delay(0), 0.0)
        self.assertEqual(reopen_delay(-3), 0.0)


class TestGiveUpIfStarved(unittest.TestCase):
    def _camera(self):
        return CameraCapture(
            name="left_arm_left_gripper",
            device="/dev/v4l/by-path/pci-0000:05:00.4-usb-0:1.1.2:1.0-video-index0",
            width=640,
            height=480,
            fps=30,
            rotate180=False,
        )

    def test_a_new_camera_is_not_starved(self):
        self.assertFalse(self._camera().starved)

    def test_it_holds_on_below_the_threshold(self):
        cam = self._camera()
        cam._give_up_if_starved(_STARVE_GIVE_UP_ATTEMPTS - 1)
        self.assertFalse(cam.starved)

    def test_a_stream_that_never_delivered_is_given_up_on(self):
        cam = self._camera()
        cam._give_up_if_starved(_STARVE_GIVE_UP_ATTEMPTS)
        self.assertTrue(cam.starved)

    def test_a_camera_that_once_worked_keeps_retrying(self):
        # This is the difference between a bus that refused a stream and a cable
        # somebody knocked out: the second can be plugged back in, so it is
        # retried for as long as the session lasts.
        cam = self._camera()
        cam._last_frame_mono = 123.0
        cam._give_up_if_starved(_STARVE_GIVE_UP_ATTEMPTS * 10)
        self.assertFalse(cam.starved)


class TestStopOwnership(unittest.TestCase):
    def test_stop_releases_a_handle_no_thread_ever_owned(self):
        # open() is used on its own as a readiness check, with no capture thread
        # behind it; that handle still has to be released by stop().
        cam = CameraCapture("c", 0, 640, 480, 30, False)
        released = []

        class _Cap:
            def release(self):
                released.append(True)

        cam._cap = _Cap()
        cam.stop()
        self.assertEqual(released, [True])
        self.assertIsNone(cam._cap)

    def test_stop_never_releases_a_handle_a_live_thread_is_reading(self):
        # Releasing a VideoCapture from here while the capture thread is inside a
        # blocking read aborts the process -- that double release is what turned
        # Ctrl+C on the console into a core dump.
        cam = CameraCapture("c", 0, 640, 480, 30, False)
        released = []

        class _Cap:
            def release(self):
                released.append(True)

        class _WedgedThread:
            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

        cam._cap = _Cap()
        cam._thread = _WedgedThread()
        cam.stop()
        self.assertEqual(released, [])


if __name__ == "__main__":
    unittest.main()
