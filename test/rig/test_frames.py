"""The three methods a capture thread needs, and a store that has them.

The point of FramePublisher is that a capture thread stops depending on one
rig's blackboard. So the test that matters is structural: an object with these
three methods is enough, whatever else it is or is not.
"""

import threading
import unittest

import numpy as np

from actoris_harena.recording.frames import FramePublisher, FrameStore


def _rgb(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


class TestFrameStore(unittest.TestCase):
    def test_a_published_frame_reads_back(self):
        store = FrameStore()
        store.set_rgb_image(_rgb(7), "wrist", t_capture=1.5)
        got = store.get_rgb_image("wrist")
        self.assertIsNotNone(got)
        self.assertEqual(int(got[0, 0, 0]), 7)

    def test_an_unknown_name_reads_as_none_rather_than_raising(self):
        # A monitor asks for every configured camera, including one whose
        # device never opened; that has to be an empty panel, not a crash.
        self.assertIsNone(FrameStore().get_rgb_image("nothing here"))

    def test_the_newest_frame_wins(self):
        store = FrameStore()
        store.set_rgb_image(_rgb(1), "wrist")
        store.set_rgb_image(_rgb(2), "wrist")
        self.assertEqual(int(store.get_rgb_image("wrist")[0, 0, 0]), 2)

    def test_depth_is_a_separate_namespace_from_colour(self):
        # A RealSense publishes both under the SAME stream name, so they must
        # not overwrite each other.
        store = FrameStore()
        store.set_rgb_image(_rgb(3), "central")
        store.set_depth_image(np.full((4, 4), 900, dtype=np.uint16), "central")
        self.assertEqual(int(store.get_rgb_image("central")[0, 0, 0]), 3)
        self.assertEqual(int(store.get_depth_image("central")[0, 0]), 900)

    def test_camera_names_are_the_ones_published_into(self):
        store = FrameStore()
        store.set_rgb_image(_rgb(1), "b")
        store.set_rgb_image(_rgb(1), "a")
        self.assertEqual(store.rgb_camera_names(), ["a", "b"])

    def test_shutdown_is_off_until_it_is_asked_for(self):
        store = FrameStore()
        self.assertFalse(store.is_shutdown_requested())
        store.request_shutdown()
        self.assertTrue(store.is_shutdown_requested())

    def test_concurrent_writers_do_not_lose_a_name(self):
        store = FrameStore()

        def publish(name):
            for _ in range(50):
                store.set_rgb_image(_rgb(1), name)

        threads = [threading.Thread(target=publish, args=(f"c{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(store.rgb_camera_names(), [f"c{i}" for i in range(8)])


class TestTheContractIsStructural(unittest.TestCase):
    def test_frame_store_satisfies_the_protocol(self):
        self.assertIsInstance(FrameStore(), FramePublisher)

    def test_any_object_with_the_three_methods_satisfies_it(self):
        class Minimal:
            def set_rgb_image(self, rgb, name, t_capture=None):
                pass

            def set_depth_image(self, depth16, name, t_capture=None):
                pass

            def is_shutdown_requested(self):
                return False

        # Nothing inherits, nothing registers. This is what lets the dual
        # SO-101's DualDataManager and a single-arm rig's own store both work
        # with the same capture threads.
        self.assertIsInstance(Minimal(), FramePublisher)

    def test_an_object_missing_one_does_not(self):
        class NoDepth:
            def set_rgb_image(self, rgb, name, t_capture=None):
                pass

            def is_shutdown_requested(self):
                return False

        self.assertNotIsInstance(NoDepth(), FramePublisher)


if __name__ == "__main__":
    unittest.main()
