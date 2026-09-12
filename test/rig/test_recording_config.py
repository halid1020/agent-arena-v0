"""The recording config schema: strict, and loud about why.

so101_garment keeps its own test of the checked-in recording.yaml -- which
cameras it names, which have exposure pinned, which controls resolve. Those are
facts about that bench. What is tested here is the SCHEMA those facts are
checked against, with synthetic configs, because that is what this package owns
and what a second rig's file will be validated by.

The regime under test is loud failure. A missing key, an unknown key or a
missing file raises an error NAMING the offending key, because the alternative
is a camera silently not recorded or a typo'd `exposure` quietly ignored -- a
dataset that looks right and is not, discovered hours into a training run.
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from actoris_harena.recording.config import (
    load_recording_config,
    recording_config_path,
    set_recording_config_path,
)

_DATASET = {"fps": 30, "image_writer_threads_per_camera": 4, "robot_type": "example"}
_SIDECAR = {"enabled": True, "rate_hz": 100, "include_hw_frame_goal": False}
_CAMERA = {
    "enabled": True,
    "device": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "rotate180": False,
}


def _config(**overrides):
    cfg = {
        "dataset": dict(_DATASET),
        "sidecar": dict(_SIDECAR),
        "cameras": {"wrist": dict(_CAMERA)},
    }
    cfg.update(overrides)
    return cfg


def _written(cfg) -> str:
    path = Path(tempfile.mkdtemp()) / "recording.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


class TestTheSchemaIsStrict(unittest.TestCase):
    def test_a_minimal_config_loads(self):
        cfg = load_recording_config(_written(_config()))
        self.assertEqual(cfg["dataset"]["fps"], 30)
        self.assertEqual(sorted(cfg["cameras"]), ["wrist"])

    def test_a_camera_name_is_arbitrary(self):
        # The schema fixes the KEYS of a camera, never its name: a rig with one
        # `wrist` is as valid as one with `central` and four fingertips.
        cfg = _config(cameras={"anything_at_all": dict(_CAMERA)})
        self.assertIn(
            "anything_at_all", load_recording_config(_written(cfg))["cameras"]
        )

    def test_an_unknown_top_level_key_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            load_recording_config(_written(_config(nonsense={})))
        self.assertIn("nonsense", str(caught.exception))

    def test_an_unknown_camera_key_is_refused_by_name(self):
        bad = dict(_CAMERA, expsoure=300)  # the typo this exists to catch
        with self.assertRaises(ValueError) as caught:
            load_recording_config(_written(_config(cameras={"wrist": bad})))
        self.assertIn("expsoure", str(caught.exception))

    def test_a_missing_required_key_is_refused(self):
        thin = {k: v for k, v in _CAMERA.items() if k != "width"}
        with self.assertRaises(ValueError):
            load_recording_config(_written(_config(cameras={"wrist": thin})))

    def test_no_cameras_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            load_recording_config(_written(_config(cameras={})))

    def test_a_missing_file_names_the_path(self):
        with self.assertRaises(Exception) as caught:
            load_recording_config("/nonexistent/recording.yaml")
        self.assertIn("recording.yaml", str(caught.exception))


class TestOptionalSections(unittest.TestCase):
    def test_realsense_may_be_absent(self):
        self.assertNotIn("realsense", load_recording_config(_written(_config())))

    def test_a_realsense_is_validated_when_present(self):
        rs = {
            "enabled": True,
            "rgb_name": "wrist",
            "depth_name": "wrist_depth",
            "width": 640,
            "height": 480,
            "fps": 30,
            "serial": "",
            "align_to_color": True,
            "lock_auto_exposure": True,
        }
        cfg = load_recording_config(_written(_config(realsense=rs)))
        # rgb_name is the stream name, with no rig default -- which is what
        # lets the same class be a wrist camera on one rig and a central one
        # on another.
        self.assertEqual(cfg["realsense"]["rgb_name"], "wrist")

    def test_an_unknown_realsense_key_is_refused(self):
        with self.assertRaises(ValueError):
            load_recording_config(_written(_config(realsense={"enabled": True})))


class TestThePathIsDeclared(unittest.TestCase):
    def test_asking_before_declaring_refuses_and_names_the_fix(self):
        import actoris_harena.recording.config as mod

        before = mod._RECORDING_PATH
        mod._RECORDING_PATH = None
        try:
            with self.assertRaises(RuntimeError) as caught:
                recording_config_path()
            self.assertIn("set_recording_config_path", str(caught.exception))
        finally:
            mod._RECORDING_PATH = before

    def test_a_declared_path_is_what_gets_read(self):
        import actoris_harena.recording.config as mod

        before = mod._RECORDING_PATH
        try:
            written = _written(_config())
            set_recording_config_path(written)
            self.assertEqual(recording_config_path(), Path(written))
            self.assertEqual(load_recording_config()["dataset"]["fps"], 30)
        finally:
            mod._RECORDING_PATH = before


if __name__ == "__main__":
    unittest.main()
