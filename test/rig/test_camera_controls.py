"""Unit tests for applying per-camera image controls.

The ordering requirement here is invisible in the code: an exposure value set
while the camera is still choosing its own is simply ignored, so automatic mode
has to be switched off first.

The other half of the original test -- the tuner's YAML round-trip -- stays in
so101_garment, because tool/tune_cameras.py is a per-rig tool and not part of
this package.

Run:  python -m unittest test.rig.test_camera_controls
"""

import unittest

import cv2  # type: ignore[import]

from actoris_harena.recording.camera_controls import (
    CONTROL_NAMES,
    EXPOSURE_MANUAL,
    apply_controls,
    control_yaml_line,
)


class _FakeCap:
    def __init__(self):
        self.calls = []

    def set(self, prop, value):
        self.calls.append((prop, value))
        return True


class TestApplyControls(unittest.TestCase):
    def test_exposure_leaves_automatic_mode_before_the_value_is_sent(self):
        # Order is the whole point: a value set while the camera is still
        # choosing its own exposure has no effect at all.
        cap = _FakeCap()
        apply_controls(cap, {"exposure": 300})
        props = [p for p, _ in cap.calls]
        self.assertLess(
            props.index(cv2.CAP_PROP_AUTO_EXPOSURE), props.index(cv2.CAP_PROP_EXPOSURE)
        )
        self.assertEqual(cap.calls[0], (cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_MANUAL))

    def test_none_leaves_a_control_alone(self):
        cap = _FakeCap()
        applied = apply_controls(cap, {name: None for name in CONTROL_NAMES})
        self.assertEqual(applied, [])
        self.assertEqual(cap.calls, [])

    def test_zero_is_a_real_value_not_an_absent_one(self):
        # Gain and brightness legitimately take 0, which is why absence is None.
        cap = _FakeCap()
        self.assertEqual(apply_controls(cap, {"gain": 0}), ["gain"])

    def test_other_controls_do_not_touch_the_exposure_mode(self):
        cap = _FakeCap()
        apply_controls(cap, {"gain": 5, "brightness": 10})
        self.assertNotIn(cv2.CAP_PROP_AUTO_EXPOSURE, [p for p, _ in cap.calls])

    def test_yaml_line_renders_absence_as_null(self):
        self.assertEqual(control_yaml_line("gain", None).strip(), "gain: null")
        self.assertEqual(control_yaml_line("gain", 30.0).strip(), "gain: 30")
