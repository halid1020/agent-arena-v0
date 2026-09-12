"""The layout rule, and the schema that gives it a width.

so101_garment keeps the tests that assert ITS layout -- twelve channels, the
gripper fifth and eleventh, the EE keys LeRobot's classifier must ignore. What
is tested here is the rule those follow from, on both shapes this package has to
serve: two five-joint arms with a gripper each, and one six-joint arm with one.

The derivation is the point. The SO-101's gripper columns come out (5, 11),
which is exactly what two analysis modules used to have written down.
"""

import unittest

import numpy as np

from actoris_harena.recording.features import (
    RobotSchema,
    build_action,
    build_dataset_features,
    build_observation_state,
    fresh_limbs,
)

DUAL = RobotSchema(
    limbs=("left", "right"),
    body_joints=(
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ),
)
SINGLE = RobotSchema(
    limbs=("arm",),
    body_joints=(
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ),
)


class TestTheSchemaDerivesTheLayout(unittest.TestCase):
    def test_a_dual_five_joint_rig_is_twelve_channels(self):
        self.assertEqual(DUAL.state_dim, 12)
        self.assertEqual(DUAL.body_dof, 5)
        self.assertEqual(DUAL.limb_dof, 6)

    def test_a_single_six_joint_rig_is_seven(self):
        self.assertEqual(SINGLE.state_dim, 7)
        self.assertEqual(SINGLE.limb_dof, 7)

    def test_the_gripper_columns_are_derived_not_written_down(self):
        # (5, 11) is what analysis/phases.py and analysis/perturb.py each had as
        # a literal. It falls out of the layout instead.
        self.assertEqual(DUAL.gripper_columns, (5, 11))
        self.assertEqual(SINGLE.gripper_columns, (6,))

    def test_a_rig_with_no_gripper_has_no_gripper_column(self):
        no_jaw = RobotSchema(limbs=("arm",), body_joints=("a", "b"), gripper=False)
        self.assertEqual(no_jaw.gripper_columns, ())
        self.assertEqual(no_jaw.state_dim, 2)

    def test_channel_names_are_limb_major_with_the_gripper_last(self):
        self.assertEqual(DUAL.state_names[0], "left_shoulder_pan.pos")
        self.assertEqual(DUAL.state_names[5], "left_gripper.pos")
        self.assertEqual(DUAL.state_names[6], "right_shoulder_pan.pos")
        self.assertEqual(DUAL.state_names[11], "right_gripper.pos")

    def test_the_ee_vector_is_seven_per_limb(self):
        self.assertEqual(DUAL.ee_dim, 14)
        self.assertEqual(SINGLE.ee_dim, 7)
        self.assertEqual(SINGLE.ee_names[:4], ["arm_x", "arm_y", "arm_z", "arm_qw"])

    def test_a_schema_with_no_limbs_or_no_joints_is_refused(self):
        with self.assertRaises(ValueError):
            RobotSchema(limbs=(), body_joints=("a",))
        with self.assertRaises(ValueError):
            RobotSchema(limbs=("arm",), body_joints=())

    def test_duplicate_names_are_refused(self):
        # Two limbs called the same thing would silently overwrite each other in
        # every per-limb dict the recorder keys by name.
        with self.assertRaises(ValueError):
            RobotSchema(limbs=("arm", "arm"), body_joints=("a",))
        with self.assertRaises(ValueError):
            RobotSchema(limbs=("arm",), body_joints=("a", "a"))


class TestBuildingAFrameOnEitherShape(unittest.TestCase):
    def test_state_interleaves_joints_and_grippers_per_limb(self):
        state = build_observation_state(
            np.arange(10.0), {"left": 0.25, "right": 0.75}, DUAL
        )
        self.assertEqual(state.shape, (12,))
        self.assertAlmostEqual(float(state[5]), 0.25)
        self.assertAlmostEqual(float(state[11]), 0.75)
        # the right arm's joints follow its own five, not the left's
        self.assertAlmostEqual(float(state[6]), 5.0)

    def test_state_on_a_single_arm(self):
        state = build_observation_state(np.arange(6.0), {"arm": 0.5}, SINGLE)
        self.assertEqual(state.shape, (7,))
        self.assertAlmostEqual(float(state[6]), 0.5)

    def test_a_joint_vector_of_the_wrong_width_is_refused(self):
        with self.assertRaises(ValueError):
            build_observation_state(np.arange(6.0), {"left": 0.0, "right": 0.0}, DUAL)

    def test_a_fresh_command_becomes_the_action(self):
        state = build_observation_state(np.zeros(6), {"arm": 0.0}, SINGLE)
        cmds = {"arm": (np.full(6, 3.0), 0.9, 100.0)}
        action = build_action(state, cmds, SINGLE, now_mono=100.01)
        self.assertAlmostEqual(float(action[0]), 3.0)
        self.assertAlmostEqual(float(action[6]), 0.9)

    def test_a_stale_command_falls_back_to_that_limb_s_measured_state(self):
        # Covers a homing move and a released clutch, both of which bypass the
        # command path -- the action must not teach a spurious reach.
        state = build_observation_state(np.full(6, 7.0), {"arm": 0.1}, SINGLE)
        cmds = {"arm": (np.full(6, 3.0), 0.9, 100.0)}
        action = build_action(state, cmds, SINGLE, now_mono=105.0)
        self.assertAlmostEqual(float(action[0]), 7.0)
        self.assertAlmostEqual(float(action[6]), 0.1)

    def test_only_the_stale_limb_falls_back(self):
        state = build_observation_state(
            np.full(10, 7.0), {"left": 0.1, "right": 0.1}, DUAL
        )
        cmds = {
            "left": (np.full(5, 3.0), 0.9, 100.0),
            "right": (np.full(5, 3.0), 0.9, 50.0),
        }
        action = build_action(state, cmds, DUAL, now_mono=100.01)
        self.assertAlmostEqual(float(action[0]), 3.0)  # left, fresh
        self.assertAlmostEqual(float(action[6]), 7.0)  # right, stale
        self.assertEqual(fresh_limbs(cmds, DUAL, 100.01), {"left"})


class TestTheFeatureSpec(unittest.TestCase):
    def test_the_state_and_action_features_carry_the_schema_s_names(self):
        feats = build_dataset_features([("wrist", 480, 640)], SINGLE)
        self.assertEqual(feats["observation.state"]["shape"], (7,))
        self.assertEqual(feats["action"]["names"], SINGLE.state_names)

    def test_a_camera_becomes_a_video_feature(self):
        feats = build_dataset_features([("wrist", 480, 640)], SINGLE)
        self.assertEqual(feats["observation.images.wrist"]["dtype"], "video")
        self.assertEqual(feats["observation.images.wrist"]["shape"], (480, 640, 3))

    def test_the_ee_features_are_off_unless_asked_for(self):
        self.assertNotIn("ee_pose", build_dataset_features([], SINGLE))
        with_ee = build_dataset_features([], SINGLE, include_ee=True)
        self.assertEqual(with_ee["ee_pose"]["shape"], (7,))
        self.assertEqual(with_ee["ee_target"]["shape"], (7,))

    def test_the_phase_flag_is_off_unless_asked_for(self):
        # The sim collector leaves it off, so its schema is unchanged.
        self.assertNotIn("teleop_active", build_dataset_features([], DUAL))
        self.assertIn(
            "teleop_active", build_dataset_features([], DUAL, include_phase=True)
        )


if __name__ == "__main__":
    unittest.main()
