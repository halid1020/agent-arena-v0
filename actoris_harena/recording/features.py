"""The feature schema a rig records, and the per-frame builders.

No hardware, no dataset, no threads -- just the mapping from a rig's measured
state to a LeRobotDataset feature specification and per-frame dicts.

WHAT A RIG HAS TO SAY. This module used to hold three constants describing one
bench: five named body joints, two sides, and therefore a twelve-channel state.
A single six-joint arm with one gripper is seven channels, so those three are now
a :class:`RobotSchema` the caller passes in. Everything else -- the layout rule,
the freshness rule, the EE convention, the frame assembly -- is the same on any
rig and stays here.

THE LAYOUT RULE, unchanged and load-bearing: ``observation.state`` is, per limb
in order, that limb's body joints in URDF order followed by its gripper open
fraction (0 closed, 1 open). ``action`` is the joint-space command actually sent,
in the SAME layout and units, used only while fresh (age < ``ACTION_FRESH_S``);
a stale limb falls back to its own measured state, which covers homing moves and
a released clutch, both of which bypass the command path. Cameras become
``observation.images.<name>`` video features.

The whole internal pipeline is URDF degrees; converting to whatever the motors
want is confined to each rig's bus boundary, and pi0.5 normalises with dataset
statistics regardless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from actoris_harena.sync import mat_to_quat


@dataclass(frozen=True)
class RobotSchema:
    """How wide a rig's state and action vectors are, and what each channel is.

    ``limbs`` is ordered and its order is the vector's order: ``("left",
    "right")`` on the dual SO-101, ``("arm",)`` on a single one. ``body_joints``
    is one limb's actuated joints in URDF order -- every limb is assumed alike,
    which is true of both rigs here and is the thing to revisit first if it ever
    is not.
    """

    limbs: "tuple[str, ...]"
    body_joints: "tuple[str, ...]"
    gripper: bool = True

    def __post_init__(self) -> None:
        if not self.limbs:
            raise ValueError("a rig has at least one limb")
        if not self.body_joints:
            raise ValueError("a limb has at least one actuated joint")
        if len(set(self.limbs)) != len(self.limbs):
            raise ValueError(f"limb names must be unique: {self.limbs}")
        if len(set(self.body_joints)) != len(self.body_joints):
            raise ValueError(f"joint names must be unique: {self.body_joints}")

    @property
    def body_dof(self) -> int:
        """Actuated joints per limb, excluding the gripper."""
        return len(self.body_joints)

    @property
    def limb_dof(self) -> int:
        """Channels one limb occupies: its joints, plus a gripper if it has one."""
        return self.body_dof + (1 if self.gripper else 0)

    @property
    def state_dim(self) -> int:
        return self.limb_dof * len(self.limbs)

    @property
    def state_names(self) -> "list[str]":
        """The channel names, limb-major, e.g. ``left_shoulder_pan.pos``."""
        joints = (*self.body_joints, "gripper") if self.gripper else self.body_joints
        return [f"{limb}_{joint}.pos" for limb in self.limbs for joint in joints]

    @property
    def gripper_columns(self) -> "tuple[int, ...]":
        """Which state/action columns are gripper commands.

        This is what ``actoris_harena.action_layout`` wants declared, and it is
        derived here rather than written down twice -- on the dual SO-101 it
        comes out (5, 11), which is what those two analyses had hardcoded.
        """
        if not self.gripper:
            return ()
        return tuple(i * self.limb_dof + self.body_dof for i in range(len(self.limbs)))

    @property
    def ee_names(self) -> "list[str]":
        return [f"{limb}_{c}" for limb in self.limbs for c in _EE_COMPONENTS]

    @property
    def ee_dim(self) -> int:
        return len(self.ee_names)


# A command is used as the action only while fresher than this (seconds). The
# joint threads write at ~100 Hz, so 30 ms comfortably admits the latest write.
ACTION_FRESH_S = 0.030

# Non-policy phase flag recorded per frame. It is a bare top-level key (not
# under ``observation.`` and not ``action``), so LeRobot's feature classifier
# ignores it for policy input/output (feature_utils.dataset_to_policy_features)
# while it stays queryable in the dataset — used to mask non-teleop frames
# (e.g. homing moves, where the action falls back to the measured state) at
# train time. 1.0 = teleoperation active, 0.0 = not.
TELEOP_ACTIVE_KEY = "teleop_active"

# End-effector pose channels per side: 3 position + 4 quaternion (w, x, y, z),
# expressed in that arm's OWN base frame. ``ee_pose`` is the measured pose;
# ``ee_target`` is the projected+constrained TARGET pose (the label an EE-space
# policy predicts). BOTH use NEUTRAL top-level keys (not ``observation.`` /
# ``action``) so LeRobot's feature classifier ignores them for the default
# joint-space policy — its state/action are unchanged by their presence. An
# EE-space experiment remaps ``ee_target`` → action (and optionally ``ee_pose``
# → an observation) explicitly, so both policies train from the SAME episodes
# with no re-collection. Recorded only in quest/IK mode, where EE targets exist.
_EE_COMPONENTS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
OBS_EE_KEY = "ee_pose"
ACTION_EE_KEY = "ee_target"


def pose_to_vec7(pose_4x4: np.ndarray) -> np.ndarray:
    """4×4 homogeneous pose → (7,) float32 [x, y, z, qw, qx, qy, qz]."""
    p = np.asarray(pose_4x4, dtype=np.float64)
    quat = mat_to_quat(p[:3, :3])  # (w, x, y, z)
    return np.array([*p[:3, 3], *quat], dtype=np.float32)


def build_dataset_features(
    camera_specs: list[tuple[str, int, int]],
    schema: RobotSchema,
    include_phase: bool = False,
    include_ee: bool = False,
) -> dict[str, dict]:
    """Return the LeRobotDataset feature spec for the enabled streams.

    ``camera_specs`` is a list of ``(name, height, width)`` for each ENABLED
    camera; each becomes an ``observation.images.<name>`` video feature. When
    ``include_phase`` is set, a maskable ``teleop_active`` scalar is added (the
    real teleop recorder opts in; the sim oracle collector leaves it off so its
    schema is unchanged). When ``include_ee`` is set, the EE-space state
    (``observation.ee_pose``) and target (``action_ee``) features are added so
    an EE-space policy trains from the same episodes as the joint-space one.
    """
    names = schema.state_names
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": list(names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": list(names),
        },
    }
    if include_ee:
        ee_names = schema.ee_names
        features[OBS_EE_KEY] = {
            "dtype": "float32",
            "shape": (len(ee_names),),
            "names": list(ee_names),
        }
        features[ACTION_EE_KEY] = {
            "dtype": "float32",
            "shape": (len(ee_names),),
            "names": list(ee_names),
        }
    for name, height, width in camera_specs:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    if include_phase:
        features[TELEOP_ACTIVE_KEY] = {
            "dtype": "float32",
            "shape": (1,),
            "names": [TELEOP_ACTIVE_KEY],
        }
    return features


def fresh_limbs(
    last_commands: dict[str, tuple[np.ndarray | None, float | None, float | None]],
    schema: RobotSchema,
    now_mono: float,
    fresh_s: float = ACTION_FRESH_S,
) -> set[str]:
    """Return the sides whose last command is fresh enough to be the action.

    The recorder uses this to tally how often the action fell back to the
    measured state (a stale/missing command), which flags frames that teach a
    spurious "hold" and should be masked or excluded at train time.
    """
    out: set[str] = set()
    for limb in schema.limbs:
        _urdf, _grip, t_mono = last_commands[limb]
        if t_mono is not None and (now_mono - t_mono) < fresh_s:
            out.add(limb)
    return out


def build_observation_state(
    measured_joints: np.ndarray,
    gripper_open: dict[str, float],
    schema: RobotSchema,
) -> np.ndarray:
    """Assemble the (12,) float32 observation.state vector.

    ``measured_joints_10`` is the 10-DOF URDF-degree joint array (left five,
    right five); ``gripper_open`` maps each side to its 0-1 open fraction.
    """
    joints = np.asarray(measured_joints, dtype=np.float64)
    want = schema.body_dof * len(schema.limbs)
    if joints.shape != (want,):
        raise ValueError(
            f"measured_joints must have shape ({want},), got {joints.shape}"
        )
    out = np.empty(schema.state_dim, dtype=np.float32)
    dof, limb_dof = schema.body_dof, schema.limb_dof
    for i, limb in enumerate(schema.limbs):
        base = i * limb_dof
        out[base : base + dof] = joints[i * dof : (i + 1) * dof]
        if schema.gripper:
            out[base + dof] = gripper_open[limb]
    return out


def build_action(
    observation_state: np.ndarray,
    last_commands: dict[str, tuple[np.ndarray | None, float | None, float | None]],
    schema: RobotSchema,
    now_mono: float,
    fresh_s: float = ACTION_FRESH_S,
) -> np.ndarray:
    """Assemble the (12,) float32 action vector from the last sent commands.

    ``last_commands[side]`` is ``(urdf_deg_5, gripper_open, t_mono)`` as
    returned by ``DualDataManager.get_last_sent_command``. Each side whose
    command is missing or older than ``fresh_s`` falls back to that side's
    slice of ``observation_state`` (deterministic: covers HOMING and a
    released clutch, where no fresh command exists).
    """
    action = np.asarray(observation_state, dtype=np.float32).copy()
    dof, limb_dof = schema.body_dof, schema.limb_dof
    for i, limb in enumerate(schema.limbs):
        base = i * limb_dof
        urdf_deg, gripper_open, t_mono = last_commands[limb]
        if (
            urdf_deg is not None
            and gripper_open is not None
            and t_mono is not None
            and (now_mono - t_mono) < fresh_s
        ):
            action[base : base + dof] = np.asarray(urdf_deg, dtype=np.float32)
            if schema.gripper:
                action[base + dof] = np.float32(gripper_open)
    return action


def assemble_frame(
    observation_state: np.ndarray,
    action: np.ndarray,
    images: dict[str, np.ndarray],
    task: str,
    teleop_active: bool | None = None,
    ee_pose: np.ndarray | None = None,
    ee_target: np.ndarray | None = None,
) -> dict:
    """Build the LeRobotDataset frame dict (features + task, no bookkeeping keys).

    ``images`` maps each enabled camera name to its RGB (H, W, 3) uint8 array.
    ``teleop_active`` records whether teleoperation drove this frame (for
    train-time masking); pass ``None`` (the default, used by the sim collector)
    to omit the key so the frame matches a feature spec built without
    ``include_phase``. ``ee_pose`` / ``ee_target`` are the (14,) measured and
    projected+constrained EE vectors; pass ``None`` to omit them (feature spec
    built without ``include_ee``). Never adds timestamp/frame_index — LeRobot
    derives those from fps.
    """
    frame: dict = {
        "observation.state": np.asarray(observation_state, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    if teleop_active is not None:
        frame[TELEOP_ACTIVE_KEY] = np.array(
            [1.0 if teleop_active else 0.0], dtype=np.float32
        )
    if ee_pose is not None:
        frame[OBS_EE_KEY] = np.asarray(ee_pose, dtype=np.float32)
    if ee_target is not None:
        frame[ACTION_EE_KEY] = np.asarray(ee_target, dtype=np.float32)
    for name, rgb in images.items():
        frame[f"observation.images.{name}"] = rgb
    return frame
