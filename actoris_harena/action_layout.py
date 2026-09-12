"""Which columns of an action vector are grippers. Declared, never guessed.

Two analyses need this and neither can work it out. ``analysis.phases`` reads
the gripper channels to say when a grasp closes; ``analysis.perturb`` EXCLUDES
them when it occludes a stream, because perturbing a gripper command measures
something other than what the study is asking.

Both used to hold the literal ``(5, 11)`` -- the dual SO-101's layout, five body
joints and a gripper per arm. A single six-joint arm with one gripper is
``(6,)``, and the failure mode of a wrong answer is the reason this is not a
default: an empty guess would quietly let ``perturb`` occlude the gripper
channels and report a number nobody could tell was wrong, while a stale ``(5,
11)`` would occlude a joint on a seven-wide action. So asking before declaring
raises.

When the recorder's ``RobotSchema`` moves here, this becomes derived from it
rather than declared beside it -- the layout is already ``[body..., gripper]``
per limb -- and this module becomes its accessor.
"""

_GRIPPER_COLUMNS: "tuple[int, ...] | None" = None


def set_gripper_columns(columns: "tuple[int, ...]") -> None:
    """Declare which action columns are gripper commands."""
    global _GRIPPER_COLUMNS
    _GRIPPER_COLUMNS = tuple(int(c) for c in columns)


def gripper_columns() -> "tuple[int, ...]":
    """The declared gripper columns, or a refusal naming the fix."""
    if _GRIPPER_COLUMNS is None:
        raise RuntimeError(
            "no gripper columns have been declared. A rig calls "
            "actoris_harena.action_layout.set_gripper_columns() once "
            "(so101_garment does it in common/rig_profile.py), or a caller "
            "passes columns= explicitly."
        )
    return _GRIPPER_COLUMNS
