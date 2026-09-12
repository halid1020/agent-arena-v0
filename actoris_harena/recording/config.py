"""Loading and strictly validating a rig's recording config.

Strict on purpose, and loudly. A missing key, an unknown key or a missing file
raises an error naming the offending key, because the alternative -- a camera
silently not recorded, a typo'd `exposure` quietly ignored -- is a dataset that
looks right and is not, discovered hours into a training run.

The SCHEMA is shared and the FILE is not. Every rig's recording.yaml has a
``dataset`` section, a ``sidecar`` one, a map of named cameras and optionally a
RealSense and an audio cue; what it puts in them -- which camera names, which
device paths, which fps -- is that rig's, and lives in its own repo.

The teleoperation half of this module (the shared IK parameters and the
per-method ones) stayed in so101_garment: those schemas describe a two-handed
Quest mapping onto two five-joint arms, which is not a shared shape.
"""

from pathlib import Path

import yaml  # type: ignore[import]

from actoris_harena.recording.camera_controls import CONTROL_NAMES

# Frozen schema for recording.yaml. The top-level sections and their keys are
# fixed; the "cameras" section is a map of arbitrary stream names, each of whose
# value must match _CAMERA_SCHEMA exactly. Guarded by
# test/unit/test_recording_config.py.
# Where this rig's recording.yaml is. Declared, not derived: the old module
# computed it from its own location, which named the rig's conf/ only while it
# lived there.
_RECORDING_PATH: "Path | None" = None


def set_recording_config_path(path: "Path | str") -> None:
    """Declare where this rig's recording.yaml lives."""
    global _RECORDING_PATH
    _RECORDING_PATH = Path(path)


def recording_config_path() -> Path:
    """The declared recording config, or a refusal naming the fix."""
    if _RECORDING_PATH is None:
        raise RuntimeError(
            "no recording.yaml has been declared. A rig calls "
            "actoris_harena.recording.config.set_recording_config_path() once "
            "(so101_garment does it in common/rig_profile.py), or a caller "
            "passes an explicit path."
        )
    return _RECORDING_PATH


_RECORDING_SCHEMA: dict[str, frozenset[str]] = {
    "dataset": frozenset({"fps", "image_writer_threads_per_camera", "robot_type"}),
    "sidecar": frozenset({"enabled", "rate_hz", "include_hw_frame_goal"}),
}
_CAMERA_SCHEMA: frozenset[str] = frozenset(
    {"enabled", "device", "width", "height", "fps", "rotate180"}
)
# Default capture pixel format. MJPG (compressed on the wire) is required, not a
# preference: several 640x480@30 streams in an uncompressed format (YUYV needs
# ~18 MB/s EACH) exceed what the shared USB controllers deliver, which starves
# the wrist cameras to ~10-15 fps and eventually drops the device mid-episode.
_DEFAULT_CAMERA_FOURCC = "MJPG"
# Optional per-camera image controls (actoris_harena.recording.camera_controls). Each
# defaults to None: leave the camera's own setting alone, so an existing
# recording.yaml behaves exactly as it did. None rather than 0 because 0 is a
# legitimate value for most of them.
#
# Exposure is the one that is not merely cosmetic: it caps frame rate, since no
# camera delivers frames faster than it exposes them. MEASURED on the wrist
# cameras -- every exposure up to 300 sustains 27.4 fps, 400 gives 22.8, and 500
# (what automatic exposure chose under collection lighting) gives 18.2, which is
# the rate those streams had been recording at. Below 300, brightness is free.
_CAMERA_DEFAULTS: dict[str, object] = {
    "fourcc": _DEFAULT_CAMERA_FOURCC,
    **{name: None for name in CONTROL_NAMES},
}
# Optional central RGB-D (RealSense) section. Absent in older maps (validated
# only when present, so existing recording.yaml files stay valid).
_REALSENSE_SCHEMA: frozenset[str] = frozenset(
    {
        "enabled",
        "serial",
        "width",
        "height",
        "fps",
        "align_to_color",
        "rgb_name",
        "depth_name",
        "lock_auto_exposure",
    }
)
# Optional audible record-cue section. Absent in older maps (validated only when
# present, so existing recording.yaml files stay valid).
_AUDIO_SCHEMA: frozenset[str] = frozenset({"enabled", "start_sound", "stop_sound"})


def _load_yaml_strict(path: Path) -> dict:
    """Load a YAML file, raising a clear error if it is missing or not a mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Required teleop config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a top-level mapping, got {type(data).__name__}"
        )
    return data


def _validate_keys(
    path: Path,
    where: str,
    present: object,
    expected: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    """Raise if ``present`` (a mapping) is not exactly keyed by ``expected``.

    Keys in ``optional`` are accepted but not required, so a config file written
    before an optional key existed stays valid.
    """
    if not isinstance(present, dict):
        raise ValueError(f"{path}: section '{where}' must be a mapping")
    keys = set(present)
    missing = expected - keys
    unknown = keys - expected - optional
    if missing:
        raise ValueError(
            f"{path}: {where} is missing required key(s): {sorted(missing)}"
        )
    if unknown:
        raise ValueError(f"{path}: {where} has unknown key(s): {sorted(unknown)}")


def load_recording_config(path: str | None = None) -> dict:
    """Load and strictly validate the data-collection (recording) config.

    Validates the top-level sections (``dataset``, ``sidecar``, ``cameras``)
    and every key within each. The ``cameras`` section is a map of arbitrary
    stream names, each of which must be keyed exactly by the camera schema.
    Same loud-failure regime as ``load_teleop_shared``: any missing file,
    missing key, or unknown key raises a clear error naming the offending key.
    """
    cfg_path = Path(path) if path is not None else recording_config_path()
    data = _load_yaml_strict(cfg_path)
    # Required top-level sections plus an OPTIONAL ``realsense`` one: older maps
    # (no central RGB-D camera) omit it and must still load.
    required_top = frozenset(_RECORDING_SCHEMA) | {"cameras"}
    optional_top = {"realsense", "audio"}
    keys = set(data)
    missing = required_top - keys
    unknown = keys - required_top - optional_top
    if missing:
        raise ValueError(
            f"{cfg_path}: top-level is missing required key(s): {sorted(missing)}"
        )
    if unknown:
        raise ValueError(f"{cfg_path}: top-level has unknown key(s): {sorted(unknown)}")
    for section, expected in _RECORDING_SCHEMA.items():
        _validate_keys(cfg_path, section, data[section], expected)
    if "realsense" in data:
        _validate_keys(cfg_path, "realsense", data["realsense"], _REALSENSE_SCHEMA)
    if "audio" in data:
        _validate_keys(cfg_path, "audio", data["audio"], _AUDIO_SCHEMA)

    cameras = data["cameras"]
    if not isinstance(cameras, dict):
        raise ValueError(f"{cfg_path}: section 'cameras' must be a mapping")
    if not cameras:
        raise ValueError(f"{cfg_path}: section 'cameras' must not be empty")
    optional_cam = frozenset(_CAMERA_DEFAULTS)
    for cam_name, cam_cfg in cameras.items():
        _validate_keys(
            cfg_path, f"cameras.{cam_name}", cam_cfg, _CAMERA_SCHEMA, optional_cam
        )
        for key, default in _CAMERA_DEFAULTS.items():
            cam_cfg.setdefault(key, default)
        # A -1 device is the file's own marker for "nothing wired here", so it is
        # only meaningful on a disabled stream. Enabled, it used to be accepted
        # and then failed much later as an opaque camera-open error at session
        # start; the config is the place that knows it is wrong.
        if cam_cfg.get("enabled") and cam_cfg.get("device") == -1:
            raise ValueError(
                f"{cfg_path}: camera '{cam_name}' is enabled but has device -1 "
                "— set its /dev/videoN index (or assign it a stable node in "
                "src/conf/sensor_map.yaml), or disable the stream"
            )
    return data
