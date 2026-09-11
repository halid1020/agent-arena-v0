"""Per-camera image controls, and the one that is really a frame-rate control.

A UVC camera exposes a handful of image settings, and on this rig they are not
merely a matter of taste. Exposure sets an upper bound on frame rate, because no
camera can deliver frames faster than it exposes them: an exposure longer than
the frame period forces the sensor to a slower one. Left automatic it lengthens
exactly when the scene is dim, which for a wrist camera looking down at a close,
arm-shadowed workspace is most of the time -- so the capture rate ends up a
property of the room rather than of the configuration. Measured on our wrist
cameras: any exposure up to 300 sustains 27.4 fps, 400 gives 22.8, and 500 --
which is what automatic exposure chose under collection lighting -- gives 18.2.

The practical consequence is the ordering below 300: brightness there is free.
Every value from 20 to 300 measured the same rate, so an over-exposed stream can
be darkened at no cost, and gain can lift a dark one without touching exposure at
all. That is why these are tunable per camera rather than fixed once: two wrist
cameras on the same rig are lit differently and want different values.

``None`` means "leave the camera's own setting alone", for every control. It has
to be None rather than zero because zero is a legitimate value for most of these.

Pure and unit-tested: the caller supplies a mapping and an object with ``set``.
"""

from __future__ import annotations

from typing import Any, Mapping

# Control name -> the OpenCV property id it maps to. Names are the keys used in
# recording.yaml, chosen to read as what an operator is adjusting.
CONTROL_PROPERTIES: "dict[str, str]" = {
    "exposure": "CAP_PROP_EXPOSURE",
    "gain": "CAP_PROP_GAIN",
    "brightness": "CAP_PROP_BRIGHTNESS",
    "contrast": "CAP_PROP_CONTRAST",
    "saturation": "CAP_PROP_SATURATION",
}
CONTROL_NAMES = tuple(CONTROL_PROPERTIES)

# V4L2's exposure-mode values as OpenCV passes them through: 1 selects manual
# exposure, 3 the camera's aperture-priority automatic mode. Setting an exposure
# value while the camera is still choosing its own has no effect, so the mode has
# to be switched first.
EXPOSURE_MANUAL = 1
EXPOSURE_AUTO = 3


def apply_controls(cap: Any, controls: "Mapping[str, Any]") -> "list[str]":
    """Apply every non-None control in ``controls`` to ``cap``. Returns the names.

    Exposure is special-cased: the camera is taken out of automatic exposure
    first, because a value set while automatic is still in charge is simply
    ignored. Controls left as None are not sent at all, so a camera keeps
    whatever it powers up with.
    """
    import cv2  # type: ignore[import]

    applied: list[str] = []
    if controls.get("exposure") is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_MANUAL)
    for name in CONTROL_NAMES:
        value = controls.get(name)
        if value is None:
            continue
        cap.set(getattr(cv2, CONTROL_PROPERTIES[name]), float(value))
        applied.append(name)
    return applied


def read_controls(cap: Any) -> "dict[str, float]":
    """The camera's current value for every control it will report."""
    import cv2  # type: ignore[import]

    out: dict[str, float] = {}
    for name, prop in CONTROL_PROPERTIES.items():
        try:
            out[name] = float(cap.get(getattr(cv2, prop)))
        except Exception:
            continue
    return out


def control_yaml_line(name: str, value: "float | None", indent: int = 4) -> str:
    """One recording.yaml line for a control, formatted as the file writes them."""
    shown = "null" if value is None else f"{int(round(float(value)))}"
    return " " * indent + f"{name}: {shown}"
