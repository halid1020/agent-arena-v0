"""Which part of a policy's conditioning belongs to which input stream.

This is the one structural fact the whole package rests on: both policies
trained here reduce their inputs to a vector, and **each stream owns a
contiguous piece of it**. Knowing which piece is what makes "how much did the
left fingertip contribute" a question with an arithmetic answer rather than a
picture to squint at.

**ACT** (``policies/act/modeling_act.py``) builds its encoder input as a list of
tokens::

    [ latent ] [ state ] [ env_state ]? [ cam0 x H*W ] [ cam1 x H*W ] ...

one token per cell of each camera's ResNet feature map, cameras in
``config.image_features`` order -- which is the order the checkpoint's own
``config.json`` records, not alphabetical. At 480x640 a ResNet-18 ``layer4`` map
is 15x20, so each camera owns 300 consecutive token indices and five cameras
plus latent and state make 1 502.

**Diffusion** (``policies/diffusion/modeling_diffusion.py``) concatenates
``[state, cam0 feat, cam1 feat, ...]`` per observation step and flattens::

    [ step0: state cam0 cam1 ... ] [ step1: state cam0 cam1 ... ] ...

so a camera owns a contiguous run WITHIN each step block and the blocks repeat.
That is why a span here is a list of ranges rather than one: with
``n_obs_steps`` = 2, every stream appears twice.

Pure: a config in, index ranges out. No torch, no weights, no GPU -- which is
what lets the layout be tested against both policies' real shapes on a laptop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CAMERA_PREFIX = "observation.images."
STATE_KEY = "observation.state"
ENV_STATE_KEY = "observation.environment_state"

#: What a ResNet trunk divides its input by, from stem to ``layer4``. Used only
#: to PREDICT a token count; the real one is measured when weights are at hand
#: (see :func:`act_layout`), because a config that changes the final stride --
#: ACT's ``replace_final_stride_with_dilation`` does exactly that -- makes the
#: prediction wrong by a factor of four and would silently mis-assign every
#: token after the first camera.
RESNET_STRIDE = 32


@dataclass(frozen=True)
class Stream:
    """One input the policy was given, whatever a policy calls it internally."""

    name: str  # "central", or "state"
    kind: str  # "camera" | "state" | "env_state" | "latent"
    key: str  # the batch key, e.g. "observation.images.central"

    @property
    def is_camera(self) -> bool:
        return self.kind == "camera"


@dataclass
class Span:
    """Where one stream lives in a conditioning vector.

    ``ranges`` are half-open ``[start, stop)`` index pairs. More than one when
    the stream appears once per observation step, which is diffusion's layout.
    """

    stream: Stream
    ranges: "list[tuple[int, int]]" = field(default_factory=list)

    @property
    def width(self) -> int:
        return sum(stop - start for start, stop in self.ranges)

    def indices(self) -> "list[int]":
        out: "list[int]" = []
        for start, stop in self.ranges:
            out.extend(range(start, stop))
        return out


def camera_names(cfg) -> "list[str]":
    """The cameras in the order the policy stacks them. Order is load-bearing.

    Both policies build their image batch as
    ``[batch[key] for key in config.image_features]``, so this dict's order --
    the checkpoint's, not the filesystem's and not alphabetical -- decides which
    block of the conditioning vector is which camera. Sorting here would silently
    mislabel every attribution on a rig whose camera names do not happen to sort
    into the order they were trained in.
    """
    return [key[len(CAMERA_PREFIX) :] for key in cfg.image_features]


def streams_of(cfg) -> "list[Stream]":
    """Every input this checkpoint takes: the state, then the cameras."""
    out: "list[Stream]" = []
    if getattr(cfg, "robot_state_feature", None) is not None:
        out.append(Stream("state", "state", STATE_KEY))
    if getattr(cfg, "env_state_feature", None) is not None:
        out.append(Stream("env_state", "env_state", ENV_STATE_KEY))
    for name in camera_names(cfg):
        out.append(Stream(name, "camera", CAMERA_PREFIX + name))
    return out


def feature_map_hw(
    height: int, width: int, stride: int = RESNET_STRIDE
) -> "tuple[int, int]":
    """The ResNet feature-map size for an input, by the network's own arithmetic.

    Every downsampling stage of a ResNet rounds UP (stride-2 convolutions with
    padding 1, and a ceil-mode-free max-pool that still lands on the ceiling for
    odd sizes), so 480x640 -> 15x20 and, say, 180x240 -> 6x8.
    """
    return math.ceil(height / stride), math.ceil(width / stride)


def tokens_per_camera(cfg, stride: int = RESNET_STRIDE) -> int:
    """How many encoder tokens one camera becomes. See :func:`feature_map_hw`.

    Predicted from the config's declared image shape. Prefer the measured value
    where weights are loaded -- ``replace_final_stride_with_dilation`` quadruples
    this and nothing in the shape says so.
    """
    features = list(cfg.image_features.values())
    if not features:
        return 0
    _, height, width = features[0].shape
    effective = (
        stride // 2
        if getattr(cfg, "replace_final_stride_with_dilation", False)
        else stride
    )
    rows, cols = feature_map_hw(height, width, effective)
    return rows * cols


def act_layout(cfg, per_camera: "int | None" = None) -> "list[Span]":
    """Where each stream sits among ACT's encoder tokens.

    ``per_camera`` is the measured token count when one is available; without it
    the count is predicted from the config. Index 0 is always the VAE latent,
    which is not an input at all and so is not a stream -- it is skipped rather
    than attributed to anything.
    """
    width = int(per_camera if per_camera is not None else tokens_per_camera(cfg))
    spans: "list[Span]" = []
    cursor = 1  # token 0 is the latent
    for stream in streams_of(cfg):
        size = width if stream.is_camera else 1
        spans.append(Span(stream, [(cursor, cursor + size)]))
        cursor += size
    return spans


def act_token_total(cfg, per_camera: "int | None" = None) -> int:
    """How many encoder tokens the whole observation becomes, latent included."""
    spans = act_layout(cfg, per_camera)
    return 1 + sum(s.width for s in spans)


def diffusion_layout(cfg, camera_feature_dim: int) -> "list[Span]":
    """Where each stream sits in diffusion's flattened ``global_cond``.

    One block per observation step, and within a block: the state, then each
    camera's encoder output in ``image_features`` order. ``camera_feature_dim``
    is the encoder's ``feature_dim``, which depends on the crop and the backbone
    and so is measured from the loaded model rather than derived here.
    """
    state = getattr(cfg, "robot_state_feature", None)
    state_dim = int(state.shape[0]) if state is not None else 0
    cameras = camera_names(cfg)
    per_step = state_dim + camera_feature_dim * len(cameras)
    steps = int(getattr(cfg, "n_obs_steps", 1) or 1)

    spans = {s.name: Span(s, []) for s in streams_of(cfg)}
    for step in range(steps):
        base = step * per_step
        if state_dim:
            spans["state"].ranges.append((base, base + state_dim))
        for index, name in enumerate(cameras):
            start = base + state_dim + index * camera_feature_dim
            spans[name].ranges.append((start, start + camera_feature_dim))
    return list(spans.values())


def check_layout(spans: "list[Span]", total: int, latent: int = 0) -> "list[str]":
    """Does this layout actually tile the vector? Returns what is wrong.

    Worth asserting rather than assuming: an off-by-one in a camera's token
    count does not raise anywhere, it just attributes the tail of one camera to
    the head of the next -- and the resulting figure looks entirely plausible.
    """
    problems: "list[str]" = []
    covered = sorted((start, stop) for span in spans for start, stop in span.ranges)
    if not covered:
        return ["no stream owns any of the conditioning vector"]
    cursor = latent
    for start, stop in covered:
        if start < cursor:
            problems.append(f"streams overlap at index {start}")
        elif start > cursor:
            problems.append(f"nothing owns indices {cursor}..{start - 1}")
        cursor = max(cursor, stop)
    if cursor != total:
        problems.append(f"layout covers {cursor} of {total} entries")
    return problems


def token_layout(cfg, tokens_per_camera: int) -> "list[Span]":
    """Where each stream sits in a TOKEN model's prefix (pi0.5 and friends).

    The third architecture this rig trains. Unlike ACT -- which puts one latent
    and one state token ahead of the cameras -- and unlike diffusion -- which
    flattens everything into one vector per observation step -- a token model
    lays its cameras out as image patches followed by the state, all in one
    sequence, in ``image_features`` order.

    ``tokens_per_camera`` is the patch count of whatever vision tower the
    checkpoint carries and is MEASURED from the loaded model, never derived from
    the declared image shape: the tower resizes to its own resolution first, so
    the declared shape says nothing about how many patches come out.

    Returns spans over the PREFIX only. The action tokens that follow are the
    output, not conditioning, so they own no span here.
    """
    cameras = camera_names(cfg)
    state = getattr(cfg, "robot_state_feature", None)
    state_dim = int(state.shape[0]) if state is not None else 0

    spans = {s.name: Span(s, []) for s in streams_of(cfg)}
    cursor = 0
    for name in cameras:
        spans[name].ranges.append((cursor, cursor + tokens_per_camera))
        cursor += tokens_per_camera
    if state_dim and "state" in spans:
        # One token, however wide the state vector: a token model projects the
        # whole state into the model width rather than giving each joint a slot.
        spans["state"].ranges.append((cursor, cursor + 1))
    return list(spans.values())
