"""Which cameras this rig has, and what a fixed-view policy should do with them.

Three tables here name a rig's own cameras, and every one of them is a fact
about the bench rather than about the code that reads it:

* which camera feeds which of pi0.5's three pretrained slots,
* which cameras tile together into one composite image feature,
* and how a camera's name is shortened in a run-directory slug.

They used to be module constants in ``dataset_view``, written for a dual SO-101
with a central camera, two wrists and four fingertips. A single UR3e with one
wrist camera has different answers to all three, so they are a PROFILE the rig
sets once, not constants a shared package can hold.

What is NOT here is pi0.5's own slot names and their aliases. Those are facts
about the pretrained model, identical on every rig, and they stay in
``dataset_view`` beside the code that assigns them.

The empty default is legitimate, not a missing value: a rig with no composites
and no camera whose viewpoint pi0.5 was pretrained on is correctly described by
it. Every unmapped camera simply takes the next free slot in
``PI05_SLOT_ORDER``, which is already the documented rule for a viewpoint the
model has never seen.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraProfile:
    """One rig's camera naming, as the view builder needs to know it."""

    #: Rig camera name -> the openpi image feature key it should be renamed to.
    #: A camera absent from this map is one pi0.5 has no pretrained view of.
    pi05_slots: "dict[str, str]" = field(default_factory=dict)

    #: Composite name -> the cameras tiled into it, in reading order. A
    #: composite spends ONE image feature on several cameras, for a policy
    #: whose architecture fixes how many views it takes.
    composites: "dict[str, tuple[str, ...]]" = field(default_factory=dict)

    #: ``(prefix, replacement)`` pairs applied to a camera name when a run
    #: directory is named after it. Cosmetic, but it is in directory names that
    #: already exist, so it is a rig's to decide and not to re-derive.
    slug_elisions: "tuple[tuple[str, str], ...]" = ()


_ACTIVE = CameraProfile()


def set_profile(profile: CameraProfile) -> None:
    """Declare this rig's cameras. Called once, before any view is built.

    Process-global on purpose. The console serves one rig at a time and drives
    every other rig through a subprocess with its own interpreter, so there is
    never a second profile to hold -- and a parameter threaded through every
    caller would buy only a capability the architecture does not use.
    """
    global _ACTIVE
    _ACTIVE = profile


def profile() -> CameraProfile:
    """The active rig's camera profile; empty until a rig sets one."""
    return _ACTIVE
