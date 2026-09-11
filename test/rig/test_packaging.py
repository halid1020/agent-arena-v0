"""The two facts that make this package installable next to LeRobot.

Both were broken before, and both are the kind of thing that breaks again
silently -- an eager import added to __init__.py costs nothing on the sim
machine and makes every robot venv unable to import the package at all.
"""

import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestExtrasDoNotCollide(unittest.TestCase):
    """[sim] and [rig] must stay separable, and core must stay tiny."""

    def setUp(self) -> None:
        with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
            self.pyproject = tomllib.load(handle)
        self.extras = self.pyproject["project"]["optional-dependencies"]

    def test_core_dependencies_pull_in_neither_stack(self) -> None:
        # Anything here is paid for by BOTH consumers, so it may not name a
        # package that belongs to only one of them.
        core = " ".join(self.pyproject["project"]["dependencies"])
        for sim_only in ("gym", "robosuite", "pybullet", "dm_control", "ray"):
            self.assertNotIn(sim_only, core)
        for rig_only in ("aiohttp", "pyarrow", "opencv", "matplotlib"):
            self.assertNotIn(rig_only, core)

    def test_the_numpy_pin_lives_in_sim_and_nowhere_else(self) -> None:
        # MEASURED: LeRobot pins numpy>=2.0,<2.3; the sim stack pins numpy<2.0.
        # A `numpy<2` reachable from [rig] or from core would make the robot
        # venvs uninstallable, which is the whole reason for the split.
        self.assertIn("numpy<2.0", self.extras["sim"])
        for name in ("rig", "realsense"):
            for spec in self.extras[name]:
                self.assertNotIn("<2.0", spec, f"{name} must not cap numpy")
        for spec in self.pyproject["project"]["dependencies"]:
            self.assertNotIn("<2.0", spec, "core must not cap numpy")

    def test_rig_does_not_declare_lerobot_or_torch(self) -> None:
        # Each robot repo installs LeRobot editable from a sibling checkout at
        # a pinned commit. A range here would fight that pin; torch arrives
        # with it.
        joined = " ".join(self.extras["rig"])
        self.assertNotIn("lerobot", joined)
        self.assertNotIn("torch", joined)


class TestImportIsCheap(unittest.TestCase):
    def test_importing_the_package_loads_no_heavy_dependency(self) -> None:
        """`import actoris_harena` must not drag the sim stack in.

        Run in a SUBPROCESS: whatever imported this test file has already
        populated sys.modules, so checking in-process would prove nothing.
        """
        code = (
            "import sys; import actoris_harena; "
            "print(','.join(m for m in "
            "('torch','pybullet','dm_control','robosuite','gym','ray','cv2') "
            "if m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            out.stdout.strip(),
            "",
            "importing actoris_harena pulled in a heavy dependency; "
            "an eager import crept back into __init__.py",
        )

    def test_every_lazy_name_names_a_real_module(self) -> None:
        import actoris_harena

        for name, module_path in actoris_harena._LAZY.items():
            self.assertTrue(
                (REPO_ROOT / Path(*module_path.split("."))).with_suffix(".py").exists()
                or (REPO_ROOT / Path(*module_path.split(".")) / "__init__.py").exists(),
                f"{name} points at {module_path}, which is not on disk",
            )

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import actoris_harena

        with self.assertRaises(AttributeError):
            actoris_harena.definitely_not_a_real_name


if __name__ == "__main__":
    unittest.main()


class TestTheCameraProfileIsTheRigsToSet(unittest.TestCase):
    """The three tables that name a bench's cameras are not this package's.

    A shared view builder that shipped one rig's camera names would silently
    give a different rig the wrong answer -- a composite naming four cameras it
    does not have, or a pi0.5 slot mapping for a viewpoint it never sees. The
    empty default is what makes that impossible: nothing is assumed until a rig
    says so.
    """

    def tearDown(self) -> None:
        from actoris_harena.recording.camera_profile import CameraProfile, set_profile

        set_profile(CameraProfile())

    def test_the_default_profile_names_no_camera(self) -> None:
        from actoris_harena.recording.camera_profile import CameraProfile, profile

        set_to_default = CameraProfile()
        self.assertEqual(set_to_default.pi05_slots, {})
        self.assertEqual(set_to_default.composites, {})
        self.assertEqual(set_to_default.slug_elisions, ())
        self.assertEqual(profile().composites, {})

    def test_a_rig_installs_its_own_and_the_view_builder_sees_it(self) -> None:
        from actoris_harena.recording import dataset_view
        from actoris_harena.recording.camera_profile import CameraProfile, set_profile

        set_profile(CameraProfile(composites={"quad": ("a", "b", "c", "d")}))
        self.assertEqual(dataset_view._composites(), {"quad": ("a", "b", "c", "d")})

    def test_pi05s_own_slot_names_stay_in_the_module(self) -> None:
        # These are facts about the pretrained model, identical on every rig,
        # so they are NOT part of the profile.
        from actoris_harena.recording.dataset_view import (
            PI05_SLOT_ALIASES,
            PI05_SLOT_ORDER,
        )

        self.assertEqual(len(PI05_SLOT_ORDER), 3)
        self.assertEqual(set(PI05_SLOT_ALIASES.values()), set(PI05_SLOT_ORDER))
