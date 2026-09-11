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
