"""Unit tests for the console's whole-dataset operations (actoris_harena.web.lifecycle).

The rules that decide whether a rename or a merge may go ahead are pure, so they
are tested directly; the two operations that touch the drive (rename, delete)
run against a temporary directory because their safety -- an atomic rename, and
trashing before removing -- is exactly what the tests are for. No aiohttp, no
LeRobot, no hardware.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from actoris_harena.recording.dataset_edit import ReadOnlyDatasetError
from actoris_harena.web.lifecycle import (
    _meta_json_choice,
    delete_dataset,
    directory_size,
    feature_signature,
    is_working_dir,
    merge_compatibility,
    merge_extra_ops,
    read_dataset_meta,
    rename_dataset,
    valid_dataset_name,
)


class TestNameValidation(unittest.TestCase):
    def test_accepts_the_names_the_drive_already_uses(self):
        for name in ("cube-pnp", "fold_short", "towel_fold2", "v3.0-run"):
            self.assertIsNone(valid_dataset_name(name), name)

    def test_refuses_traversal_and_separators(self):
        for name in ("../escape", "a/b", "/abs", ".hidden", ""):
            self.assertIsNotNone(valid_dataset_name(name), name)

    def test_refuses_spaces_and_shell_trouble(self):
        for name in ("my dataset", "run;rm -rf", "quote'd", "star*"):
            self.assertIsNotNone(valid_dataset_name(name), name)

    def test_refuses_the_working_directory_suffixes(self):
        self.assertIsNotNone(valid_dataset_name("cube-pnp.trash-20260819-110341"))
        self.assertIsNotNone(valid_dataset_name("cube-pnp.tmp-20260819-110341"))

    def test_working_directories_are_recognised(self):
        self.assertTrue(is_working_dir("cube-pnp.trash-20260819-110341"))
        self.assertTrue(is_working_dir("cube-pnp.tmp-20260101-000000"))
        self.assertFalse(is_working_dir("cube-pnp"))
        self.assertFalse(is_working_dir("trash-can"))

    def test_refuses_a_very_long_name(self):
        self.assertIsNotNone(valid_dataset_name("x" * 200))


class TestFeatureSignature(unittest.TestCase):
    def test_keeps_dtype_and_shape_only(self):
        sig = feature_signature(
            {
                "action": {"dtype": "float32", "shape": [12], "names": ["a"] * 12},
                "observation.images.central": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "info": {"video.fps": 30},
                },
            }
        )
        self.assertEqual(sig["action"], ("float32", (12,)))
        self.assertEqual(sig["observation.images.central"], ("video", (480, 640, 3)))

    def test_survives_rubbish(self):
        self.assertEqual(feature_signature({"a": "not a dict"}), {})
        self.assertEqual(feature_signature({}), {})


def _meta(name, **over):
    base = {
        "name": name,
        "fps": 30,
        "robot_type": "so101_dual",
        "features": {
            "action": ("float32", (12,)),
            "observation.images.central": ("video", (480, 640, 3)),
        },
        "pending": 0,
        "size_bytes": 1000,
    }
    base.update(over)
    return base


class TestMergeCompatibility(unittest.TestCase):
    def test_two_matching_datasets_may_merge(self):
        reasons = merge_compatibility(
            [_meta("a"), _meta("b")], "ab", free_bytes=10**9
        )
        self.assertEqual(reasons, [])

    def test_needs_at_least_two_sources(self):
        reasons = merge_compatibility([_meta("a")], "ab", free_bytes=10**9)
        self.assertTrue(any("at least two" in r for r in reasons))

    def test_refuses_a_bad_or_taken_output_name(self):
        self.assertTrue(
            merge_compatibility([_meta("a"), _meta("b")], "../x", free_bytes=10**9)
        )
        reasons = merge_compatibility(
            [_meta("a"), _meta("b")], "ab", free_bytes=10**9, taken_names=("ab",)
        )
        self.assertTrue(any("already exists" in r for r in reasons))

    def test_refuses_writing_over_a_source(self):
        reasons = merge_compatibility(
            [_meta("a"), _meta("b")], "a", free_bytes=10**9, taken_names=("a", "b")
        )
        self.assertTrue(any("same name as a source" in r for r in reasons))

    def test_refuses_different_frame_rates(self):
        reasons = merge_compatibility(
            [_meta("a"), _meta("b", fps=15)], "ab", free_bytes=10**9
        )
        self.assertTrue(any("fps" in r for r in reasons))

    def test_refuses_a_different_robot(self):
        reasons = merge_compatibility(
            [_meta("a"), _meta("b", robot_type="so101_single")], "ab", 10**9
        )
        self.assertTrue(any("so101_single" in r for r in reasons))

    def test_refuses_a_different_camera_set(self):
        other = _meta("b")
        other["features"] = dict(other["features"])
        other["features"]["observation.images.scene"] = ("video", (480, 640, 3))
        reasons = merge_compatibility([_meta("a"), other], "ab", 10**9)
        self.assertTrue(any("observation.images.scene" in r for r in reasons))

    def test_refuses_a_different_action_width(self):
        other = _meta("b")
        other["features"] = dict(other["features"])
        other["features"]["action"] = ("float32", (6,))
        reasons = merge_compatibility([_meta("a"), other], "ab", 10**9)
        self.assertTrue(any("action differs in shape" in r for r in reasons))

    def test_refuses_pending_soft_deletes(self):
        reasons = merge_compatibility(
            [_meta("a", pending=3), _meta("b")], "ab", 10**9
        )
        self.assertTrue(any("marked for deletion" in r for r in reasons))

    def test_refuses_when_the_drive_is_too_full(self):
        reasons = merge_compatibility(
            [_meta("a", size_bytes=6 * 10**9), _meta("b", size_bytes=6 * 10**9)],
            "ab",
            free_bytes=10**9,
        )
        self.assertTrue(any("free" in r for r in reasons))

    def test_collects_every_reason_at_once(self):
        reasons = merge_compatibility(
            [_meta("a", pending=1), _meta("b", fps=60)], "ab", free_bytes=10**9
        )
        self.assertGreaterEqual(len(reasons), 2)


class TestMergeExtraOps(unittest.TestCase):
    def test_offsets_follow_the_concatenation_order(self):
        ops = merge_extra_ops([2, 3])
        moved = {(s, a, b) for s, a, b in ops if "episode_" in a}
        self.assertIn(
            (0, "extra/episode_000000.parquet", "extra/episode_000000.parquet"), moved
        )
        self.assertIn(
            (1, "extra/episode_000000.parquet", "extra/episode_000002.parquet"), moved
        )
        self.assertIn(
            (1, "extra/episode_000002.parquet", "extra/episode_000004.parquet"), moved
        )

    def test_covers_drift_and_identity_files(self):
        ops = merge_extra_ops([1, 1])
        rels = {a for _s, a, _b in ops}
        self.assertIn("extra/drift_000000.parquet", rels)
        self.assertIn("extra/uid_000000.json", rels)

    def test_depth_directories_are_reindexed_too(self):
        ops = merge_extra_ops([1, 1], depth_names=["central"])
        self.assertIn(
            (
                1,
                "extra/depth/central/episode_000000",
                "extra/depth/central/episode_000001",
            ),
            ops,
        )

    def test_no_sources_no_ops(self):
        self.assertEqual(merge_extra_ops([]), [])


class TestOnDisk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "one" / "meta").mkdir(parents=True)
        (self.root / "one" / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "fps": 30,
                    "robot_type": "so101_dual",
                    "total_episodes": 2,
                    "total_frames": 100,
                    "features": {
                        "action": {"dtype": "float32", "shape": [12]},
                        "observation.images.central": {
                            "dtype": "video",
                            "shape": [480, 640, 3],
                        },
                    },
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_what_the_console_lists(self):
        meta = read_dataset_meta(self.root / "one")
        self.assertEqual(meta["fps"], 30)
        self.assertEqual(meta["total_episodes"], 2)
        self.assertEqual(meta["streams"], ["observation.images.central"])
        self.assertEqual(meta["pending"], 0)

    def test_an_unreadable_dataset_still_lists(self):
        meta = read_dataset_meta(self.root / "nothing-here")
        self.assertIsNone(meta["fps"])
        self.assertEqual(meta["streams"], [])

    def test_directory_size_counts_the_files(self):
        (self.root / "one" / "blob").write_bytes(b"x" * 2048)
        self.assertGreaterEqual(directory_size(self.root / "one"), 2048)

    def test_rename_moves_the_directory(self):
        rename_dataset(self.root, "one", "two")
        self.assertFalse((self.root / "one").exists())
        self.assertTrue((self.root / "two" / "meta" / "info.json").is_file())

    def test_rename_refuses_an_existing_target(self):
        (self.root / "two").mkdir()
        with self.assertRaises(FileExistsError):
            rename_dataset(self.root, "one", "two")

    def test_rename_refuses_a_bad_name(self):
        with self.assertRaises(ValueError):
            rename_dataset(self.root, "one", "../escape")

    def test_rename_refuses_an_unknown_dataset(self):
        with self.assertRaises(FileNotFoundError):
            rename_dataset(self.root, "missing", "two")

    def test_delete_trashes_before_removing(self):
        trash = delete_dataset(self.root, "one")
        self.assertFalse((self.root / "one").exists())
        self.assertTrue(trash.is_dir())
        self.assertTrue(is_working_dir(trash.name))

    def test_delete_refuses_an_unknown_dataset(self):
        with self.assertRaises(FileNotFoundError):
            delete_dataset(self.root, "missing")

    def test_delete_refuses_a_read_only_drive(self):
        os.chmod(self.root, 0o555)
        try:
            with self.assertRaises(ReadOnlyDatasetError):
                delete_dataset(self.root, "one")
        finally:
            os.chmod(self.root, 0o755)

    def test_meta_json_is_carried_over_only_when_the_sources_agree(self):
        for name in ("one", "two"):
            (self.root / name / "meta").mkdir(parents=True, exist_ok=True)
            (self.root / name / "meta" / "action_space.json").write_text('{"a": 1}')
        roots = [self.root / "one", self.root / "two"]
        self.assertIsNotNone(_meta_json_choice(roots, "action_space.json"))
        (self.root / "two" / "meta" / "action_space.json").write_text('{"a": 2}')
        with self.assertRaises(ValueError):
            _meta_json_choice(roots, "action_space.json")

    def test_meta_json_absent_everywhere_is_not_an_error(self):
        self.assertIsNone(_meta_json_choice([self.root / "one"], "realsense.json"))


if __name__ == "__main__":
    unittest.main()
