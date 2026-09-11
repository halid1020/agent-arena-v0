"""Camera-ablation views: what they show, what they share, and what they fix.

An ablation is only evidence if the datasets it compares differ in ONE thing.
These tests hold a view to that: the episodes, frames and actions come through
untouched, the kept cameras are exactly the ones asked for, and every place the
metadata names a camera agrees with every other place -- because a view that
still advertises a dropped camera's statistics is a dataset that describes
itself wrongly, and LeRobot would believe it.

The task rewrite is checked against the trap the real data set: on
``cube-pnp-new`` the typo'd instruction is the FIRST-registered one, so a rule
that trusts position picks the wrong string and only a rule that weighs frames
picks the right one.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_dataset_view
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from actoris_harena.recording.camera_profile import CameraProfile, set_profile
from actoris_harena.recording.dataset_view import (
    COMPOSITE_SIZE,
    PI05_SLOT_ORDER,
    ViewError,
    available_camera_names,
    build_view,
    camera_keys,
    composite_feature,
    composite_grid,
    composite_stats,
    dropped_meta_columns,
    filtered_info,
    filtered_stats,
    pi05_rename_map,
    resolve_cameras,
    short_name,
    split_selection,
    task_remap,
    view_slug,
)

# The rig this module's fixtures describe. These three tables are no longer in
# dataset_view -- they name a particular bench's cameras, and the package
# serves more than one -- so the test installs them, which is also the check
# that the seam works: every assertion below passes only because a profile was
# set, and the same code with no profile simply has no composites.
PI05_SLOTS = {
    "central": "observation.images.base_0_rgb",
    "wrist_camera_left": "observation.images.left_wrist_0_rgb",
    "wrist_camera_right": "observation.images.right_wrist_0_rgb",
}
COMPOSITES = {
    "tactile_quad": (
        "left_arm_left_gripper",
        "left_arm_right_gripper",
        "right_arm_left_gripper",
        "right_arm_right_gripper",
    ),
}


def setUpModule():
    set_profile(
        CameraProfile(
            pi05_slots=dict(PI05_SLOTS),
            composites=dict(COMPOSITES),
            slug_elisions=(("wrist_camera_", "wrist_"),),
        )
    )


def tearDownModule():
    set_profile(CameraProfile())


CAMERAS = ["central", "wrist_camera_left", "wrist_camera_right"]
KEYS = [f"observation.images.{c}" for c in CAMERAS]
GOOD = "pick the cube and place it on the plate"
TYPO = "pick the cube and place it one the plate"
LENGTH = 4


def build_source(root: Path, tasks=(TYPO, GOOD), task_of_episode=(0, 1, 1)) -> Path:
    """A miniature dataset shaped like the real one.

    ``tasks`` is written in registration order and ``task_of_episode`` says which
    index each episode carries, so a fixture can reproduce the real dataset's
    arrangement: the typo registered first, the good spelling used by most.
    """
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": [12]},
        "action": {"dtype": "float32", "shape": [12]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }
    for key in KEYS:
        features[key] = {"dtype": "video", "shape": [480, 640, 3]}
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "total_episodes": len(task_of_episode),
                "total_frames": len(task_of_episode) * LENGTH,
                "total_tasks": len(tasks),
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": features,
            }
        )
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps({k: {"mean": [0.0]} for k in ["observation.state", *KEYS]})
    )
    pd.DataFrame(
        {"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")
    ).to_parquet(root / "meta" / "tasks.parquet")

    rows = []
    for episode, task_index in enumerate(task_of_episode):
        row = {
            "episode_index": episode,
            "tasks": [tasks[task_index]],
            "length": LENGTH,
            "dataset_from_index": episode * LENGTH,
            "dataset_to_index": (episode + 1) * LENGTH,
            "stats/observation.state/mean": [0.0],
        }
        for key in KEYS:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = 0
            row[f"stats/{key}/mean"] = [0.5]
        rows.append(row)
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([episode] * LENGTH, pa.int64()),
                    "frame_index": pa.array(range(LENGTH), pa.int64()),
                    "task_index": pa.array([task_index] * LENGTH, pa.int64()),
                }
            ),
            root / "data" / "chunk-000" / f"file-{episode:03d}.parquet",
        )
    pq.write_table(
        pa.Table.from_pylist(rows),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    for key in KEYS:
        d = root / "videos" / key / "chunk-000"
        d.mkdir(parents=True)
        (d / "file-000.mp4").write_bytes(b"not really a video")
    return root


class TestNaming(unittest.TestCase):
    def setUp(self):
        self.info = json.loads(
            json.dumps(
                {
                    "features": {
                        **{k: {"dtype": "video", "shape": [1]} for k in KEYS},
                        "observation.state": {"dtype": "float32", "shape": [12]},
                    }
                }
            )
        )

    def test_camera_keys_are_the_video_features_in_order(self):
        self.assertEqual(camera_keys(self.info), KEYS)

    def test_short_name_strips_the_prefix(self):
        self.assertEqual(short_name("observation.images.central"), "central")

    def test_resolve_accepts_short_and_full_names(self):
        self.assertEqual(resolve_cameras(self.info, ["central"]), [KEYS[0]])
        self.assertEqual(resolve_cameras(self.info, [KEYS[0]]), [KEYS[0]])

    def test_all_selects_every_camera(self):
        self.assertEqual(resolve_cameras(self.info, ["all"]), KEYS)

    def test_result_is_dataset_order_regardless_of_request_order(self):
        asked = ["wrist_camera_right", "central"]
        self.assertEqual(resolve_cameras(self.info, asked), [KEYS[0], KEYS[2]])

    def test_a_camera_asked_for_twice_is_one_camera(self):
        self.assertEqual(resolve_cameras(self.info, ["central", "central"]), [KEYS[0]])

    def test_an_unknown_camera_names_the_ones_that_exist(self):
        with self.assertRaises(ViewError) as caught:
            resolve_cameras(self.info, ["wrist_left"])
        message = str(caught.exception)
        self.assertIn("wrist_left", message)
        self.assertIn("central", message)

    def test_slug_says_all_only_for_the_whole_set(self):
        self.assertEqual(view_slug(self.info, KEYS), "all")
        self.assertEqual(view_slug(self.info, KEYS[:2]), "central+wrist_left")
        self.assertEqual(view_slug(self.info, [KEYS[1]]), "wrist_left")


class TestMetadataFiltering(unittest.TestCase):
    def setUp(self):
        self.info = {
            "features": {
                "observation.state": {"dtype": "float32", "shape": [12]},
                **{k: {"dtype": "video", "shape": [480, 640, 3]} for k in KEYS},
            }
        }

    def test_only_the_dropped_cameras_leave_the_feature_table(self):
        out = filtered_info(self.info, [KEYS[1]])
        self.assertEqual(camera_keys(out), [KEYS[1]])
        self.assertIn("observation.state", out["features"])

    def test_the_source_info_is_not_mutated(self):
        filtered_info(self.info, [KEYS[1]])
        self.assertEqual(camera_keys(self.info), KEYS)

    def test_stats_lose_the_dropped_cameras_only(self):
        stats = {"observation.state": 1, **{k: 1 for k in KEYS}}
        out = filtered_stats(stats, [KEYS[0]], KEYS)
        self.assertEqual(sorted(out), sorted(["observation.state", KEYS[0]]))

    def test_both_video_and_stats_columns_of_a_dropped_camera_go(self):
        columns = [
            "episode_index",
            f"videos/{KEYS[0]}/chunk_index",
            f"stats/{KEYS[0]}/mean",
            f"videos/{KEYS[1]}/chunk_index",
            f"stats/{KEYS[1]}/mean",
        ]
        dropped = dropped_meta_columns(columns, [KEYS[1]], KEYS)
        self.assertEqual(
            sorted(dropped),
            sorted([f"videos/{KEYS[0]}/chunk_index", f"stats/{KEYS[0]}/mean"]),
        )
        self.assertNotIn("episode_index", dropped)


class TestTaskRemap(unittest.TestCase):
    def test_the_spelling_with_the_most_frames_wins(self):
        chosen, _ = task_remap([TYPO, GOOD], counts={0: 2612, 1: 38429})
        self.assertEqual(chosen, GOOD)

    def test_position_does_not_decide_it(self):
        """The real dataset registers the typo first; a positional rule picks it."""
        chosen, _ = task_remap([TYPO, GOOD], counts={0: 1, 1: 99})
        self.assertNotEqual(chosen, TYPO)

    def test_every_old_index_maps_onto_the_single_survivor(self):
        _, mapping = task_remap([TYPO, GOOD], counts={0: 1, 1: 9})
        self.assertEqual(set(mapping.values()), {0})
        self.assertEqual(sorted(mapping), [0, 1])

    def test_an_explicit_choice_overrides_the_counts(self):
        chosen, _ = task_remap(
            [TYPO, GOOD], counts={0: 1, 1: 99}, canonical="something else"
        )
        self.assertEqual(chosen, "something else")

    def test_no_tasks_is_refused(self):
        with self.assertRaises(ViewError):
            task_remap([])


class TestPi05Slots(unittest.TestCase):
    """pi0.5 arrives with fixed slots, so a rig camera is mapped, never renamed."""

    def test_each_camera_lands_on_the_slot_that_means_the_same_viewpoint(self):
        self.assertEqual(
            pi05_rename_map(KEYS),
            {
                KEYS[0]: "observation.images.base_0_rgb",
                KEYS[1]: "observation.images.left_wrist_0_rgb",
                KEYS[2]: "observation.images.right_wrist_0_rgb",
            },
        )

    def test_an_ablated_view_maps_only_what_it_has(self):
        # The slots left over are the ablation: pi0.5 pads and masks them.
        self.assertEqual(
            pi05_rename_map([KEYS[1]]), {KEYS[1]: "observation.images.left_wrist_0_rgb"}
        )

    def test_two_cameras_never_share_a_slot(self):
        self.assertEqual(len(set(PI05_SLOTS.values())), len(PI05_SLOTS))

    def test_a_camera_pi05_never_saw_still_gets_a_slot(self):
        # This rig's fingertip cameras have no counterpart in pi0.5's
        # pretraining -- no slot was ever shown a gel image -- so naming is
        # impossible and the choice is between an arbitrary slot and refusing
        # to train pi0.5 on a tactile dataset at all. It takes the first free
        # one, and the map is printed by the driver so the assignment is on the
        # record rather than implied.
        gel = "observation.images.left_arm_left_gripper"
        self.assertEqual(pi05_rename_map([gel]), {gel: PI05_SLOT_ORDER[0]})

    def test_a_named_camera_keeps_its_own_slot_when_an_unnamed_one_is_present(self):
        # The overhead view must not be pushed off base_0_rgb by a camera that
        # merely came first: what a slot learned is the reason to use it.
        gel = "observation.images.left_arm_left_gripper"
        self.assertEqual(
            pi05_rename_map([gel, KEYS[0]]),
            {KEYS[0]: PI05_SLOT_ORDER[0], gel: PI05_SLOT_ORDER[1]},
        )

    def test_more_cameras_than_slots_is_refused_rather_than_dropped(self):
        five = KEYS + [
            "observation.images.left_arm_left_gripper",
            "observation.images.right_arm_left_gripper",
        ]
        with self.assertRaises(ViewError) as caught:
            pi05_rename_map(five)
        self.assertIn("3 image slots", str(caught.exception))
        self.assertIn("5 cameras", str(caught.exception))

    def test_a_pinned_map_overrides_the_named_slots(self):
        # An ablation asks what a slot's pretraining is worth, so it has to be
        # able to put a camera somewhere the naming would not.
        self.assertEqual(
            pi05_rename_map(KEYS[:2], {"central": "right_wrist"}),
            {KEYS[0]: "observation.images.right_wrist_0_rgb"},
        )

    def test_a_pinned_map_naming_a_camera_the_view_lacks_is_refused(self):
        with self.assertRaises(ViewError) as caught:
            pi05_rename_map(KEYS[:1], {"wrist_camera_left": "base"})
        self.assertIn("wrist_camera_left", str(caught.exception))

    def test_a_pinned_map_naming_something_that_is_not_a_slot_is_refused(self):
        with self.assertRaises(ViewError) as caught:
            pi05_rename_map(KEYS[:1], {"central": "overhead"})
        self.assertIn("overhead", str(caught.exception))


class TestCompositeNames(unittest.TestCase):
    """A composite is offered only when the dataset can actually build it."""

    def _info(self, cameras):
        return {
            "features": {f"observation.images.{c}": {"dtype": "video"} for c in cameras}
        }

    def test_a_dataset_with_every_part_is_offered_the_composite(self):
        parts = COMPOSITES["tactile_quad"]
        names = available_camera_names(self._info(["central", *parts]))
        self.assertIn("tactile_quad", names)

    def test_a_dataset_missing_one_part_is_not(self):
        parts = COMPOSITES["tactile_quad"][:-1]
        names = available_camera_names(self._info(["central", *parts]))
        self.assertNotIn("tactile_quad", names)

    def test_the_dataset_s_own_cameras_come_first_and_in_order(self):
        names = available_camera_names(self._info(CAMERAS))
        self.assertEqual(names[: len(CAMERAS)], CAMERAS)


class TestBuildView(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = build_source(self.tmp / "cube-pnp-new")
        self.dst = self.tmp / "view"

    def info(self, root: Path) -> dict:
        return json.loads((root / "meta" / "info.json").read_text())

    def episode_rows(self, root: Path) -> list:
        return pq.read_table(
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ).to_pylist()

    def data_task_indices(self, root: Path) -> list:
        out = []
        for path in sorted((root / "data").rglob("*.parquet")):
            out += pq.read_table(path).column("task_index").to_pylist()
        return out

    def test_the_view_names_only_the_kept_cameras(self):
        build_view(self.src, self.dst, ["central", "wrist_camera_left"])
        self.assertEqual(camera_keys(self.info(self.dst)), KEYS[:2])

    def test_video_directories_are_symlinks_to_the_source(self):
        build_view(self.src, self.dst, ["wrist_camera_left"])
        link = self.dst / "videos" / KEYS[1]
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), (self.src / "videos" / KEYS[1]).resolve())

    def test_a_dropped_camera_leaves_no_directory_behind(self):
        build_view(self.src, self.dst, ["wrist_camera_left"])
        self.assertFalse((self.dst / "videos" / KEYS[0]).exists())

    def test_dropped_cameras_leave_the_per_episode_table_too(self):
        build_view(self.src, self.dst, ["wrist_camera_left"])
        columns = self.episode_rows(self.dst)[0].keys()
        self.assertNotIn(f"stats/{KEYS[0]}/mean", columns)
        self.assertIn(f"stats/{KEYS[1]}/mean", columns)

    def test_stats_json_follows_the_features(self):
        build_view(self.src, self.dst, ["wrist_camera_left"])
        stats = json.loads((self.dst / "meta" / "stats.json").read_text())
        self.assertIn(KEYS[1], stats)
        self.assertNotIn(KEYS[0], stats)

    def test_episodes_and_frames_are_untouched(self):
        build_view(self.src, self.dst, ["wrist_camera_left"])
        self.assertEqual(
            self.info(self.dst)["total_episodes"], self.info(self.src)["total_episodes"]
        )
        self.assertEqual(
            len(self.data_task_indices(self.dst)), len(self.data_task_indices(self.src))
        )

    def test_the_source_dataset_is_never_modified(self):
        before = json.dumps(self.info(self.src), sort_keys=True)
        build_view(self.src, self.dst, ["wrist_camera_left"])
        self.assertEqual(json.dumps(self.info(self.src), sort_keys=True), before)
        self.assertEqual(sorted(set(self.data_task_indices(self.src))), [0, 1])

    def test_the_typo_is_dropped_for_the_spelling_most_frames_carry(self):
        build_view(self.src, self.dst, ["all"])
        tasks = pd.read_parquet(self.dst / "meta" / "tasks.parquet")
        self.assertEqual(list(tasks.index), [GOOD])
        self.assertEqual(set(self.data_task_indices(self.dst)), {0})
        self.assertEqual(self.info(self.dst)["total_tasks"], 1)

    def test_the_per_episode_task_strings_are_rewritten_too(self):
        build_view(self.src, self.dst, ["all"])
        for row in self.episode_rows(self.dst):
            self.assertEqual(list(row["tasks"]), [GOOD])

    def test_a_single_task_dataset_is_left_alone(self):
        src = build_source(self.tmp / "single", tasks=(GOOD,), task_of_episode=(0, 0))
        build_view(src, self.tmp / "single_view", ["all"])
        tasks = pd.read_parquet(self.tmp / "single_view" / "meta" / "tasks.parquet")
        self.assertEqual(list(tasks.index), [GOOD])

    def test_task_index_keeps_its_parquet_type(self):
        build_view(self.src, self.dst, ["all"])
        path = sorted((self.dst / "data").rglob("*.parquet"))[0]
        self.assertEqual(pq.read_schema(path).field("task_index").type, pa.int64())

    def test_building_again_reuses_the_view(self):
        build_view(self.src, self.dst, ["wrist_camera_left"])
        marker = self.dst / "meta" / "marker"
        marker.write_text("kept")
        build_view(self.src, self.dst, ["wrist_camera_left"])
        self.assertTrue(marker.exists())

    def test_force_rebuilds_it(self):
        build_view(self.src, self.dst, ["wrist_camera_left"])
        marker = self.dst / "meta" / "marker"
        marker.write_text("gone")
        build_view(self.src, self.dst, ["wrist_camera_left"], force=True)
        self.assertFalse(marker.exists())

    def test_nothing_is_left_beside_the_view(self):
        build_view(self.src, self.dst, ["central"])
        self.assertEqual(
            [p.name for p in self.tmp.iterdir() if p.name.startswith(".")], []
        )

    def test_an_unknown_camera_builds_nothing(self):
        with self.assertRaises(ViewError):
            build_view(self.src, self.dst, ["no_such_camera"])
        self.assertFalse(self.dst.exists())

    def test_a_missing_source_is_refused(self):
        with self.assertRaises(ViewError):
            build_view(self.tmp / "nowhere", self.dst, ["all"])

    def test_extra_is_not_carried_into_the_view(self):
        (self.src / "extra").mkdir()
        (self.src / "extra" / "drift_000000.parquet").write_bytes(b"x")
        build_view(self.src, self.dst, ["all"])
        self.assertFalse((self.dst / "extra").exists())


if __name__ == "__main__":
    unittest.main()


class TestCompositeGeometry(unittest.TestCase):
    """Four fingertip cameras into one frame, because FastWAM takes two views."""

    def test_four_cameras_tile_two_by_two(self):
        self.assertEqual(composite_grid(4), (2, 2))

    def test_a_count_that_cannot_tile_is_refused(self):
        with self.assertRaises(ViewError):
            composite_grid(3)

    def test_the_size_is_two_of_these_side_by_side(self):
        # FastWAM's default image_size is 224x448 -- exactly the overhead view
        # and this composite. If the composite stopped being square the pair
        # would no longer fit, and the failure would be a shape error inside
        # the model rather than anything readable.
        height, width = COMPOSITE_SIZE
        self.assertEqual(height, width)
        self.assertEqual(height * 2, 448)

    def test_every_tile_divides_the_frame_exactly(self):
        rows, cols = composite_grid(len(COMPOSITES["tactile_quad"]))
        height, width = COMPOSITE_SIZE
        self.assertEqual(height % rows, 0)
        self.assertEqual(width % cols, 0)


class TestCompositeMetadata(unittest.TestCase):
    def _info(self, cameras):
        return {
            "features": {
                f"observation.images.{c}": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {"video.height": 480, "video.width": 640, "video.fps": 30},
                }
                for c in cameras
            }
        }

    def test_the_composite_is_described_as_an_ordinary_camera(self):
        # LeRobot must open it as a camera and the policy must never be told it
        # is a tiling; only the frame size differs from its parts.
        parts = [f"observation.images.{p}" for p in COMPOSITES["tactile_quad"]]
        info = self._info(COMPOSITES["tactile_quad"])
        feature = composite_feature(info, parts, (224, 224))
        self.assertEqual(feature["dtype"], "video")
        self.assertEqual(feature["shape"], [224, 224, 3])
        self.assertEqual(feature["info"]["video.fps"], 30)
        self.assertEqual(feature["info"]["video.height"], 224)

    def test_a_view_names_the_composite_and_drops_its_parts(self):
        info = self._info(["central", *COMPOSITES["tactile_quad"]])
        keep, composites = split_selection(info, ["central", "tactile_quad"])
        out = filtered_info(info, keep, composites)
        names = {short_name(k) for k in camera_keys(out)}
        self.assertEqual(names, {"central", "tactile_quad"})

    def test_all_never_pulls_in_a_composite(self):
        # Spending one view on four cameras is a choice about what the policy
        # sees, so it has to be asked for.
        info = self._info(["central", *COMPOSITES["tactile_quad"]])
        keep, composites = split_selection(info, ["all"])
        self.assertEqual(composites, [])
        self.assertEqual(len(keep), 5)

    def test_a_composite_the_dataset_cannot_build_is_refused_by_part(self):
        info = self._info(["central", *COMPOSITES["tactile_quad"][:2]])
        with self.assertRaises(ViewError) as caught:
            split_selection(info, ["tactile_quad"])
        self.assertIn(COMPOSITES["tactile_quad"][2], str(caught.exception))


class TestCompositeStats(unittest.TestCase):
    """Derived from the parts', because equal-area tiles allow it exactly."""

    def _stats(self, mean, std):
        return {
            "mean": [[[mean]]],
            "std": [[[std]]],
            "min": [[[0.0]]],
            "max": [[[1.0]]],
            "count": [10],
        }

    def test_the_mean_of_equal_tiles_is_the_mean_of_their_means(self):
        stats = {"a": self._stats(0.2, 0.1), "b": self._stats(0.4, 0.1)}
        out = composite_stats(stats, ["a", "b"])
        self.assertAlmostEqual(out["mean"][0][0][0], 0.3)

    def test_the_spread_grows_when_the_tiles_differ(self):
        # Two tiles 0.2 apart are more varied together than either alone, and a
        # std that ignored that would normalise the tiled frame wrongly.
        same = composite_stats(
            {"a": self._stats(0.3, 0.1), "b": self._stats(0.3, 0.1)}, ["a", "b"]
        )
        apart = composite_stats(
            {"a": self._stats(0.2, 0.1), "b": self._stats(0.4, 0.1)}, ["a", "b"]
        )
        self.assertAlmostEqual(same["std"][0][0][0], 0.1)
        self.assertGreater(apart["std"][0][0][0], same["std"][0][0][0])

    def test_the_extremes_carry_straight_over(self):
        stats = {"a": self._stats(0.2, 0.1), "b": self._stats(0.4, 0.1)}
        stats["b"]["max"] = [[[0.5]]]
        out = composite_stats(stats, ["a", "b"])
        self.assertEqual(out["max"][0][0][0], 1.0)
        self.assertEqual(out["min"][0][0][0], 0.0)

    def test_the_shape_is_the_one_lerobot_writes(self):
        out = composite_stats(
            {"a": self._stats(0.2, 0.1), "b": self._stats(0.4, 0.1)}, ["a", "b"]
        )
        for key in ("mean", "std", "min", "max"):
            self.assertEqual(out[key], [[[out[key][0][0][0]]]], key)

    def test_statistics_that_cannot_be_read_give_none_not_an_exception(self):
        # They only normalise an image; a dataset whose stats are shaped
        # differently should still produce a composite.
        self.assertIsNone(composite_stats({"a": {"mean": "?"}}, ["a"]))
        self.assertIsNone(composite_stats({}, ["a"]))
