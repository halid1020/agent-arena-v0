"""Unit tests for the per-episode reader (actoris_harena.recording.dataset_read).

Covers the layout arithmetic that turns an episode's metadata row into the files
holding its frames, and the fault isolation that lets one damaged or still-open
file cost only the episodes inside it. No LeRobot and no video decoding here; the
decode pass is exercised against real recordings instead.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_dataset_read
"""

import json
import tempfile
import unittest
from pathlib import Path

from actoris_harena.recording.dataset_read import (
    camera_label,
    episode_data_path,
    episode_video_path,
    episode_window,
    playable_window,
    read_episode_row,
    video_keys,
)

_CAM = "observation.images.central"


def _row(chunk=0, file=2, start=44.4, end=58.7, data_chunk=0, data_file=1):
    return {
        "episode_index": 3,
        "data/chunk_index": data_chunk,
        "data/file_index": data_file,
        f"videos/{_CAM}/chunk_index": chunk,
        f"videos/{_CAM}/file_index": file,
        f"videos/{_CAM}/from_timestamp": start,
        f"videos/{_CAM}/to_timestamp": end,
    }


class TestLayoutPaths(unittest.TestCase):
    def test_video_path_uses_the_row_indices_not_the_episode_number(self):
        # Several episodes share one video file, so the file is named by the
        # row's chunk/file index; episode 3 living in file-002 is normal.
        path = episode_video_path(Path("/ds"), _row(), _CAM)
        self.assertEqual(path, Path(f"/ds/videos/{_CAM}/chunk-000/file-002.mp4"))

    def test_video_path_zero_pads_to_three_digits(self):
        path = episode_video_path(Path("/ds"), _row(chunk=1, file=12), _CAM)
        self.assertTrue(path.as_posix().endswith("chunk-001/file-012.mp4"))

    def test_data_path_is_independent_of_the_video_indices(self):
        self.assertEqual(
            episode_data_path(Path("/ds"), _row()),
            Path("/ds/data/chunk-000/file-001.parquet"),
        )

    def test_window_is_the_episode_slice_inside_the_shared_file(self):
        self.assertEqual(episode_window(_row(), _CAM), (44.4, 58.7))

    def test_label_drops_the_feature_namespace(self):
        self.assertEqual(camera_label(_CAM), "central")
        self.assertEqual(camera_label("wrist_camera_left"), "wrist_camera_left")


class TestPlayableWindow(unittest.TestCase):
    """Where an episode's own frames stop, as opposed to where its window does."""

    def test_the_last_frame_is_one_period_inside_the_recorded_end(self):
        # The recorded end is exclusive; the frame sitting on it is the next
        # recording's first, which is what a viewer stopping there displayed.
        start, last = playable_window(_row(start=44.4667, end=58.7333), _CAM, 30)
        self.assertAlmostEqual(start, 44.4667)
        self.assertAlmostEqual(last, 58.7333 - 1 / 30)

    def test_consecutive_episodes_no_longer_overlap(self):
        # Their RAW windows touch exactly -- that is the whole trap.
        first = _row(start=44.4667, end=58.7333)
        second = _row(start=58.7333, end=77.2000)
        self.assertEqual(
            episode_window(first, _CAM)[1], episode_window(second, _CAM)[0]
        )
        self.assertLess(
            playable_window(first, _CAM, 30)[1], playable_window(second, _CAM, 30)[0]
        )

    def test_the_span_covers_one_frame_fewer_than_the_raw_window(self):
        row = _row(start=0.0, end=10.0)
        raw = episode_window(row, _CAM)
        play = playable_window(row, _CAM, 30)
        self.assertAlmostEqual((raw[1] - raw[0]) - (play[1] - play[0]), 1 / 30)

    def test_a_single_frame_episode_collapses_rather_than_inverting(self):
        start, last = playable_window(_row(start=5.0, end=5.0 + 1 / 30), _CAM, 30)
        self.assertEqual(start, last)

    def test_a_zero_length_window_cannot_go_backwards(self):
        start, last = playable_window(_row(start=5.0, end=5.0), _CAM, 30)
        self.assertEqual((start, last), (5.0, 5.0))

    def test_a_missing_rate_leaves_the_window_alone(self):
        self.assertEqual(playable_window(_row(start=1.0, end=2.0), _CAM, 0)[1], 2.0)


def _write_info(root, features):
    path = Path(root) / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fps": 30, "features": features}))


class TestVideoKeys(unittest.TestCase):
    def test_reads_camera_streams_in_recorded_order(self):
        with tempfile.TemporaryDirectory() as d:
            _write_info(
                d,
                {
                    "observation.state": {"dtype": "float32"},
                    "observation.images.central": {"dtype": "video"},
                    "observation.images.wrist_camera_left": {"dtype": "video"},
                },
            )
            self.assertEqual(
                video_keys(d),
                [
                    "observation.images.central",
                    "observation.images.wrist_camera_left",
                ],
            )

    def test_depth_streams_are_not_playable_streams(self):
        with tempfile.TemporaryDirectory() as d:
            _write_info(
                d,
                {
                    "observation.images.central": {"dtype": "video"},
                    "observation.images.central_depth": {
                        "dtype": "video",
                        "info": {"is_depth_map": True},
                    },
                },
            )
            self.assertEqual(video_keys(d), ["observation.images.central"])

    def test_missing_declaration_reads_as_no_cameras(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(video_keys(d), [])


def _write_rows(root, rel, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {k: [r[k] for r in rows] for k in rows[0]}
    pq.write_table(pa.table(columns), path)


class TestReadEpisodeRow(unittest.TestCase):
    def test_finds_the_row_in_whichever_file_holds_it(self):
        with tempfile.TemporaryDirectory() as d:
            _write_rows(
                d,
                "meta/episodes/chunk-000/file-000.parquet",
                [{"episode_index": 0, "length": 10}],
            )
            _write_rows(
                d,
                "meta/episodes/chunk-000/file-001.parquet",
                [{"episode_index": 1, "length": 20}],
            )
            self.assertEqual(read_episode_row(d, 1)["length"], 20)

    def test_a_damaged_file_costs_only_its_own_episodes(self):
        # The point of reading file by file: an episode still being written must
        # not make an earlier, complete episode unreadable.
        with tempfile.TemporaryDirectory() as d:
            _write_rows(
                d,
                "meta/episodes/chunk-000/file-000.parquet",
                [{"episode_index": 0, "length": 10}],
            )
            bad = Path(d) / "meta/episodes/chunk-000/file-001.parquet"
            bad.write_text("not a parquet footer")
            self.assertEqual(read_episode_row(d, 0)["length"], 10)
            self.assertIsNone(read_episode_row(d, 1))

    def test_stats_columns_are_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            _write_rows(
                d,
                "meta/episodes/chunk-000/file-000.parquet",
                [{"episode_index": 0, "length": 10, "stats/action/mean": 1.0}],
            )
            row = read_episode_row(d, 0)
            self.assertIn("length", row)
            self.assertNotIn("stats/action/mean", row)

    def test_unknown_episode_and_missing_directory_read_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(read_episode_row(d, 4))
            _write_rows(
                d,
                "meta/episodes/chunk-000/file-000.parquet",
                [{"episode_index": 0, "length": 10}],
            )
            self.assertIsNone(read_episode_row(d, 4))


if __name__ == "__main__":
    unittest.main()
