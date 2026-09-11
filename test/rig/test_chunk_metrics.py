"""Unit tests for the numbers a chunking strategy is judged on.

The seam ratio is the load-bearing one: it must be near 1 for a trajectory whose
joins are invisible and large for one that flinches at every chunk boundary, and
it must refuse to answer when the run cannot support the comparison.
"""

import unittest

import numpy as np

from actoris_harena.deploy.chunk_metrics import (
    compare,
    path_length,
    seam_ratio,
    step_sizes,
    summarise,
)


def smooth(ticks: int, dim: int = 3) -> np.ndarray:
    """A trajectory that moves the same amount every tick."""
    return np.tile(np.arange(ticks, dtype=float).reshape(-1, 1), (1, dim))


class TestStepSizes(unittest.TestCase):
    def test_a_constant_march_has_constant_steps(self):
        steps = step_sizes(smooth(5, dim=1))
        np.testing.assert_allclose(steps, np.ones(4))

    def test_a_single_tick_has_no_transitions(self):
        self.assertEqual(len(step_sizes(np.zeros((1, 12)))), 0)

    def test_a_trajectory_that_is_not_a_table_is_refused(self):
        with self.assertRaises(ValueError):
            step_sizes(np.zeros(12))


class TestSeamRatio(unittest.TestCase):
    def test_an_invisible_join_scores_about_one(self):
        # Every tick moves by 1, including the ones a chunk boundary falls on.
        self.assertAlmostEqual(seam_ratio(smooth(30, dim=1), [10, 20]), 1.0)

    def test_a_flinch_at_every_boundary_is_reported(self):
        actions = smooth(30, dim=1)
        for seam in (10, 20):
            actions[seam:] += 4.0  # a step change of 5 instead of 1
        self.assertAlmostEqual(seam_ratio(actions, [10, 20]), 5.0)

    def test_it_compares_the_step_INTO_the_seam(self):
        # A jump placed one tick late must NOT be attributed to the seam.
        actions = smooth(30, dim=1)
        actions[11:] += 9.0
        ratio = seam_ratio(actions, [10, 20])
        self.assertIsNotNone(ratio)
        self.assertLess(ratio, 2.0)

    def test_too_few_seams_is_not_a_result(self):
        self.assertIsNone(seam_ratio(smooth(30, dim=1), [10]))
        self.assertIsNone(seam_ratio(smooth(30, dim=1), []))

    def test_a_trajectory_that_never_moved_is_not_a_result(self):
        # Otherwise the ratio is a division of noise by noise and reads as data.
        self.assertIsNone(seam_ratio(np.zeros((30, 12)), [10, 20]))

    def test_seams_outside_the_trajectory_are_ignored(self):
        self.assertIsNone(seam_ratio(smooth(10, dim=1), [0, 99]))


class TestPathLength(unittest.TestCase):
    def test_thrashing_covers_more_ground_than_going_straight(self):
        straight = smooth(11, dim=1)
        thrash = np.array([[i % 2] for i in range(11)], dtype=float)
        self.assertGreater(path_length(thrash), 0.0)
        self.assertAlmostEqual(path_length(straight), 10.0)


class TestSummarise(unittest.TestCase):
    def test_it_reports_what_the_rollout_cost(self):
        row = summarise(
            smooth(30, dim=1),
            [10, 20],
            held_ticks=6,
            total_ticks=30,
            success=True,
            round_trips_s=[0.5, 0.7, 0.6],
        )
        self.assertEqual(row["chunks"], 2)
        self.assertTrue(row["success"])
        self.assertAlmostEqual(row["held_fraction"], 0.2)
        self.assertAlmostEqual(row["round_trip_ms_median"], 600.0)
        self.assertAlmostEqual(row["round_trip_ms_max"], 700.0)

    def test_a_local_run_has_no_round_trip_to_report(self):
        row = summarise(smooth(5, dim=1), [], total_ticks=5)
        self.assertIsNone(row["round_trip_ms_median"])
        self.assertIsNone(row["success"])

    def test_no_ticks_is_not_a_division_by_zero(self):
        self.assertEqual(
            summarise(np.zeros((0, 12)), [], total_ticks=0)["held_fraction"], 0.0
        )


class TestCompare(unittest.TestCase):
    def test_the_smoothest_strategy_is_listed_first(self):
        table = compare(
            [
                {"strategy": "append", "seam_ratio": 8.0, "success": False},
                {"strategy": "blend", "seam_ratio": 1.1, "success": True},
                {"strategy": "replace", "seam_ratio": 3.4, "success": True},
            ]
        )
        order = [line.split("|")[2].strip() for line in table.splitlines()[2:]]
        self.assertEqual(order, ["blend", "replace", "append"])

    def test_a_strategy_with_no_ratio_sorts_last_rather_than_first(self):
        # Absent must not masquerade as perfect.
        table = compare(
            [
                {"strategy": "unmeasured", "seam_ratio": None},
                {"strategy": "measured", "seam_ratio": 5.0},
            ]
        )
        order = [line.split("|")[2].strip() for line in table.splitlines()[2:]]
        self.assertEqual(order, ["measured", "unmeasured"])

    def test_a_trial_that_was_counted_is_told_apart_from_one_that_was_not(self):
        # Two attempts under one splice are two rows, not an average: the scene
        # was put back by hand in between, and that is usually the whole point.
        table = compare(
            [
                {"strategy": "blend", "seam_ratio": 2.0, "trial": 0},
                {"strategy": "blend", "seam_ratio": 1.0, "trial": 1},
                {"strategy": "blend", "seam_ratio": 3.0},
            ]
        )
        trials = [line.split("|")[1].strip() for line in table.splitlines()[2:]]
        self.assertEqual(trials, ["1", "0", "—"])

    def test_no_runs_says_so(self):
        self.assertIn("no runs", compare([]))


if __name__ == "__main__":
    unittest.main()
