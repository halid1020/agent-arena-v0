"""Unit tests for the chunk splice strategies.

Pure numpy: what each strategy does with the queue it inherits and the chunk
that just landed. The point of the module is that a plan is executed at the
moment it was written for, so most of these assert on WHICH rows survive, not
on smoothness.
"""

import unittest

import numpy as np

from actoris_harena.deploy.chunking import (
    DEFAULT_BLEND_WINDOW,
    DEFAULT_EXECUTE_RATIO,
    STRATEGIES,
    ChunkingError,
    delay_ticks,
    plan_lag,
    ramp,
    splice,
)


def ladder(rows: int, start: float = 0.0, dim: int = 2) -> np.ndarray:
    """A chunk whose every row is its own index, so drops are visible."""
    return np.tile(np.arange(start, start + rows, dtype=float).reshape(-1, 1), (1, dim))


class TestDelayTicks(unittest.TestCase):
    def test_a_round_trip_rounds_up_to_whole_ticks(self):
        # 0.6 s at 30 Hz is 18 ticks exactly; 0.61 s must not round down.
        self.assertEqual(delay_ticks(0.6, 30.0), 18)
        self.assertEqual(delay_ticks(0.61, 30.0), 19)

    def test_an_unmeasured_link_has_no_delay(self):
        self.assertEqual(delay_ticks(0.0, 30.0), 0)
        self.assertEqual(delay_ticks(0.5, 0.0), 0)


class TestRamp(unittest.TestCase):
    def test_it_starts_on_the_old_plan(self):
        # Weight 0 for the new plan at the seam is what makes the join
        # continuous: the first action is the one already going to be executed.
        for kind in ("linear", "exp"):
            self.assertEqual(ramp(6, kind)[0], 0.0)

    def test_it_rises_towards_the_new_plan_without_reaching_it_early(self):
        weights = ramp(5, "linear")
        self.assertTrue(np.all(np.diff(weights) > 0))
        self.assertLess(weights[-1], 1.0)

    def test_exp_leaves_the_old_plan_more_slowly_than_linear(self):
        linear, exponential = ramp(8, "linear"), ramp(8, "exp")
        self.assertTrue(np.all(exponential[1:] < linear[1:]))

    def test_degenerate_windows_are_not_an_error(self):
        self.assertEqual(len(ramp(0)), 0)
        np.testing.assert_array_equal(ramp(1), np.zeros(1))

    def test_an_unknown_ramp_is_refused(self):
        with self.assertRaises(ChunkingError):
            ramp(4, "cubic")


class TestAppend(unittest.TestCase):
    """The behaviour the rig had before this module, kept so it can be beaten."""

    def test_nothing_is_dropped_however_late_the_chunk(self):
        out = splice("append", ladder(4), ladder(6, start=100), delay=3)
        self.assertEqual(len(out), 10)
        np.testing.assert_allclose(out[:4], ladder(4))
        np.testing.assert_allclose(out[4:], ladder(6, start=100))

    def test_an_empty_queue_just_takes_the_chunk(self):
        out = splice("append", np.zeros((0, 2)), ladder(3), delay=2)
        np.testing.assert_allclose(out, ladder(3))


class TestSync(unittest.TestCase):
    def test_the_whole_chunk_is_executed_from_its_first_action(self):
        out = splice("sync", np.zeros((0, 2)), ladder(5), delay=9)
        np.testing.assert_allclose(out, ladder(5))

    def test_a_leftover_means_the_caller_was_not_synchronous(self):
        with self.assertRaises(ChunkingError):
            splice("sync", ladder(2), ladder(5))


class TestReceding(unittest.TestCase):
    """Execute a fraction of the plan, then go and look again."""

    def test_it_keeps_the_ratio_of_whatever_length_came_back(self):
        # A fraction, not a count, so it means the same against ACT's hundred
        # actions and diffusion's thirty-two.
        out = splice("receding", np.zeros((0, 2)), ladder(12), execute_ratio=0.5)
        self.assertEqual(len(out), 6)
        np.testing.assert_allclose(out, ladder(6))
        self.assertEqual(
            len(splice("receding", np.zeros((0, 2)), ladder(100), execute_ratio=0.25)),
            25,
        )

    def test_a_ratio_of_one_is_the_whole_chunk(self):
        out = splice("receding", np.zeros((0, 2)), ladder(9), execute_ratio=1.0)
        np.testing.assert_allclose(out, ladder(9))

    def test_it_always_executes_at_least_one_action(self):
        # Otherwise a small ratio on a short chunk makes no progress at all and
        # the run stalls looking busy.
        out = splice("receding", np.zeros((0, 2)), ladder(3), execute_ratio=0.01)
        self.assertEqual(len(out), 1)

    def test_a_leftover_means_the_caller_was_not_blocking(self):
        with self.assertRaises(ChunkingError):
            splice("receding", ladder(2), ladder(8))

    def test_a_ratio_outside_the_unit_interval_is_refused(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ChunkingError, msg=str(bad)):
                splice("receding", np.zeros((0, 2)), ladder(8), execute_ratio=bad)

    def test_the_delay_does_not_enter_into_it(self):
        # The arms held still for the reply, so row 0 is still the right start.
        a = splice("receding", np.zeros((0, 2)), ladder(8), delay=5, execute_ratio=0.5)
        b = splice("receding", np.zeros((0, 2)), ladder(8), delay=0, execute_ratio=0.5)
        np.testing.assert_allclose(a, b)

    def test_the_default_ratio_is_half(self):
        self.assertEqual(DEFAULT_EXECUTE_RATIO, 0.5)


class TestReplace(unittest.TestCase):
    def test_the_rows_whose_moment_has_passed_are_dropped(self):
        out = splice("replace", ladder(7), ladder(10), delay=4)
        # Row 4 was planned for the tick we are now at, so it leads.
        np.testing.assert_allclose(out, ladder(6, start=4))

    def test_the_leftovers_do_not_survive(self):
        out = splice("replace", ladder(7, start=900), ladder(10), delay=0)
        self.assertEqual(len(out), 10)
        self.assertLess(out.max(), 900)

    def test_a_round_trip_longer_than_the_chunk_leaves_nothing(self):
        # Legitimate, and the caller must treat it as a stall: every action the
        # policy planned was for a tick that has already gone by.
        out = splice("replace", ladder(3), ladder(5), delay=9)
        self.assertEqual(len(out), 0)
        self.assertEqual(out.shape[1], 2)

    def test_rtc_splices_client_side_exactly_like_replace(self):
        args = (ladder(7), ladder(10), 4)
        np.testing.assert_allclose(splice("rtc", *args), splice("replace", *args))


class TestBlend(unittest.TestCase):
    def test_the_seam_is_continuous_with_what_was_executing(self):
        old = np.full((6, 2), 5.0)
        new = np.full((10, 2), 9.0)
        out = splice("blend", old, new, delay=0, window=4)
        # The first action is exactly the one already queued, so no step change.
        np.testing.assert_allclose(out[0], old[0])

    def test_it_arrives_at_the_new_plan_and_stays_there(self):
        old = np.full((6, 2), 5.0)
        new = np.full((10, 2), 9.0)
        out = splice("blend", old, new, delay=0, window=4)
        np.testing.assert_allclose(out[4:], new[4:])

    def test_the_crossfade_is_monotone_between_the_two_plans(self):
        out = splice("blend", np.full((8, 1), 0.0), np.full((8, 1), 1.0), window=5)
        np.testing.assert_allclose(np.sort(out[:5, 0]), out[:5, 0])

    def test_it_blends_against_the_aligned_chunk_not_the_raw_one(self):
        # With delay 3 the join must start from row 3, not row 0.
        out = splice("blend", ladder(6, start=50), ladder(10), delay=3, window=1)
        self.assertEqual(len(out), 7)
        np.testing.assert_allclose(out[1:], ladder(6, start=4))

    def test_with_no_leftovers_it_is_replace(self):
        out = splice("blend", np.zeros((0, 2)), ladder(9), delay=2)
        np.testing.assert_allclose(out, ladder(7, start=2))

    def test_the_window_never_outruns_the_shorter_side(self):
        out = splice("blend", ladder(2), ladder(9), delay=0, window=99)
        self.assertEqual(len(out), 9)


class TestEnsemble(unittest.TestCase):
    def test_the_overlap_is_averaged_and_the_tail_is_not(self):
        old = np.full((4, 1), 0.0)
        new = np.full((10, 1), 1.0)
        out = splice("ensemble", old, new, delay=0, new_weight=0.7)
        np.testing.assert_allclose(out[:4, 0], 0.7)
        np.testing.assert_allclose(out[4:, 0], 1.0)

    def test_a_conservative_weight_stays_nearer_the_old_plan(self):
        old, new = np.zeros((5, 1)), np.ones((5, 1))
        careful = splice("ensemble", old, new, new_weight=0.3)
        eager = splice("ensemble", old, new, new_weight=0.7)
        self.assertTrue(np.all(careful[:5] < eager[:5]))

    def test_a_weight_outside_the_unit_interval_is_refused(self):
        with self.assertRaises(ChunkingError):
            splice("ensemble", ladder(3), ladder(3), new_weight=1.4)


class TestRefusals(unittest.TestCase):
    def test_an_unknown_strategy_is_refused(self):
        with self.assertRaises(ChunkingError):
            splice("clever", ladder(2), ladder(2))

    def test_an_empty_chunk_is_refused(self):
        with self.assertRaises(ChunkingError):
            splice("replace", ladder(2), np.zeros((0, 2)))

    def test_an_action_width_that_changes_mid_run_is_refused(self):
        for strategy in ("append", "blend", "ensemble"):
            with self.assertRaises(ChunkingError, msg=strategy):
                splice(strategy, ladder(3, dim=12), ladder(3, dim=6))

    def test_a_chunk_that_is_not_a_table_of_actions_is_refused(self):
        with self.assertRaises(ChunkingError):
            splice("replace", ladder(2), np.arange(6, dtype=float))


class TestPlanLag(unittest.TestCase):
    def test_appending_pays_the_round_trip_and_the_queue_it_waits_behind(self):
        # The diffusion case from the docstring: 27 queued, 18 ticks in flight.
        self.assertEqual(plan_lag("append", leftover_len=27, delay=18), 45)

    def test_every_aligning_strategy_executes_the_plan_on_time(self):
        for strategy in ("replace", "blend", "ensemble", "rtc", "sync", "receding"):
            self.assertEqual(plan_lag(strategy, 27, 18), 0, msg=strategy)


class TestEveryStrategyIsUsable(unittest.TestCase):
    def test_each_named_strategy_splices_something(self):
        for strategy in STRATEGIES:
            blocking = strategy in ("sync", "receding")
            leftover = np.zeros((0, 2)) if blocking else ladder(4)
            out = splice(strategy, leftover, ladder(12), delay=2)
            self.assertGreater(len(out), 0, msg=strategy)
            self.assertEqual(out.shape[1], 2, msg=strategy)

    def test_the_default_blend_window_is_shorter_than_a_diffusion_chunk(self):
        self.assertLess(DEFAULT_BLEND_WINDOW, 32)


if __name__ == "__main__":
    unittest.main()
