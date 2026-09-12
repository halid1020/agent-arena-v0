"""The attribution methods' arithmetic, and the conventions that carry meaning.

No checkpoint is loaded here: what is checked is everything that can be wrong
without a GPU -- the baselines, the chunk metric, the phase segmentation, the
rank agreement two methods are compared with, and the completeness axiom that
makes integrated gradients' shares comparable at all.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_analysis_methods
"""

import unittest

import numpy as np

from actoris_harena.action_layout import set_gripper_columns
from actoris_harena.analysis import phases
from actoris_harena.analysis.attention import (
    by_stream,
    deviation,
    summarise,
    uniform_share,
)
from actoris_harena.analysis.gradients import (
    _ranks,
    completeness_error,
    rank_agreement,
    shares,
    target_gripper,
    target_norm,
)
from actoris_harena.analysis.perturb import (
    BASELINES,
    baseline_frame,
    baseline_state,
    chunk_delta,
)
from actoris_harena.analysis.sources import parse_range
from actoris_harena.analysis.streams import Span, Stream

# The bench these fixtures describe: five body joints then a gripper, per arm,
# so the two gripper channels of the 12-D action are 5 and 11. Declared here
# rather than defaulted in the package -- a stale default would make `perturb`
# occlude a joint on a seven-wide action and report a number nobody could tell
# was wrong.
set_gripper_columns((5, 11))


def frame(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (32, 48, 3), dtype=np.uint8)


class TestTheBaselines(unittest.TestCase):
    """There is no neutral image, so which one was used is part of the result."""

    def test_zeros_is_black_and_mean_keeps_the_brightness(self):
        f = frame()
        self.assertEqual(baseline_frame(f, "zeros").max(), 0)
        replaced = baseline_frame(f, "mean")
        self.assertAlmostEqual(float(replaced.mean()), float(f.mean()), delta=1.0)
        self.assertEqual(replaced.std(axis=(0, 1)).max(), 0.0)  # no structure left

    def test_blur_keeps_the_layout_and_destroys_the_detail(self):
        f = frame()
        blurred = baseline_frame(f, "blur")
        self.assertLess(blurred.std(), f.std())
        self.assertGreater(blurred.std(), 0.0)

    def test_shuffle_needs_a_real_frame_of_the_same_shape(self):
        f = frame()
        with self.assertRaises(ValueError):
            baseline_frame(f, "shuffle")
        with self.assertRaises(ValueError):
            baseline_frame(f, "shuffle", np.zeros((4, 4, 3), np.uint8))
        self.assertTrue(
            np.array_equal(baseline_frame(f, "shuffle", frame(1)), frame(1))
        )

    def test_an_unknown_baseline_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            baseline_frame(frame(), "greyscale")
        self.assertIn("greyscale", str(caught.exception))
        self.assertEqual(set(BASELINES), {"zeros", "mean", "blur", "shuffle"})

    def test_a_state_has_no_blur_so_it_falls_back_to_zeros(self):
        state = np.arange(12.0)
        self.assertTrue(np.array_equal(baseline_state(state, "blur"), np.zeros(12)))
        self.assertTrue(np.allclose(baseline_state(state, "mean"), state.mean()))


class TestTheChunkMetric(unittest.TestCase):
    def test_an_unchanged_plan_moved_nowhere(self):
        chunk = np.random.default_rng(0).normal(size=(20, 12))
        delta = chunk_delta(chunk, chunk)
        self.assertEqual(delta["l2"], 0.0)
        self.assertAlmostEqual(delta["cosine"], 1.0)

    def test_the_grippers_are_reported_apart_from_the_arms(self):
        # A plan that moves ONLY the grippers must not read as barely changed:
        # a grasp is won or lost in those two channels.
        reference = np.zeros((10, 12))
        moved = reference.copy()
        moved[:, [5, 11]] = 1.0
        delta = chunk_delta(reference, moved)
        self.assertAlmostEqual(delta["gripper"], 1.0)
        self.assertGreater(delta["gripper"], delta["l2"] / 12)

    def test_where_in_the_chunk_it_diverged_is_reported(self):
        reference = np.zeros((10, 12))
        moved = reference.copy()
        moved[7] = 5.0
        self.assertEqual(chunk_delta(reference, moved)["argmax_step"], 7)

    def test_chunks_of_different_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            chunk_delta(np.zeros((10, 12)), np.zeros((9, 12)))


class TestScoringOneMethodAgainstAnother(unittest.TestCase):
    def test_ties_share_a_rank(self):
        self.assertEqual(_ranks([3, 1, 2, 1]), [3.0, 0.5, 2.0, 0.5])

    def test_the_same_order_agrees_and_the_reverse_disagrees(self):
        a = {"x": 1, "y": 2, "z": 3}
        self.assertAlmostEqual(rank_agreement(a, {"x": 10, "y": 20, "z": 30}), 1.0)
        self.assertAlmostEqual(rank_agreement(a, {"x": 30, "y": 20, "z": 10}), -1.0)

    def test_too_few_streams_to_correlate_says_so_rather_than_guessing(self):
        self.assertTrue(np.isnan(rank_agreement({"x": 1}, {"x": 2})))

    def test_completeness_is_relative_so_targets_can_be_compared(self):
        self.assertAlmostEqual(completeness_error(10.0, 2.0, 8.0), 0.0)
        self.assertAlmostEqual(completeness_error(10.0, 2.0, 8.8), 0.1)

    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(shares({"a": 1.0, "b": 3.0}).values()), 1.0)

    def test_shares_of_nothing_are_zero_rather_than_a_division_by_zero(self):
        self.assertEqual(shares({"a": 0.0, "b": 0.0}), {"a": 0.0, "b": 0.0})


class TestTheGradientTargets(unittest.TestCase):
    """The scalar a gradient is taken of. Torch tensors: these run in the graph."""

    def test_the_norm_target_grows_with_the_plan(self):
        import torch

        small = target_norm(torch.tensor([[1.0, 0.0]]))
        big = target_norm(torch.tensor([[10.0, 0.0]]))
        self.assertGreater(float(big), float(small))

    def test_the_gripper_target_ignores_the_arm_channels(self):
        # The reason it exists: a huge arm sweep dominates the norm while saying
        # nothing about the two channels a grasp turns on.
        import torch

        chunk = torch.zeros(4, 12)
        chunk[:, 0] = 100.0
        self.assertEqual(float(target_gripper(chunk)), 0.0)
        chunk[:, 5] = 1.0
        self.assertEqual(float(target_gripper(chunk)), 4.0)

    def test_both_targets_stay_differentiable(self):
        import torch

        chunk = torch.zeros(4, 12, requires_grad=True)
        for target in (target_norm, target_gripper):
            self.assertTrue(target(chunk + 1.0).requires_grad)


class TestFoldingAttentionOntoStreams(unittest.TestCase):
    def setUp(self):
        self.spans = [
            Span(Stream("state", "state", "observation.state"), [(1, 2)]),
            Span(Stream("a", "camera", "observation.images.a"), [(2, 5)]),
            Span(Stream("b", "camera", "observation.images.b"), [(5, 8)]),
        ]

    def test_a_streams_tokens_are_summed_not_averaged(self):
        # Attention over tokens sums to one, so summing keeps that property; a
        # mean would make a 300-token camera look negligible beside one state
        # token carrying the same total.
        weights = np.ones((2, 8)) / 8.0
        folded = by_stream(weights, self.spans)
        self.assertAlmostEqual(folded["a"][0], 3 / 8)
        self.assertAlmostEqual(folded["state"][0], 1 / 8)

    def test_uniform_is_the_null_a_deviation_is_measured_against(self):
        null = uniform_share(self.spans, 8)
        self.assertAlmostEqual(null["a"], 3 / 8)
        weights = np.ones((2, 8)) / 8.0
        means = summarise(by_stream(weights, self.spans))
        # Perfectly uniform attention deviates from uniform by nothing, which is
        # the whole point: the raw mass would have read as "a matters 3x state".
        for value in deviation(means, self.spans, 8).values():
            self.assertAlmostEqual(value, 0.0)

    def test_a_consulted_stream_shows_a_positive_deviation(self):
        weights = np.zeros((1, 8))
        weights[0, 2:5] = 1 / 3  # all attention on camera a
        means = summarise(by_stream(weights, self.spans))
        moved = deviation(means, self.spans, 8)
        self.assertGreater(moved["a"], 0.0)
        self.assertLess(moved["b"], 0.0)

    def test_layers_are_averaged_when_there_is_more_than_one(self):
        weights = np.stack([np.ones((2, 8)), np.zeros((2, 8))])
        self.assertAlmostEqual(by_stream(weights, self.spans)["a"][0], 1.5)


class TestSegmentingAnEpisodeByTheGrippers(unittest.TestCase):
    def build(self, grip):
        actions = np.zeros((len(grip), 12))
        actions[:, 5] = grip
        actions[:, 11] = grip
        return actions

    def test_a_close_then_hold_then_open_is_found(self):
        grip = np.concatenate(
            [
                np.full(10, 0.30),
                np.linspace(0.30, 0.05, 6),
                np.full(10, 0.05),
                np.linspace(0.05, 0.30, 6),
                np.full(10, 0.30),
            ]
        )
        labels = phases.segment(self.build(grip))
        self.assertIn("closing", labels)
        self.assertIn("holding", labels)
        self.assertIn("opening", labels)
        # What comes after the last grasp is 'released', not another approach.
        self.assertEqual(labels[-1], "released")
        self.assertEqual(labels[0], "reaching")

    def test_a_still_episode_is_all_one_phase_rather_than_all_transitions(self):
        # Every threshold is a quantile, so a constant trace makes them all zero;
        # without the guard every frame would be called 'closing'.
        labels = phases.segment(self.build(np.full(30, 0.2)))
        self.assertEqual(set(labels), {"reaching"})

    def test_the_thresholds_are_this_episodes_own_and_not_absolute(self):
        # The same shape at a different offset must segment the same way: these
        # grippers work in a band whose position depends on the day.
        shape = np.concatenate(
            [np.full(8, 0.9), np.linspace(0.9, 0.6, 5), np.full(8, 0.6)]
        )
        high = phases.segment(self.build(shape))
        low = phases.segment(self.build(shape - 0.55))
        self.assertEqual(high, low)

    def test_spans_collapse_runs_for_shading_a_plot(self):
        spans = phases.spans_of(["a", "a", "b", "a"])
        self.assertEqual(spans, [("a", 0, 2), ("b", 2, 3), ("a", 3, 4)])

    def test_the_mean_within_each_phase_is_reported(self):
        by = phases.by_phase(["reaching", "holding", "holding"], [1.0, 2.0, 4.0])
        self.assertEqual(by, {"reaching": 1.0, "holding": 3.0})

    def test_too_short_to_segment_is_not_an_error(self):
        self.assertEqual(
            phases.segment(self.build(np.array([0.1, 0.2]))), ["reaching"] * 2
        )
        self.assertEqual(phases.segment(np.zeros((0, 12))), [])


class TestChoosingEpisodes(unittest.TestCase):
    def test_a_range_and_a_list_both_work(self):
        self.assertEqual(parse_range("0-3,7"), [0, 1, 2, 3, 7])

    def test_nothing_means_every_episode(self):
        self.assertEqual(parse_range(""), [])
        self.assertEqual(parse_range(None), [])


if __name__ == "__main__":
    unittest.main()
