"""Unit tests for the grid a chunking sweep runs.

The grid is the experiment design, so the things worth pinning are the ones a
wrong grid would quietly get away with: a cell carrying a parameter its strategy
never reads (two rows of the same run under different labels), ``rtc`` sneaking
into a sweep against a host that cannot do it (a duplicate of ``replace`` under
a name nobody could trust), and a malformed spec being absorbed instead of
refused an hour before the sweep starts.
"""

import unittest

from actoris_harena.deploy.chunk_sweep import (
    CONSUMES,
    DEFAULTS,
    SweepSpecError,
    cell_label,
    expand_grid,
)


class TestBuiltInGrid(unittest.TestCase):
    def setUp(self):
        self.cells = expand_grid("all")

    def test_the_grid_is_seventeen_cells(self):
        self.assertEqual(len(self.cells), 17)

    def test_rtc_is_excluded(self):
        # Its smoothing is the host's job; against a host that does not do it,
        # the client-side splice is exactly 'replace'.
        self.assertNotIn("rtc", [c["strategy"] for c in self.cells])

    def test_each_strategy_appears_the_expected_number_of_times(self):
        counts: dict = {}
        for cell in self.cells:
            counts[cell["strategy"]] = counts.get(cell["strategy"], 0) + 1
        self.assertEqual(
            counts,
            {
                "sync": 1,
                "append": 1,
                "replace": 1,
                "receding": 4,
                "blend": 6,
                "ensemble": 4,
            },
        )

    def test_a_cell_carries_only_what_its_strategy_reads(self):
        for cell in self.cells:
            params = set(cell) - {"strategy"}
            self.assertEqual(
                params,
                set(CONSUMES[cell["strategy"]]),
                f"{cell} carries the wrong parameters",
            )

    def test_the_axes_are_the_ones_the_sweep_argues_about(self):
        ratios = [c["execute_ratio"] for c in self.cells if c["strategy"] == "receding"]
        self.assertEqual(sorted(ratios), [0.25, 0.5, 0.75, 1.0])
        weights = [c["new_weight"] for c in self.cells if c["strategy"] == "ensemble"]
        self.assertEqual(sorted(weights), [0.3, 0.5, 0.7, 0.9])
        blends = sorted(
            (c["blend_window"], c["ramp_kind"])
            for c in self.cells
            if c["strategy"] == "blend"
        )
        self.assertEqual(
            blends,
            [
                (3, "exp"),
                (3, "linear"),
                (5, "exp"),
                (5, "linear"),
                (10, "exp"),
                (10, "linear"),
            ],
        )

    def test_defaults_do_not_move_the_built_in_grid(self):
        # Every parameter in the built-in grid is stated, so a tool's flags
        # cannot change what "all" means from one machine to the next.
        moved = expand_grid(
            "all",
            defaults={
                "execute_ratio": 0.9,
                "blend_window": 99,
                "new_weight": 0.1,
                "ramp_kind": "exp",
            },
        )
        self.assertEqual(moved, self.cells)


class TestCellLabel(unittest.TestCase):
    def test_labels_are_unique_across_the_built_in_grid(self):
        labels = [cell_label(c) for c in expand_grid("all")]
        self.assertEqual(len(set(labels)), len(labels))

    def test_labels_are_stable_and_readable(self):
        self.assertEqual(cell_label({"strategy": "append"}), "append")
        self.assertEqual(
            cell_label({"strategy": "receding", "execute_ratio": 0.25}),
            "receding@0.25",
        )
        self.assertEqual(
            cell_label({"strategy": "blend", "blend_window": 5, "ramp_kind": "exp"}),
            "blend@w5/exp",
        )
        self.assertEqual(
            cell_label({"strategy": "ensemble", "new_weight": 0.7}), "ensemble@0.7"
        )

    def test_a_label_depends_on_nothing_but_the_cell(self):
        cell = {"strategy": "blend", "blend_window": 3, "ramp_kind": "linear"}
        self.assertEqual(
            cell_label(cell), cell_label(dict(reversed(list(cell.items()))))
        )


class TestCustomSpecs(unittest.TestCase):
    def test_a_bare_strategy_is_one_cell_at_its_defaults(self):
        self.assertEqual(expand_grid("append"), [{"strategy": "append"}])
        self.assertEqual(
            expand_grid("receding"),
            [{"strategy": "receding", "execute_ratio": DEFAULTS["execute_ratio"]}],
        )

    def test_defaults_fill_in_what_a_term_does_not_name(self):
        cells = expand_grid("blend:ramp_kind=exp", defaults={"blend_window": 7})
        self.assertEqual(
            cells, [{"strategy": "blend", "blend_window": 7, "ramp_kind": "exp"}]
        )

    def test_terms_are_cartesian_producted_and_concatenated(self):
        cells = expand_grid(
            "receding:execute_ratio=0.5,1.0;blend:blend_window=5:ramp_kind=linear,exp"
        )
        self.assertEqual(
            cells,
            [
                {"strategy": "receding", "execute_ratio": 0.5},
                {"strategy": "receding", "execute_ratio": 1.0},
                {"strategy": "blend", "blend_window": 5, "ramp_kind": "linear"},
                {"strategy": "blend", "blend_window": 5, "ramp_kind": "exp"},
            ],
        )

    def test_values_are_typed_by_the_parameter(self):
        cell = expand_grid("blend:blend_window=10")[0]
        self.assertIsInstance(cell["blend_window"], int)
        self.assertIsInstance(
            expand_grid("ensemble:new_weight=1")[0]["new_weight"], float
        )

    def test_a_duplicated_cell_is_dropped(self):
        # Two identical runs under one label would be one experiment counted
        # twice in the ranking.
        self.assertEqual(expand_grid("append;append"), [{"strategy": "append"}])

    def test_rtc_can_still_be_asked_for_explicitly(self):
        self.assertEqual(expand_grid("rtc"), [{"strategy": "rtc"}])


class TestRefusals(unittest.TestCase):
    def assert_refused(self, spec: str, *needles: str):
        with self.assertRaises(SweepSpecError) as caught:
            expand_grid(spec)
        message = str(caught.exception)
        for needle in needles:
            self.assertIn(needle, message)
        # A ValueError, so a caller need not import this module to catch it.
        self.assertIsInstance(caught.exception, ValueError)

    def test_an_empty_spec_is_refused(self):
        self.assert_refused("", "empty")

    def test_an_unknown_strategy_names_its_term(self):
        self.assert_refused("blend;wobble:blend_window=3", "wobble:blend_window=3")

    def test_a_parameter_the_strategy_does_not_read_names_its_term(self):
        self.assert_refused(
            "blend:new_weight=0.3", "blend:new_weight=0.3", "does not read"
        )

    def test_an_unknown_parameter_names_its_term(self):
        self.assert_refused("blend:widow=3", "blend:widow=3", "widow")

    def test_a_value_outside_the_splice_range_is_refused_at_parse_time(self):
        self.assert_refused("receding:execute_ratio=1.5", "receding:execute_ratio=1.5")
        self.assert_refused("ensemble:new_weight=-0.1", "new_weight")
        self.assert_refused("blend:blend_window=0", "blend_window")
        self.assert_refused("blend:ramp_kind=cosine", "cosine")

    def test_a_parameter_without_a_value_is_refused(self):
        self.assert_refused("blend:blend_window", "blend:blend_window")
        self.assert_refused("blend:blend_window=", "blend:blend_window=")

    def test_a_parameter_given_twice_is_refused(self):
        self.assert_refused("blend:blend_window=3:blend_window=5", "twice")

    def test_an_empty_term_is_refused(self):
        self.assert_refused("append;;replace", "empty term")


if __name__ == "__main__":
    unittest.main()
