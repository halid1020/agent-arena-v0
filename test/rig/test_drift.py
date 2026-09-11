"""Unit tests for the per-frame drift telemetry (common/recording/drift)."""

import unittest

from actoris_harena.recording.drift import DriftLog


class TestDriftLog(unittest.TestCase):
    def test_empty(self):
        d = DriftLog()
        self.assertEqual(len(d), 0)
        self.assertEqual(d.summary(), {})
        self.assertEqual(d.format_summary(), "no drift samples")
        self.assertIsNone(d.write_parquet("/tmp/does-not-matter", 0))

    def test_summary_signed_mean_abs_extremes(self):
        d = DriftLog()
        # drifts in seconds; summary reports milliseconds.
        d.add(1, 0.0, {"joints": 0.010, "cam": -0.004})
        d.add(2, 0.033, {"joints": -0.030, "cam": 0.002})
        s = d.summary()
        # joints: mean of (+10, -30) ms = -10 ms; abs extremes 10 and 30.
        self.assertAlmostEqual(s["joints"]["mean_ms"], -10.0, places=6)
        self.assertAlmostEqual(s["joints"]["max_abs_ms"], 30.0, places=6)
        # p95 over {10, 30} abs is close to the max.
        self.assertGreater(s["joints"]["p95_abs_ms"], 28.0)
        self.assertAlmostEqual(s["cam"]["mean_ms"], -1.0, places=6)
        self.assertAlmostEqual(s["cam"]["max_abs_ms"], 4.0, places=6)

    def test_reset_clears(self):
        d = DriftLog()
        d.add(1, 0.0, {"joints": 0.01})
        self.assertEqual(len(d), 1)
        d.reset()
        self.assertEqual(len(d), 0)

    def test_format_summary_orders_worst_first(self):
        d = DriftLog()
        d.add(1, 0.0, {"tight": 0.001, "loose": 0.050})
        text = d.format_summary()
        # The worst (highest p95 abs) stream appears first.
        self.assertLess(text.index("loose"), text.index("tight"))

    def test_write_parquet_roundtrip(self):
        import tempfile

        import pyarrow.parquet as pq

        d = DriftLog()
        d.add(1, 100.0, {"joints": 0.01, "cam": -0.002})
        d.add(2, 100.033, {"joints": 0.008, "cam": 0.001})
        with tempfile.TemporaryDirectory() as tmp:
            path = d.write_parquet(tmp, 7)
            self.assertIsNotNone(path)
            self.assertTrue(str(path).endswith("drift_000007.parquet"))
            table = pq.read_table(path)
            cols = set(table.column_names)
            self.assertEqual(
                cols, {"frame_index", "t_ref", "drift_ms_joints", "drift_ms_cam"}
            )
            self.assertEqual(table.num_rows, 2)


if __name__ == "__main__":
    unittest.main()
