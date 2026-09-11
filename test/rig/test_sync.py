"""Unit tests for the pure timestamped-sync machinery (common/sync)."""

import unittest

import numpy as np

from actoris_harena.sync import (
    TimestampedHistory,
    interp_pose,
    lerp,
    mat_to_quat,
    nearest_index,
    quat_to_mat,
    select_interp,
    select_nearest,
    slerp_quat,
)


class TestNearest(unittest.TestCase):
    def test_nearest_index_picks_closest(self):
        times = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(nearest_index(times, 1.4), 1)
        self.assertEqual(nearest_index(times, 1.6), 2)
        self.assertEqual(nearest_index(times, -5.0), 0)
        self.assertEqual(nearest_index(times, 99.0), 3)

    def test_nearest_index_tie_prefers_earlier(self):
        self.assertEqual(nearest_index([0.0, 2.0], 1.0), 0)

    def test_nearest_index_empty_raises(self):
        with self.assertRaises(ValueError):
            nearest_index([], 0.0)

    def test_select_nearest_reports_signed_drift(self):
        samples = [(0.0, "a"), (1.0, "b"), (2.0, "c")]
        val, drift = select_nearest(samples, 1.2)
        self.assertEqual(val, "b")
        self.assertAlmostEqual(drift, 0.2)  # ref ahead of the sample → positive
        val, drift = select_nearest(samples, 1.9)
        self.assertEqual(val, "c")
        self.assertAlmostEqual(drift, -0.1)  # ref behind the sample → negative

    def test_select_nearest_empty(self):
        self.assertIsNone(select_nearest([], 0.0))


class TestInterp(unittest.TestCase):
    def test_lerp_vector(self):
        out = lerp([0.0, 10.0], [10.0, 20.0], 0.25)
        np.testing.assert_allclose(out, [2.5, 12.5])

    def test_select_interp_between_samples(self):
        samples = [(0.0, np.array([0.0])), (1.0, np.array([10.0]))]
        val, drift = select_interp(samples, 0.3, lerp)
        np.testing.assert_allclose(val, [3.0])
        # nearest real sample is t=0.0 → drift 0.3
        self.assertAlmostEqual(drift, 0.3)

    def test_select_interp_holds_at_edges_no_extrapolation(self):
        samples = [(1.0, np.array([5.0])), (2.0, np.array([7.0]))]
        val, drift = select_interp(samples, 5.0, lerp)  # past the end
        np.testing.assert_allclose(val, [7.0])  # held, not extrapolated
        self.assertAlmostEqual(drift, 3.0)
        val, drift = select_interp(samples, 0.0, lerp)  # before the start
        np.testing.assert_allclose(val, [5.0])
        self.assertAlmostEqual(drift, -1.0)

    def test_select_interp_empty(self):
        self.assertIsNone(select_interp([], 0.0, lerp))


class TestQuaternion(unittest.TestCase):
    def _rot_z(self, deg):
        r = np.deg2rad(deg)
        c, s = np.cos(r), np.sin(r)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def test_mat_quat_roundtrip(self):
        for deg in (0.0, 30.0, 90.0, 179.0, -120.0):
            R = self._rot_z(deg)
            np.testing.assert_allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-9)

    def test_slerp_midpoint_is_half_angle(self):
        q0 = mat_to_quat(self._rot_z(0.0))
        q1 = mat_to_quat(self._rot_z(90.0))
        mid = quat_to_mat(slerp_quat(q0, q1, 0.5))
        np.testing.assert_allclose(mid, self._rot_z(45.0), atol=1e-9)

    def test_slerp_takes_shortest_arc(self):
        # 0° and 350° should blend through 355°, not the long way through 175°.
        q0 = mat_to_quat(self._rot_z(0.0))
        q1 = mat_to_quat(self._rot_z(350.0))
        mid = quat_to_mat(slerp_quat(q0, q1, 0.5))
        np.testing.assert_allclose(mid, self._rot_z(355.0), atol=1e-7)

    def test_interp_pose_blends_translation_and_rotation(self):
        a = np.eye(4)
        b = np.eye(4)
        b[:3, :3] = self._rot_z(90.0)
        b[:3, 3] = [2.0, 4.0, 6.0]
        mid = interp_pose(a, b, 0.5)
        np.testing.assert_allclose(mid[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(mid[:3, :3], self._rot_z(45.0), atol=1e-9)


class TestTimestampedHistory(unittest.TestCase):
    def test_append_and_snapshot_ascending(self):
        h = TimestampedHistory(max_age_s=10.0, max_len=8)
        for i in range(4):
            h.append(float(i), np.array([float(i)]))
        snap = h.snapshot()
        self.assertEqual([t for t, _ in snap], [0.0, 1.0, 2.0, 3.0])

    def test_age_trim_keeps_recent(self):
        h = TimestampedHistory(max_age_s=1.0, max_len=100)
        h.append(0.0, "old")
        h.append(0.5, "mid")
        h.append(2.0, "new")  # cutoff = 2.0 - 1.0 = 1.0 → drops 0.0 and 0.5
        snap = h.snapshot()
        self.assertEqual([v for _, v in snap], ["new"])

    def test_len_cap(self):
        h = TimestampedHistory(max_age_s=1e9, max_len=3)
        for i in range(6):
            h.append(float(i), i)
        self.assertEqual([v for _, v in h.snapshot()], [3, 4, 5])

    def test_latest_and_empty(self):
        h = TimestampedHistory()
        self.assertIsNone(h.latest())
        h.append(1.0, "x")
        self.assertEqual(h.latest(), (1.0, "x"))

    def test_history_feeds_selectors(self):
        h = TimestampedHistory(max_age_s=10.0)
        h.append(0.0, np.array([0.0]))
        h.append(1.0, np.array([10.0]))
        val, _ = select_interp(h.snapshot(), 0.5, lerp)
        np.testing.assert_allclose(val, [5.0])


if __name__ == "__main__":
    unittest.main()
