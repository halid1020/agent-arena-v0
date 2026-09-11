"""The run log: what a rollout leaves behind for afterwards.

A failed grasp cannot be paused mid-air, so the log is the only chance to see it
again. Four properties matter and are checked here: the plans reach disk as they
land (a run that dies still leaves them), one plan is written once however many
ticks execute it, a verdict is kept against the attempt it judges, and the
frames a plan was drawn from are kept only when they were asked for.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_policy_log
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from actoris_harena.deploy.policy_log import (
    OUTCOMES,
    RunLog,
    read_jsonl,
    score,
    trial_verdicts,
)


class StubSource:
    round_trip_s = 0.55
    server_infer_s = 0.52

    def __init__(self, images=None):
        self.last_chunk = np.zeros((4, 12))
        self.last_chunk_seq = 1
        self.last_chunk_at = 1000.0
        self._images = {} if images is None else images

    def last_sent(self):
        return (np.arange(12.0), self._images)


class TestRunLog(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.log = RunLog.create(
            task="pick up the cube", source="remote 'act'", hz=30.0, root=self.tmp
        )
        self.addCleanup(self.log.close)

    def chunks(self):
        text = (self.log.root / "chunks.jsonl").read_text().splitlines()
        return [json.loads(line) for line in text if line.strip()]

    def test_the_run_is_named_and_described(self):
        meta = json.loads((self.log.root / "meta.json").read_text())

        self.assertEqual(meta["task"], "pick up the cube")
        self.assertEqual(meta["hz"], 30.0)

    def test_a_tick_keeps_both_what_was_measured_and_what_was_sent(self):
        import pandas as pd

        self.log.tick(0.0, np.arange(12.0), np.arange(12.0) + 1, "run", 30)
        self.log.tick(0.033, np.arange(12.0), None, "hold", 0)
        root = self.log.close()

        frame = pd.read_parquet(root / "ticks.parquet")
        self.assertEqual(list(frame["mode"]), ["run", "hold"])
        self.assertEqual(list(frame["served"]), [True, False])
        self.assertEqual(len(frame["state"][0]), 12)
        self.assertIsNone(frame["commanded"][1])

    def test_every_tick_says_which_attempt_it_belongs_to(self):
        import pandas as pd

        self.log.tick(0.0, np.arange(12.0), np.arange(12.0), "run", 30)
        self.assertEqual(self.log.new_trial(), 1)
        self.log.tick(1.0, np.arange(12.0), np.arange(12.0), "run", 30)
        root = self.log.close()

        # Without this column two attempts at one scene read as a single long
        # series with an unexplained pause in the middle of it.
        frame = pd.read_parquet(root / "ticks.parquet")
        self.assertEqual(list(frame["trial"]), [0, 1])

    def test_a_plan_is_written_once_however_often_it_is_offered(self):
        source = StubSource()

        self.assertTrue(self.log.note_chunk(source))
        self.assertFalse(self.log.note_chunk(source))
        source.last_chunk_seq = 2
        self.assertTrue(self.log.note_chunk(source))

        self.assertEqual([c["seq"] for c in self.chunks()], [1, 2])

    def test_a_plan_reaches_disk_before_the_run_ends(self):
        # A rollout that dies mid-episode is exactly when the log matters.
        self.log.note_chunk(StubSource())

        written = self.chunks()
        self.assertEqual(len(written[0]["actions"]), 4)
        self.assertEqual(len(written[0]["state"]), 12)
        self.assertEqual(written[0]["round_trip_s"], 0.55)

    def test_a_source_with_no_plan_yet_writes_nothing(self):
        class Empty:
            last_chunk = None
            last_chunk_seq = 0
            round_trip_s = 0.0

        self.assertFalse(self.log.note_chunk(Empty()))
        self.assertEqual(self.chunks(), [])

    def test_closing_twice_is_harmless(self):
        self.log.tick(0.0, np.zeros(12), None, "run", 0)
        self.assertEqual(self.log.close(), self.log.close())


class TestJudgingAnAttempt(unittest.TestCase):
    """A run is a sequence of attempts, and the verdicts are the comparison."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.log = RunLog.create(
            task="fold the short", source="remote 'act'", hz=30.0, root=self.tmp
        )
        self.addCleanup(self.log.close)

    def test_a_verdict_names_the_attempt_it_judges(self):
        self.log.new_trial()
        record = self.log.trial_outcome("success", "clean fold")
        self.assertEqual(record["trial"], 1)
        self.assertEqual(record["outcome"], "success")
        self.assertEqual(record["notes"], "clean fold")

    def test_a_verdict_reaches_disk_before_the_run_ends(self):
        # Same discipline as a chunk: a run killed afterwards keeps it.
        self.log.trial_outcome("failure", "missed the hem")
        written = read_jsonl(self.log.root / "trials.jsonl")
        self.assertEqual([r["outcome"] for r in written], ["failure"])

    def test_only_the_three_outcomes_are_accepted(self):
        # Nothing else can be scored later, so nothing else may be written.
        for bad in ("", "SUCCESS", "maybe", "ok"):
            with self.assertRaises(ValueError):
                self.log.trial_outcome(bad)
        for good in OUTCOMES:
            self.log.trial_outcome(good)

    def test_looking_again_supersedes_the_earlier_verdict(self):
        self.log.trial_outcome("failure", "looked like a miss")
        self.log.trial_outcome("success", "it had folded after all")
        # Both lines survive -- that somebody changed their mind is itself a
        # fact about the run -- but only the later one counts.
        self.assertEqual(len(read_jsonl(self.log.root / "trials.jsonl")), 2)
        self.assertEqual(trial_verdicts(self.log.root)[0]["outcome"], "success")

    def test_a_discarded_attempt_is_not_a_failure(self):
        # The scene was wrong or the tunnel dropped: it leaves the denominator
        # rather than counting against the policy.
        self.log.trial_outcome("success")
        self.log.new_trial()
        self.log.trial_outcome("discard", "bumped the table")
        self.log.new_trial()
        self.log.trial_outcome("failure")
        self.assertEqual(
            score(self.log.root),
            {"trials": 2, "successes": 1, "discarded": 1, "rate": 0.5},
        )

    def test_a_run_with_nothing_to_count_has_no_rate(self):
        self.log.trial_outcome("discard")
        self.assertIsNone(score(self.log.root)["rate"])

    def test_a_run_nobody_judged_scores_nothing_rather_than_zero(self):
        self.assertEqual(score(self.log.root)["trials"], 0)
        self.assertIsNone(score(self.log.root)["rate"])

    def test_a_line_truncated_by_a_kill_is_dropped_not_raised(self):
        # A rollout ends by being killed more often than not, mid-append.
        self.log.trial_outcome("success", "first")
        path = self.log.root / "trials.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"trial": 1, "outcome": "fail')
        self.assertEqual([r["notes"] for r in read_jsonl(path)], ["first"])

    def test_reading_a_run_that_kept_no_verdicts_is_empty_not_an_error(self):
        self.assertEqual(read_jsonl(self.tmp / "nothing" / "trials.jsonl"), [])


class TestKeepingTheFramesAPlanWasDrawnFrom(unittest.TestCase):
    """The pixels are the one thing an attribution study cannot do without."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.images = {
            "central": np.zeros((8, 12, 3), dtype=np.uint8),
            "left_arm_left_gripper": np.full((8, 12, 3), 200, dtype=np.uint8),
        }

    def make(self, save_frames):
        log = RunLog.create(
            task="fold the short",
            source="remote 'act'",
            hz=30.0,
            root=self.tmp / str(save_frames),
            save_frames=save_frames,
        )
        self.addCleanup(log.close)
        return log

    def test_frames_are_not_kept_unless_they_were_asked_for(self):
        log = self.make(False)
        log.note_chunk(StubSource(self.images))
        self.assertFalse((log.root / "frames").exists())
        self.assertEqual(
            json.loads((log.root / "meta.json").read_text())["frames"], False
        )

    def test_asking_for_them_keeps_one_per_camera_under_the_plan(self):
        log = self.make(True)
        log.note_chunk(StubSource(self.images))
        kept = sorted(p.name for p in (log.root / "frames" / "000001").iterdir())
        self.assertEqual(kept, ["central.jpg", "left_arm_left_gripper.jpg"])

    def test_the_chunk_line_says_how_many_frames_went_with_it(self):
        log = self.make(True)
        log.note_chunk(StubSource(self.images))
        line = read_jsonl(log.root / "chunks.jsonl")[0]
        self.assertEqual(line["frames"], 2)
        self.assertEqual(line["trial"], 0)

    def test_a_frame_in_a_shape_that_cannot_be_encoded_is_skipped(self):
        # Depth, or a mono camera: neither is three-channel, and neither should
        # stop a rollout.
        log = self.make(True)
        log.note_chunk(StubSource({"depth": np.zeros((8, 12), dtype=np.uint16)}))
        self.assertEqual(read_jsonl(log.root / "chunks.jsonl")[0]["frames"], 0)

    def test_a_source_that_kept_no_images_still_writes_its_plan(self):
        log = self.make(True)
        log.note_chunk(StubSource({}))
        self.assertEqual(len(read_jsonl(log.root / "chunks.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
