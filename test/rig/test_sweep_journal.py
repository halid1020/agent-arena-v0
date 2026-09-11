"""Unit tests for the sweep journal: a long grid must survive being killed."""

import json
import tempfile
import unittest
from pathlib import Path

from actoris_harena.deploy.sweep_journal import (
    append_row,
    done_keys,
    journal_for,
    load_rows,
    row_key,
)


class TestJournalPath(unittest.TestCase):
    def test_it_sits_beside_the_report_it_belongs_to(self):
        self.assertEqual(
            journal_for("outputs/chunking/stage_a.md"),
            Path("outputs/chunking/stage_a.md.episodes.jsonl"),
        )

    def test_two_reports_do_not_share_one_journal(self):
        self.assertNotEqual(journal_for("a.md"), journal_for("b.md"))


class TestRowKey(unittest.TestCase):
    def test_the_trial_index_distinguishes_repeats_of_one_seed(self):
        # The overfit-one-scenario protocol runs the same seed every trial, so
        # a seed alone cannot identify an episode.
        self.assertNotEqual(row_key("sync", 0), row_key("sync", 1))

    def test_the_cell_distinguishes_the_same_trial_under_two_splices(self):
        self.assertNotEqual(row_key("sync", 3), row_key("append", 3))


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "sweep.md.episodes.jsonl"

    def test_nothing_written_yet_reads_as_nothing_done(self):
        self.assertEqual(load_rows(self.path), [])
        self.assertEqual(done_keys(load_rows(self.path)), set())

    def test_a_row_survives_the_trip(self):
        append_row(self.path, {"cell": "sync", "trial": 0, "success": True})
        (row,) = load_rows(self.path)
        self.assertEqual(row["cell"], "sync")
        self.assertIs(row["success"], True)

    def test_rows_keep_the_order_they_were_scored_in(self):
        for i in range(3):
            append_row(self.path, {"cell": "blend@w5/exp", "trial": i})
        self.assertEqual([r["trial"] for r in load_rows(self.path)], [0, 1, 2])

    def test_the_parent_directory_is_created(self):
        deep = Path(self.dir.name) / "a" / "b" / "sweep.md.episodes.jsonl"
        append_row(deep, {"cell": "sync", "trial": 0})
        self.assertTrue(deep.is_file())

    def test_a_half_written_last_line_is_dropped_not_raised(self):
        # The crash the journal exists to survive can happen mid-append.
        append_row(self.path, {"cell": "sync", "trial": 0})
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write('{"cell": "sync", "tri')
        rows = load_rows(self.path)
        self.assertEqual(len(rows), 1)

    def test_the_file_is_flushed_per_row_so_a_kill_keeps_it(self):
        append_row(self.path, {"cell": "sync", "trial": 0})
        # Read through a separate handle: nothing may be sitting in a buffer.
        text = self.path.read_text(encoding="utf-8")
        self.assertEqual(json.loads(text.splitlines()[0])["cell"], "sync")


class TestDoneKeys(unittest.TestCase):
    def test_finished_episodes_are_named_so_a_resume_can_skip_them(self):
        rows = [
            {"cell": "sync", "trial": 0},
            {"cell": "sync", "trial": 1},
            {"cell": "append", "trial": 0},
        ]
        self.assertEqual(
            done_keys(rows),
            {row_key("sync", 0), row_key("sync", 1), row_key("append", 0)},
        )

    def test_a_row_from_before_trials_were_numbered_is_ignored(self):
        # Better to re-run an episode than to skip the wrong one.
        self.assertEqual(done_keys([{"cell": "sync"}]), set())


if __name__ == "__main__":
    unittest.main()
