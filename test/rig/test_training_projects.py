"""Grouping discovered runs into named experiments.

The thing under test is not the storage -- it is a list of strings in a YAML
file -- but the two properties that make it safe to keep membership OUTSIDE the
runs themselves: a key that cannot be confused with anything else, and a
membership that survives the machine holding the run being unreachable. Both
have a way of going wrong quietly, which is why they are pinned here.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_training_projects
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from actoris_harena.web import projects
from actoris_harena.web.projects import ProjectError


def store(*names: str) -> dict:
    out: dict = {"projects": []}
    for name in names:
        projects.create(out, name)
    return out


class TestRunKeys(unittest.TestCase):
    def test_a_key_round_trips(self):
        key = projects.run_key("thanos", "towel__all", "act")
        self.assertEqual(key, "thanos|towel__all|act")
        self.assertEqual(
            projects.split_key(key),
            {"dest": "thanos", "run": "towel__all", "policy": "act"},
        )

    def test_a_separator_inside_a_part_is_refused(self):
        # None of the three can contain it -- runs.check_name allows letters,
        # digits and . _ + - only -- so a part that does did not come from here,
        # and escaping it would invent a key that resolves to nothing.
        with self.assertRaises(ProjectError):
            projects.run_key("thanos", "towel|all", "act")

    def test_a_missing_part_is_refused(self):
        with self.assertRaises(ProjectError):
            projects.run_key("thanos", "", "act")

    def test_something_that_is_not_a_key_is_refused(self):
        with self.assertRaises(ProjectError):
            projects.split_key("thanos|towel__all")

    def test_a_sim_cell_name_is_a_usable_policy_part(self):
        # A sim run's log is named for its whole cell, and every part of that
        # may hold an underscore. The key must carry it unchanged.
        key = projects.run_key("local", "20260904", "simple_handover_split_so101_act")
        self.assertEqual(
            projects.split_key(key)["policy"], "simple_handover_split_so101_act"
        )


class TestNames(unittest.TestCase):
    def test_a_plain_name_is_kept(self):
        self.assertEqual(
            projects.check_project_name("  tactile fold  "), "tactile fold"
        )

    def test_an_empty_name_is_refused(self):
        with self.assertRaises(ProjectError):
            projects.check_project_name("   ")

    def test_a_slash_is_refused(self):
        with self.assertRaises(ProjectError):
            projects.check_project_name("ports/act")

    def test_a_builtin_label_cannot_be_taken(self):
        # A real project called "All runs" would silently shadow the grouping
        # every machine already has.
        for name in ("All runs", "Unassigned", "__all__"):
            with self.assertRaises(ProjectError):
                projects.check_project_name(name)


class TestOperations(unittest.TestCase):
    def setUp(self):
        self.store = store("ports")
        self.act = projects.run_key("thanos", "towel__all", "act")
        self.port = projects.run_key("thanos", "towel__all", "so101_act")

    def test_a_duplicate_name_is_refused(self):
        with self.assertRaises(ProjectError):
            projects.create(self.store, "ports")

    def test_assigning_twice_adds_once(self):
        projects.assign(self.store, "ports", [self.act])
        projects.assign(self.store, "ports", [self.act, self.port])
        self.assertEqual(
            projects.find(self.store, "ports")["runs"], [self.act, self.port]
        )

    def test_assignment_keeps_the_order_it_was_given(self):
        projects.assign(self.store, "ports", [self.port, self.act])
        self.assertEqual(
            projects.find(self.store, "ports")["runs"], [self.port, self.act]
        )

    def test_unassigning_something_absent_is_not_an_error(self):
        projects.unassign(self.store, "ports", [self.act])
        self.assertEqual(projects.find(self.store, "ports")["runs"], [])

    def test_assigning_something_that_is_not_a_key_is_refused(self):
        with self.assertRaises(ProjectError):
            projects.assign(self.store, "ports", ["towel__all"])

    def test_a_rename_keeps_the_membership(self):
        projects.assign(self.store, "ports", [self.act])
        projects.rename(self.store, "ports", "port parity")
        self.assertIsNone(projects.find(self.store, "ports"))
        self.assertEqual(projects.find(self.store, "port parity")["runs"], [self.act])

    def test_a_run_may_be_in_two_projects(self):
        projects.create(self.store, "tactile")
        projects.assign(self.store, "ports", [self.act])
        projects.assign(self.store, "tactile", [self.act])
        self.assertEqual(
            projects.projects_of(self.store, self.act), ["ports", "tactile"]
        )

    def test_deleting_a_project_is_a_label_not_the_runs(self):
        projects.assign(self.store, "ports", [self.act])
        projects.delete(self.store, "ports")
        self.assertEqual(self.store["projects"], [])


class TestMembershipSurvivesAnUnreachableMachine(unittest.TestCase):
    """The one property that makes this safe to keep outside the runs.

    A machine off the VPN this morning has not deleted anything. A project that
    silently shrank would be the one way this view could misreport what was
    run, so a member nothing answered for is REPORTED as missing.
    """

    def setUp(self):
        self.store = store("ports")
        self.here = {"dest": "local", "run": "towel__all", "policy": "act"}
        self.away = {"dest": "create", "run": "towel__all", "policy": "pi05"}
        projects.assign(
            self.store,
            "ports",
            [projects.key_of(self.here), projects.key_of(self.away)],
        )

    def test_a_run_no_machine_answered_for_is_reported_not_dropped(self):
        missing = projects.missing_runs(self.store, [self.here], "ports")
        self.assertEqual(
            missing, [{"dest": "create", "run": "towel__all", "policy": "pi05"}]
        )

    def test_nothing_is_missing_when_every_machine_answered(self):
        self.assertEqual(
            projects.missing_runs(self.store, [self.here, self.away], "ports"), []
        )


class TestTheFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.path = self.tmp / "training_projects.yaml"

    def test_a_missing_file_is_an_empty_store(self):
        self.assertEqual(projects.load(self.path), {"projects": []})

    def test_a_store_round_trips(self):
        original = store("ports")
        key = projects.run_key("thanos", "towel__all", "act")
        projects.assign(original, "ports", [key])
        projects.save(self.path, original)
        self.assertEqual(projects.load(self.path)["projects"][0]["runs"], [key])

    def test_a_file_that_no_longer_parses_costs_the_projects_only(self):
        # Read on every page load. A hand-edit that broke it must not take the
        # whole Training tab with it.
        self.path.write_text("projects: [oh no\n", encoding="utf-8")
        self.assertEqual(projects.load(self.path), {"projects": []})

    def test_rubbish_entries_are_skipped_rather_than_raising(self):
        self.path.write_text(
            "projects:\n  - 7\n  - name: ok\n    runs: nonsense\n", encoding="utf-8"
        )
        loaded = projects.load(self.path)
        self.assertEqual([p["name"] for p in loaded["projects"]], ["ok"])
        self.assertEqual(loaded["projects"][0]["runs"], [])

    def test_the_write_is_atomic(self):
        projects.save(self.path, store("ports"))
        self.assertFalse(list(self.tmp.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
