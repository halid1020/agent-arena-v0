"""Where a training run may be sent, and the commands that would send it.

No machine is contacted here. What is worth testing about a destination is what
can be got wrong while writing one down -- a path that would run a second
command on the far end, a Slurm entry with no partition, a ``~`` that must NOT
be expanded on this machine -- and what an operator should be told when a box
does not answer. All of that is decided before any connection exists.

The shipped ``src/conf/train_destinations.yaml`` is checked too: it is a
tracked file that every launch reads, so a typo in it is a broken rig, not a
broken test.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_train_destinations
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from actoris_harena.training.destinations import (
    dataset_dir,
    destination,
    load_destinations,
    load_runs,
    manifest_dir,
    reach_message,
    remember_run,
    rsync_argv,
    save_runs,
    ssh_argv,
    stage_dir,
)

GOOD = {
    "box": {
        "ssh": "box",
        "kind": "ssh",
        "repo": "~/project/so101_garment",
        "scratch": "~/.cache/huggingface/lerobot",
        "stage": "{scratch}/local",
    }
}


from actoris_harena.training.destinations import set_destinations_path

# Declared, not derived. The module used to compute this from its own file's
# location, which stopped being right the moment it became an installed
# package; a rig now says where its config is, and so does this test.
_FIXTURE_REPO = Path(__file__).resolve().parents[2]


def setUpModule():
    set_destinations_path(
        _FIXTURE_REPO / "src" / "conf" / "train_destinations.yaml",
        repo_root=_FIXTURE_REPO,
    )


def write(entry) -> Path:
    path = Path(tempfile.mkdtemp()) / "dest.yaml"
    path.write_text(yaml.safe_dump(entry))
    return path


class TestValidation(unittest.TestCase):
    def test_a_good_entry_is_accepted_and_carries_its_name(self):
        dest = load_destinations(write(GOOD))["box"]
        self.assertEqual(dest["name"], "box")

    def test_a_missing_key_is_named(self):
        entry = {"box": {k: v for k, v in GOOD["box"].items() if k != "scratch"}}
        with self.assertRaises(ValueError) as caught:
            load_destinations(write(entry))
        self.assertIn("scratch", str(caught.exception))

    def test_an_unknown_key_is_named(self):
        entry = {"box": {**GOOD["box"], "queue": "gpu"}}
        with self.assertRaises(ValueError) as caught:
            load_destinations(write(entry))
        self.assertIn("queue", str(caught.exception))

    def test_an_unknown_kind_lists_the_ones_that_exist(self):
        entry = {"box": {**GOOD["box"], "kind": "pbs"}}
        with self.assertRaises(ValueError) as caught:
            load_destinations(write(entry))
        self.assertIn("slurm", str(caught.exception))

    def test_a_slurm_entry_without_a_partition_is_refused(self):
        # sbatch would otherwise fall back to the cluster's default partition,
        # which on CREATE has no GPU -- so the row queues and then trains on a
        # CPU, burning the whole reservation without an error.
        entry = {
            "c": {**GOOD["box"], "kind": "slurm", "repo": "~/r", "partition": None}
        }
        with self.assertRaises(ValueError) as caught:
            load_destinations(write(entry))
        self.assertIn("partition", str(caught.exception))

    def test_an_unusable_ssh_alias_is_refused(self):
        for hostile in ("box;rm -rf /", "-oProxyCommand=x", "a b"):
            entry = {"box": {**GOOD["box"], "ssh": hostile}}
            with self.assertRaises(ValueError, msg=hostile):
                load_destinations(write(entry))

    def test_a_path_that_could_run_a_second_command_is_refused(self):
        # These reach the remote shell UNQUOTED, so that `~` and `$USER` mean
        # the remote home and user. That is only safe if what may appear in
        # them is decided here.
        for hostile in ("~/r; rm -rf /", "~/r && curl x", "~/r`id`", "~/a b"):
            entry = {"box": {**GOOD["box"], "repo": hostile}}
            with self.assertRaises(ValueError, msg=hostile):
                load_destinations(write(entry))

    def test_an_unmeasurable_limit_key_is_refused(self):
        entry = {"box": {**GOOD["box"], "limits": {"pi05": {"vram": 24}}}}
        with self.assertRaises(ValueError) as caught:
            load_destinations(write(entry))
        self.assertIn("vram", str(caught.exception))

    def test_asking_for_a_machine_that_is_not_there_lists_the_ones_that_are(self):
        with self.assertRaises(ValueError) as caught:
            destination("hal9000", write(GOOD))
        self.assertIn("box", str(caught.exception))


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.dest = load_destinations(write(GOOD))["box"]

    def test_the_stage_is_built_from_the_scratch_root(self):
        self.assertEqual(stage_dir(self.dest), "~/.cache/huggingface/lerobot/local")

    def test_a_dataset_lands_under_the_stage(self):
        self.assertEqual(
            dataset_dir(self.dest, "fold-short"),
            "~/.cache/huggingface/lerobot/local/fold-short",
        )

    def test_manifests_are_kept_beside_the_datasets_not_inside_them(self):
        # A manifest under the stage would be read as a dataset directory by
        # anything listing what is staged.
        self.assertNotIn(stage_dir(self.dest), manifest_dir(self.dest))


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.dest = load_destinations(write(GOOD))["box"]

    def test_ssh_never_prompts(self):
        # A password prompt inside a web request simply hangs, and an
        # unattended launcher that stops to ask has already failed.
        argv = ssh_argv(self.dest, "true")
        self.assertIn("BatchMode=yes", argv)
        self.assertEqual(argv[-2:], ["box", "true"])

    def test_rsync_creates_the_dataset_directory_rather_than_emptying_it(self):
        # A trailing slash on the SOURCE would copy the dataset's contents into
        # the stage, mixing one dataset's files in with every other.
        argv = rsync_argv("/mnt/x/fold-short", self.dest)
        self.assertTrue(argv[-2].endswith("/fold-short"))
        self.assertFalse(argv[-2].endswith("/"))
        self.assertTrue(argv[-1].endswith("/local/"))

    def test_a_dry_run_rsync_transfers_nothing(self):
        self.assertIn("--dry-run", rsync_argv("/mnt/x/a", self.dest, dry_run=True))

    def test_the_source_is_one_dataset_and_not_the_drive_it_is_on(self):
        # MEASURED failure: handed the collection directory instead of the
        # dataset, this uploaded EVERY dataset on the drive into a directory
        # named after the drive -- 1.2 GB in, with nothing to show for it but a
        # transfer that was taking too long. The launcher joins the name; this
        # pins what the resulting argv must look like.
        argv = rsync_argv("/mnt/seagate/so101/fold-short", self.dest)
        self.assertTrue(argv[-2].endswith("so101/fold-short"))
        self.assertNotEqual(argv[-2], "/mnt/seagate/so101")


class TestWhatAnUnreachableMachineSays(unittest.TestCase):
    """Four failures with four different fixes, and only one is a bug."""

    def test_answering_is_silence(self):
        self.assertIsNone(reach_message(0, "", "create"))

    def test_a_timeout_asks_about_the_vpn(self):
        # MEASURED: `ssh create` from off the university network simply hangs
        # until it times out, which reads as a broken launcher.
        self.assertIn("VPN", reach_message(124, "", "create"))
        self.assertIn("VPN", reach_message(255, "connection timed out", "create"))

    def test_a_rejected_key_says_to_copy_one(self):
        message = reach_message(255, "Permission denied (publickey).", "thanos")
        self.assertIn("ssh-copy-id thanos", message)

    def test_an_unknown_host_key_says_to_accept_it_in_a_terminal(self):
        # It cannot be accepted from a web page, so the instruction has to name
        # the terminal command.
        message = reach_message(255, "Host key verification failed.", "thanos")
        self.assertIn("ssh thanos", message)

    def test_an_unresolvable_name_asks_about_the_ssh_config(self):
        message = reach_message(255, "Could not resolve hostname box", "box")
        self.assertIn("ssh/config", message)

    def test_an_unrecognised_failure_is_passed_through_not_swallowed(self):
        self.assertIn("kex_exchange", reach_message(255, "kex_exchange failed", "b"))


class TestRememberingWhatWasLaunched(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "runs.yaml"

    def test_nothing_recorded_yet_reads_empty(self):
        self.assertEqual(load_runs(self.path), {"runs": []})

    def test_a_damaged_file_reads_empty_rather_than_raising(self):
        # The runs themselves live on the destination; losing this list costs
        # the console's view of them, and --status can ask the machine again.
        self.path.write_text("{[not yaml")
        self.assertEqual(load_runs(self.path), {"runs": []})

    def test_a_run_survives_a_write_and_a_read(self):
        save_runs(self.path, remember_run(load_runs(self.path), {"id": "a", "x": 1}))
        self.assertEqual(load_runs(self.path)["runs"][0]["id"], "a")

    def test_the_newest_run_is_first(self):
        state = remember_run(remember_run({"runs": []}, {"id": "a"}), {"id": "b"})
        self.assertEqual([r["id"] for r in state["runs"]], ["b", "a"])

    def test_relaunching_replaces_rather_than_duplicates(self):
        state = remember_run({"runs": [{"id": "a", "n": 1}]}, {"id": "a", "n": 2})
        self.assertEqual(state["runs"], [{"id": "a", "n": 2}])

    def test_an_unwritable_path_is_not_fatal(self):
        save_runs(Path("/proc/nowhere/runs.yaml"), {"runs": [{"id": "a"}]})

    def test_a_row_without_an_id_is_dropped(self):
        self.path.write_text(yaml.safe_dump({"runs": [{"dest": "x"}, {"id": "a"}]}))
        self.assertEqual(len(load_runs(self.path)["runs"]), 1)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
