"""The rules behind choosing a collection directory (``actoris_harena.web.roots``).

All pure or filesystem-local: no mount is ever performed here, and no SSH
connection is attempted -- what is checked is the command that WOULD be run, the
message a failure turns into, and how the console remembers where it has been.
"""

import json
import tempfile
import unittest
from pathlib import Path

from actoris_harena.web.roots import (
    MAX_RECENT,
    SSHFS_OPTIONS,
    format_target,
    load_state,
    looks_like_collection,
    mount_dir,
    mount_message,
    parse_mounts,
    parse_ssh_target,
    remember,
    root_problem,
    save_state,
    sshfs_argv,
)


class TestTarget(unittest.TestCase):
    def test_a_full_target_is_split(self):
        self.assertEqual(
            parse_ssh_target("halid@thanos:/mnt/seagate/so101"),
            ("halid", "thanos", "/mnt/seagate/so101"),
        )

    def test_the_user_may_be_left_to_ssh_config(self):
        self.assertEqual(parse_ssh_target("thanos:/data"), (None, "thanos", "/data"))

    def test_a_relative_remote_path_is_kept(self):
        # sshfs reads it from the remote home directory, like scp does.
        self.assertEqual(parse_ssh_target("thanos:so101"), (None, "thanos", "so101"))

    def test_a_missing_colon_says_what_the_form_is(self):
        with self.assertRaises(ValueError) as caught:
            parse_ssh_target("halid@thanos")
        self.assertIn("user@host:/path", str(caught.exception))

    def test_a_missing_path_is_refused(self):
        with self.assertRaises(ValueError):
            parse_ssh_target("thanos:")

    def test_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            parse_ssh_target("   ")

    def test_a_target_with_spaces_is_refused(self):
        # It would become extra argv words, which is how a shell gets surprised.
        with self.assertRaises(ValueError):
            parse_ssh_target("thanos:/mnt/my drive")

    def test_a_hostile_host_name_is_refused(self):
        for bad in ("thanos;rm -rf /", "-oProxyCommand=x", "tha nos"):
            with self.assertRaises(ValueError, msg=bad):
                parse_ssh_target(f"{bad}:/data")

    def test_the_canonical_form_round_trips(self):
        self.assertEqual(
            format_target("halid", "thanos", "/data"), "halid@thanos:/data"
        )
        self.assertEqual(format_target(None, "thanos", "/data"), "thanos:/data")


class TestCommand(unittest.TestCase):
    def test_the_mount_point_is_stable_for_one_remote_directory(self):
        base = Path("/tmp/mounts")
        first = mount_dir(base, "thanos", "/mnt/seagate/so101")
        again = mount_dir(base, "thanos", "/mnt/seagate/so101")
        self.assertEqual(first, again)
        self.assertEqual(first.name, "thanos_mnt_seagate_so101")

    def test_two_remote_directories_do_not_share_a_mount_point(self):
        base = Path("/tmp/mounts")
        self.assertNotEqual(
            mount_dir(base, "thanos", "/data/a"), mount_dir(base, "thanos", "/data/b")
        )

    def test_the_command_never_prompts_and_survives_a_dropped_link(self):
        argv = sshfs_argv("halid@thanos:/data", Path("/tmp/m"))
        self.assertEqual(argv[:3], ["sshfs", "halid@thanos:/data", "/tmp/m"])
        options = argv[argv.index("-o") + 1]
        self.assertIn("BatchMode=yes", options)
        self.assertIn("reconnect", options)
        self.assertEqual(options, ",".join(SSHFS_OPTIONS))

    def test_a_port_and_a_key_are_passed_through(self):
        argv = sshfs_argv("thanos:/data", Path("/tmp/m"), port=2222, identity="~/k")
        self.assertIn("-p", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "2222")
        self.assertTrue(any(a.startswith("IdentityFile=") for a in argv))
        self.assertNotIn("~", argv[-1])


class TestFailureMessages(unittest.TestCase):
    def test_success_has_no_message(self):
        self.assertIsNone(mount_message(0, ""))

    def test_a_refused_key_sends_the_operator_to_ssh_copy_id(self):
        message = mount_message(1, "Permission denied (publickey,password).")
        self.assertIn("ssh-copy-id", message)

    def test_an_unknown_host_key_cannot_be_accepted_from_a_browser(self):
        message = mount_message(1, "Host key verification failed.")
        self.assertIn("from a terminal", message)

    def test_an_unknown_failure_is_still_reported_verbatim(self):
        self.assertIn("read-only", mount_message(1, "remote is read-only"))

    def test_a_silent_failure_still_says_something(self):
        self.assertIn("exit 3", mount_message(3, ""))


class TestMountTable(unittest.TestCase):
    TABLE = (
        "proc /proc proc rw,relatime 0 0\n"
        "halid@thanos:/mnt/so101 /home/h/mounts/thanos_mnt_so101 fuse.sshfs "
        "rw,nosuid,nodev,relatime,user_id=1000 0 0\n"
        "thanos:/d /home/h/my\\040mount fuse.sshfs rw 0 0\n"
        "/dev/sda1 /mnt/seagate ext4 rw 0 0\n"
    )

    def test_only_the_sshfs_lines_are_mounts_of_ours(self):
        mounts = parse_mounts(self.TABLE)
        self.assertEqual(len(mounts), 2)
        self.assertEqual(mounts[0]["target"], "halid@thanos:/mnt/so101")
        self.assertEqual(mounts[0]["path"], "/home/h/mounts/thanos_mnt_so101")

    def test_an_escaped_space_in_a_path_is_read_back(self):
        self.assertEqual(parse_mounts(self.TABLE)[1]["path"], "/home/h/my mount")

    def test_an_empty_table_is_no_mounts(self):
        self.assertEqual(parse_mounts(""), [])


class TestTheDirectoryItself(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_real_directory_has_no_problem(self):
        self.assertIsNone(root_problem(self.root))

    def test_nothing_chosen_asks_for_one(self):
        self.assertIn("give a directory", root_problem(None))

    def test_a_missing_directory_says_so(self):
        self.assertIn("does not exist", root_problem(self.root / "nope"))

    def test_a_file_is_not_a_collection_directory(self):
        target = self.root / "notes.txt"
        target.write_text("hello")
        self.assertIn("not a directory", root_problem(target))

    def test_a_relative_path_is_refused(self):
        self.assertIn("absolute", root_problem(Path("outputs")))

    def test_a_directory_holding_datasets_is_recognised(self):
        (self.root / "cube-pnp" / "meta").mkdir(parents=True)
        (self.root / "cube-pnp" / "meta" / "info.json").write_text("{}")
        self.assertTrue(looks_like_collection(self.root))

    def test_an_ordinary_directory_is_not(self):
        (self.root / "photos").mkdir()
        self.assertFalse(looks_like_collection(self.root))


class TestRemembering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = Path(self.tmp.name) / "state" / "rig_web_roots.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_nothing_remembered_yet(self):
        self.assertEqual(load_state(self.file), {"last": None, "recent": []})

    def test_a_damaged_file_is_simply_forgotten(self):
        self.file.parent.mkdir(parents=True)
        self.file.write_text("{not json")
        self.assertEqual(load_state(self.file)["recent"], [])

    def test_a_directory_round_trips(self):
        save_state(self.file, remember(load_state(self.file), Path("/mnt/a"), "local"))
        state = load_state(self.file)
        self.assertEqual(state["last"], "/mnt/a")
        self.assertEqual(state["recent"][0]["kind"], "local")

    def test_the_newest_directory_comes_first_and_is_not_duplicated(self):
        state = {"last": None, "recent": []}
        state = remember(state, Path("/mnt/a"), "local")
        state = remember(state, Path("/mnt/b"), "ssh", "thanos:/mnt/b")
        state = remember(state, Path("/mnt/a"), "local")
        self.assertEqual([r["path"] for r in state["recent"]], ["/mnt/a", "/mnt/b"])

    def test_the_list_does_not_grow_forever(self):
        state = {"last": None, "recent": []}
        for i in range(MAX_RECENT + 4):
            state = remember(state, Path(f"/mnt/{i}"), "local")
        self.assertEqual(len(state["recent"]), MAX_RECENT)
        self.assertEqual(state["recent"][0]["path"], f"/mnt/{MAX_RECENT + 3}")

    def test_an_unwritable_place_is_not_fatal(self):
        save_state(Path("/proc/nowhere/state.json"), {"last": None, "recent": []})

    def test_what_is_written_is_readable_json(self):
        save_state(self.file, remember({"recent": []}, Path("/mnt/a"), "local"))
        with open(self.file) as f:
            self.assertEqual(json.load(f)["last"], "/mnt/a")


if __name__ == "__main__":
    unittest.main()
