"""Reading a training run's progress out of the only record that exists.

Every log line quoted here is real, copied from runs on the two machines this
rig trains on: ``cube-pnp-new__all`` on KCL CREATE (Slurm, so tqdm is disabled
and the lines are clean) and the live ACT run on thanos (tqdm enabled, so the
file is one ``\\r``-separated blob and each metric line carries the progress
frame that preceded it). Synthetic lines would not have caught either of the two
things this module exists to handle.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_training_progress
"""

import unittest

from actoris_harena.training import progress, runs

# ── Real log text ────────────────────────────────────────────────────────────

# The head of any lerobot training log: its own configuration, pprinted.
HEAD = """INFO 2026-08-24 11:50:53 ot_train.py:222 {'batch_size': 8,
 'checkpoint_path': None,
 'dataset': {'episodes': None,
             'repo_id': 'cube-pnp-new'},
 'log_freq': 100,
 'num_workers': 8,
 'save_freq': 10000,
 'steps': 80000,
 'wandb': {'enable': False}}"""

# CREATE, contiguous, from the start of the run. No progress bar anywhere:
# inside Slurm tqdm is disabled, so there is no exact step in this file at all.
CREATE_START = """INFO 2026-08-24 11:51:09 ot_train.py:596 step:100 smpl:800 ep:2 epch:0.02 loss:10.460 grdn:210.415 lr:1.0e-05 updt_s:0.135 data_s:0.013 smp/s:54 mem_gb:5.76
INFO 2026-08-24 11:51:20 ot_train.py:596 step:200 smpl:2K ep:5 epch:0.04 loss:4.004 grdn:104.632 lr:1.0e-05 updt_s:0.110 data_s:0.002 smp/s:71 mem_gb:5.77
INFO 2026-08-24 11:51:32 ot_train.py:596 step:300 smpl:2K ep:7 epch:0.06 loss:3.371 grdn:89.813 lr:1.0e-05 updt_s:0.110 data_s:0.002 smp/s:71 mem_gb:5.77"""

# CREATE, three consecutive lines from the middle of the SAME run: steps 9900,
# 10000 and 10100, all three printed as `10K`. This is the whole reason the
# x-axis cannot be the log's own step field.
CREATE_MIDDLE = """INFO 2026-08-24 12:09:47 ot_train.py:596 step:10K smpl:79K ep:235 epch:2.02 loss:0.256 grdn:16.191 lr:1.0e-05 updt_s:0.111 data_s:0.002 smp/s:71 mem_gb:5.76
INFO 2026-08-24 12:09:58 ot_train.py:596 step:10K smpl:80K ep:237 epch:2.04 loss:0.251 grdn:16.517 lr:1.0e-05 updt_s:0.111 data_s:0.002 smp/s:71 mem_gb:5.76
INFO 2026-08-24 12:10:11 ot_train.py:596 step:10K smpl:81K ep:239 epch:2.06 loss:0.248 grdn:16.167 lr:1.0e-05 updt_s:0.111 data_s:0.002 smp/s:71 mem_gb:5.76"""

CREATE_END = """INFO 2026-08-24 14:04:28 ot_train.py:641 Checkpoint policy after step 70000
INFO 2026-08-24 14:23:34 ot_train.py:641 Checkpoint policy after step 80000
INFO 2026-08-24 14:23:35 ot_train.py:721 End of training"""

# thanos, three consecutive metric lines as they appear AFTER \r translation:
# the progress frame and the INFO line are one line, and the frame carries the
# exact step the line rounds to 71K.
THANOS_BODY = """Training:  89%|.....| 71200/80000 [7:55:31<40:04,  3.66step/s]INFO 2026-08-28 19:21:06 ot_train.py:596 step:71K smpl:570K ep:1K epch:18.90 loss:0.078 grdn:6.516 lr:1.0e-05 updt_s:0.272 data_s:0.002 smp/s:29 mem_gb:11.27
Training:  89%|.....| 71300/80000 [7:55:58<39:40,  3.65step/s]INFO 2026-08-28 19:21:33 ot_train.py:596 step:71K smpl:570K ep:1K epch:18.92 loss:0.078 grdn:6.545 lr:1.0e-05 updt_s:0.272 data_s:0.002 smp/s:29 mem_gb:11.27
Training:  89%|.....| 71400/80000 [7:56:26<39:11,  3.66step/s]INFO 2026-08-28 19:22:00 ot_train.py:596 step:71K smpl:571K ep:1K epch:18.95 loss:0.080 grdn:7.167 lr:1.0e-05 updt_s:0.270 data_s:0.002 smp/s:29 mem_gb:11.27"""

# The raw tail of the same file, before translation: tqdm frames separated by
# carriage returns, no newline in sight.
THANOS_TAIL_RAW = (
    "\rTraining:  89%|...| 71377/80000 [7:56:19<39:18,  3.66step/s]"
    "\rTraining:  89%|...| 71378/80000 [7:56:20<39:16,  3.66step/s]"
    "\rTraining:  89%|...| 71379/80000 [7:56:20<39:15,  3.66step/s]"
)


class TestNumbers(unittest.TestCase):
    def test_a_magnitude_suffix_is_undone(self):
        self.assertEqual(progress.parse_big("80K"), 80000.0)
        self.assertEqual(progress.parse_big("2K"), 2000.0)
        self.assertEqual(progress.parse_big("640K"), 640000.0)
        self.assertEqual(progress.parse_big("800"), 800.0)

    def test_a_clock_is_seconds(self):
        self.assertEqual(progress.parse_clock("7:56:20"), 28580.0)
        self.assertEqual(progress.parse_clock("39:15"), 2355.0)
        self.assertIsNone(progress.parse_clock("?"))


class TestHead(unittest.TestCase):
    def test_the_run_configuration_is_read_from_the_log_itself(self):
        # This is what makes a CREATE log readable at all: without log_freq
        # there is no way to turn a line's ordinal into a step.
        head = progress.parse_head(HEAD)
        self.assertEqual(head["steps"], 80000)
        self.assertEqual(head["log_freq"], 100)
        self.assertEqual(head["save_freq"], 10000)
        self.assertEqual(head["batch_size"], 8)


class TestMetricLines(unittest.TestCase):
    def test_every_full_precision_field_is_kept(self):
        point = progress.parse_points(CREATE_START)[0]
        self.assertAlmostEqual(point["loss"], 10.460)
        self.assertAlmostEqual(point["grad_norm"], 210.415)
        self.assertAlmostEqual(point["lr"], 1.0e-05)
        self.assertAlmostEqual(point["update_s"], 0.135)
        self.assertAlmostEqual(point["dataloading_s"], 0.013)
        self.assertAlmostEqual(point["samples_per_s"], 54.0)
        self.assertAlmostEqual(point["gpu_mem_gb"], 5.76)
        self.assertAlmostEqual(point["epochs"], 0.02)

    def test_the_source_location_is_not_mistaken_for_a_metric(self):
        # The line contains `ot_train.py:596` and a wall-clock `11:51:09`, both
        # of which a general `word:number` pattern reads as measurements.
        point = progress.parse_points(CREATE_START)[0]
        self.assertNotIn("py", point)
        self.assertEqual(
            {
                "at",
                "time",
                "loss",
                "grad_norm",
                "lr",
                "update_s",
                "dataloading_s",
                "samples_per_s",
                "gpu_mem_gb",
                "epochs",
                "step_label",
                "step_divisor",
                "samples_label",
                "episodes_label",
            },
            set(point),
        )


class TestTheStepIsNotInTheStepField(unittest.TestCase):
    """The finding this module is built around."""

    def test_three_different_steps_print_as_one_label(self):
        labels = [p["step_label"] for p in progress.parse_points(CREATE_MIDDLE)]
        self.assertEqual(labels, [10000.0, 10000.0, 10000.0])

    def test_without_a_progress_bar_the_step_comes_from_the_ordinal(self):
        parsed = progress.parse_log(CREATE_START, head_text=HEAD)
        self.assertEqual([p["step"] for p in parsed["points"]], [100, 200, 300])
        self.assertEqual([p["source"] for p in parsed["points"]], ["count"] * 3)
        self.assertEqual(parsed["problems"], [])

    def test_with_a_progress_bar_the_step_is_exact(self):
        parsed = progress.parse_log(THANOS_BODY, head_text=HEAD)
        self.assertEqual([p["step"] for p in parsed["points"]], [71200, 71300, 71400])
        self.assertEqual([p["source"] for p in parsed["points"]], ["tqdm"] * 3)
        # And the labels it would have used instead are all the same number.
        self.assertEqual({p["step_label"] for p in parsed["points"]}, {71000.0})

    def test_a_resumed_run_is_offset_rather_than_counted_from_zero(self):
        # tqdm's total is what is LEFT after a resume, so `done` is relative.
        # Trusting it as absolute would put the whole curve 10 000 steps early.
        resumed = THANOS_BODY.replace("/80000 [", "/70000 [")
        parsed = progress.parse_log(resumed, head_text=HEAD)
        self.assertEqual([p["step"] for p in parsed["points"]], [81200, 81300, 81400])

    def test_a_disagreement_is_reported_rather_than_drawn(self):
        # Two disjoint pieces of a log stitched together: the ordinal count then
        # means nothing, and a curve drawn on it looks exactly like a curve.
        parsed = progress.parse_log(CREATE_START + "\n" + CREATE_MIDDLE, head_text=HEAD)
        self.assertTrue(parsed["problems"])
        self.assertIn("disagrees", parsed["problems"][0])

    def test_counting_past_the_end_of_the_run_is_reported(self):
        parsed = progress.parse_log("\n".join([CREATE_START] * 300), head_text=HEAD)
        self.assertTrue(any("past the run's" in p for p in parsed["problems"]))


class TestProgressBar(unittest.TestCase):
    def test_a_carriage_return_blob_is_read_as_frames(self):
        parsed = progress.parse_log(THANOS_TAIL_RAW)
        frame = parsed["last_frame"]
        self.assertEqual(frame["done"], 71379)
        self.assertEqual(frame["total"], 80000)
        self.assertEqual(frame["eta_s"], 2355.0)
        self.assertAlmostEqual(frame["rate"], 3.66)

    def test_seconds_per_step_is_inverted_into_a_rate(self):
        # tqdm flips its unit below one step a second, which the local laptop
        # run does throughout.
        frames = progress.parse_tqdm("1/5 [00:04<00:18,  4.00s/step]")
        self.assertAlmostEqual(frames[0]["rate"], 0.25)

    def test_an_unknown_eta_is_not_a_number(self):
        frames = progress.parse_tqdm("0/5 [00:00<?, ?step/s]")
        self.assertEqual(frames, [])


class TestMarkers(unittest.TestCase):
    def test_checkpoints_are_exact(self):
        marks = progress.parse_markers(CREATE_END)
        self.assertEqual(marks["checkpoints"], [70000, 80000])
        self.assertTrue(marks["finished"])
        self.assertIsNone(marks["failure"])

    def test_a_traceback_becomes_the_reason_it_stopped(self):
        text = (
            CREATE_START + "\nTraceback (most recent call last):\n"
            '  File "train.py", line 1, in <module>\n'
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        )
        marks = progress.parse_markers(text)
        self.assertIn("OutOfMemoryError", marks["failure"])
        self.assertIn("2.00 GiB", marks["failure"])

    def test_the_drivers_own_verdict_is_read_when_nothing_raised(self):
        marks = progress.parse_markers("❌ row 2 FAILED (cube-pnp-new / diffusion)")
        self.assertIn("row 2 FAILED", marks["failure"])


class TestSummary(unittest.TestCase):
    def make(self, text=THANOS_BODY, tail=None, age=0.0):
        parsed = progress.parse_log(text, head_text=HEAD)
        if tail:
            parsed["last_frame"] = progress.parse_tqdm(tail.replace("\r", "\n"))[-1]
        return progress.summarise(parsed, age_s=age)

    def test_where_the_run_is(self):
        # The later of the two sources wins. Here the body's last metric line
        # (71400) is a LATER moment than the tail frame kept as a fixture
        # (71379), which is why the rule is "the freshest" and not "the tail":
        # the bar is written every step and a metric line every hundredth, so
        # in a real file either can be the more current one.
        summary = self.make(tail=THANOS_TAIL_RAW)
        self.assertEqual(summary["step"], 71400)
        self.assertEqual(summary["total_steps"], 80000)
        self.assertAlmostEqual(summary["fraction"], 0.8925)
        self.assertEqual(summary["eta_s"], 2355.0)
        self.assertAlmostEqual(summary["loss_last"], 0.080)
        self.assertAlmostEqual(summary["loss_min"], 0.078)
        self.assertAlmostEqual(summary["gpu_mem_gb_max"], 11.27)

    def test_a_bar_written_after_the_last_metric_line_is_the_current_step(self):
        later = THANOS_TAIL_RAW.replace("71379/80000", "71455/80000")
        self.assertEqual(self.make(tail=later)["step"], 71455)

    def test_a_finished_run_says_so(self):
        parsed = progress.parse_log(CREATE_START + "\n" + CREATE_END, head_text=HEAD)
        self.assertEqual(progress.summarise(parsed, age_s=1e6)["state"], "done")
        self.assertEqual(progress.summarise(parsed, age_s=1e6)["checkpoint"], 80000)

    def test_a_run_whose_machine_went_away_is_stalled_not_running(self):
        # thanos rebooted under a run on 27 August and nothing said so: the log
        # simply stopped, and every count still agreed.
        parsed = progress.parse_log(CREATE_START, head_text=HEAD)
        self.assertEqual(progress.summarise(parsed, age_s=30.0)["state"], "running")
        self.assertEqual(progress.summarise(parsed, age_s=99999.0)["state"], "stalled")

    def test_a_quiet_but_living_run_is_not_libelled(self):
        # The threshold follows the run's own logging interval, so a policy that
        # writes a line every eleven seconds is stalled long before one that
        # writes every ten minutes.
        parsed = progress.parse_log(CREATE_START, head_text=HEAD)
        self.assertEqual(progress.summarise(parsed, age_s=600.0)["state"], "running")

    def test_a_failure_outranks_everything_else(self):
        parsed = progress.parse_log(CREATE_START, head_text=HEAD)
        parsed["failure"] = "torch.OutOfMemoryError: CUDA out of memory"
        parsed["finished"] = True
        self.assertEqual(progress.summarise(parsed, age_s=0.0)["state"], "failed")

    def test_an_empty_log_summarises_without_raising(self):
        summary = progress.summarise(progress.parse_log(""), age_s=None)
        self.assertEqual(summary["step"], 0)
        self.assertIsNone(summary["loss_last"])
        self.assertEqual(summary["state"], "running")


class TestThinning(unittest.TestCase):
    def test_a_short_curve_is_untouched(self):
        points = [{"step": i} for i in range(50)]
        self.assertEqual(progress.thin(points, 100), points)

    def test_a_long_curve_keeps_its_ends(self):
        points = [{"step": i} for i in range(5000)]
        kept = progress.thin(points, 500)
        self.assertEqual(len(kept), 500)
        self.assertEqual(kept[0]["step"], 0)
        self.assertEqual(kept[-1]["step"], 4999)


# ── Which log, on which machine ──────────────────────────────────────────────


THANOS = {
    "name": "thanos",
    "ssh": "thanos",
    "kind": "ssh",
    "repo": "~/project/so101_garment",
    "scratch": "~/.cache/huggingface/lerobot",
    "stage": "{scratch}/local",
    "limits": {},
}
LOCAL = {**THANOS, "name": "local", "kind": "local", "outputs": "{repo}/outputs"}


class TestWhereTheLogIs(unittest.TestCase):
    def test_the_output_root_follows_the_drivers(self):
        self.assertEqual(
            runs.out_roots(THANOS),
            ["~/.cache/huggingface/lerobot/so101_outputs"],
        )

    def test_a_second_root_is_scanned_when_one_is_named(self):
        # A console started through setup.sh puts SO101_OUTPUT_DIR in the repo,
        # so a locally started run lands there and not in the cache.
        self.assertEqual(
            runs.out_roots(LOCAL),
            [
                "~/.cache/huggingface/lerobot/so101_outputs",
                "~/project/so101_garment/outputs",
            ],
        )

    def test_the_run_directory_is_the_camera_view(self):
        self.assertEqual(
            runs.run_dir_name({"dataset": "fold-short", "cameras": ["all"]}),
            "fold-short__all",
        )

    def test_a_run_tag_wins(self):
        # It has to: a probe run under the real name leaves a checkpoint the
        # long run then reuses, reporting success at the probe's step count.
        self.assertEqual(
            runs.run_dir_name(
                {"dataset": "fold-short", "cameras": ["all"], "run_tag": "probe5cam"}
            ),
            "probe5cam",
        )

    def test_a_name_that_could_not_be_sent_is_refused(self):
        runs.check_name("cube-pnp-new__central+wrist_left")
        for bad in ("a b", "a;rm -rf /", "$(x)", "", "../etc"):
            with self.assertRaises(runs.RunsError):
                runs.check_name(bad)


class TestTheCommands(unittest.TestCase):
    def test_the_log_is_filtered_on_the_far_side(self):
        command = runs.read_command("/scratch/x/logs/train_act.log")
        self.assertIn("tr '\\r' '\\n'", command)
        self.assertIn("grep -aE", command)
        self.assertIn("head -c", command)
        self.assertIn("date +%s", command)

    def test_the_command_holds_no_non_ascii(self):
        # It reaches a remote login shell, and the driver's tick and cross do
        # not survive every locale on the way.
        runs.read_command("/x/y.log").encode("ascii")
        runs.discover_command(THANOS).encode("ascii")

    def test_the_local_machine_needs_no_ssh(self):
        self.assertEqual(runs.command_argv(LOCAL, "true")[:2], ["bash", "-lc"])
        self.assertEqual(runs.command_argv(THANOS, "true")[0], "ssh")

    def test_discovery_looks_in_every_root(self):
        command = runs.discover_command(LOCAL)
        self.assertIn(
            "~/.cache/huggingface/lerobot/so101_outputs/vla_real_long", command
        )
        self.assertIn("~/project/so101_garment/outputs/vla_real_long", command)


class TestReadingTheAnswers(unittest.TestCase):
    def test_sections_are_split_on_their_markers(self):
        text = f"{runs.SENTINEL}head\na\nb\n{runs.SENTINEL}body\nc\n"
        self.assertEqual(runs.parse_sections(text), {"head": "a\nb", "body": "c"})

    def test_a_listing_becomes_runs(self):
        listing = (
            "LOG /scratch/so101_outputs/vla_real_long/cube-pnp-new__all/logs/"
            "train_act.log 133410 1787000000\n"
            "CKPT /scratch/so101_outputs/vla_real_long/cube-pnp-new__all/train/act/"
            "checkpoints/last 080000\n"
            f"{runs.SENTINEL}now\n1787000600\n"
        )
        found = runs.parse_discovery(listing)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["run"], "cube-pnp-new__all")
        self.assertEqual(found[0]["policy"], "act")
        self.assertEqual(found[0]["checkpoint"], 80000)
        # Age against the REMOTE clock, never this machine's.
        self.assertEqual(found[0]["age_s"], 600.0)

    def test_a_run_directory_with_a_plus_in_it_survives(self):
        listing = (
            "LOG /s/so101_outputs/vla_real_long/cube-pnp-new__central+wrist_left/"
            "logs/train_pi05.log 57287 1787000000\n"
        )
        found = runs.parse_discovery(listing)
        self.assertEqual(found[0]["run"], "cube-pnp-new__central+wrist_left")
        self.assertEqual(found[0]["policy"], "pi05")


if __name__ == "__main__":
    unittest.main()


class TestSectionsSurviveALogWithNoFinalNewline(unittest.TestCase):
    """The bug this catches was found by calling the route, not by a test.

    A progress bar's last frame ends without a newline, so the marker that
    follows it lands on the END of that line rather than the start of its own.
    ``parse_sections`` then never sees the marker: the operator is shown
    ``@@so101:stat`` at the end of the log tail, and the file's modification
    time -- the whole basis for calling a run stalled -- is silently absent.
    """

    def test_the_command_terminates_every_section(self):
        command = runs.read_command("/x/train_act.log")
        # Each section that can end mid-line is followed by a bare `echo`.
        self.assertIn("| tail -n 4; echo; ", command)
        self.assertIn('"$L"; echo; ', command)

    def test_a_marker_stuck_to_a_bar_is_not_a_section(self):
        stuck = (
            f"{runs.SENTINEL}tail\n"
            "Training:  96%|...| 76706/80000 [8:20:57<15:02,  3.65step/s]"
            f"{runs.SENTINEL}stat\n6242789 1787942791\n1787942792\n"
        )
        # Without the terminating echo there is no `stat` section at all, and
        # the tail keeps a marker that means nothing to a reader.
        self.assertNotIn("stat", runs.parse_sections(stuck))

    def test_a_terminated_answer_yields_the_age(self):
        proper = (
            f"{runs.SENTINEL}tail\n"
            "Training:  96%|...| 76706/80000 [8:20:57<15:02,  3.65step/s]\n"
            f"{runs.SENTINEL}stat\n6242789 1787942791\n1787942792\n"
        )
        sections = runs.parse_sections(proper)
        self.assertEqual(
            sections["stat"].split(), ["6242789", "1787942791", "1787942792"]
        )
        self.assertNotIn(runs.SENTINEL, sections["tail"])


class TestALogCutThroughACharacter(unittest.TestCase):
    """`head -c` counts bytes, and a progress bar is drawn in three-byte glyphs.

    So the head of a log is routinely cut through the middle of one. Decoding
    that strictly raises, and the whole run then cannot be watched -- which is
    how this was found: a real local run, at byte 20012.
    """

    LOCAL = {
        "name": "here",
        "kind": "local",
        "ssh": "-",
        "repo": ".",
        "scratch": "/tmp",
        "stage": "{scratch}/local",
    }

    def test_a_truncated_glyph_does_not_take_the_whole_read_down(self):
        # The first two bytes of U+2588 FULL BLOCK, which is what tqdm draws.
        out, problem = runs.run_command(
            self.LOCAL, r"printf 'ok\xe2\x96'", timeout=20.0
        )
        self.assertIsNone(problem)
        self.assertTrue(out.startswith("ok"))
