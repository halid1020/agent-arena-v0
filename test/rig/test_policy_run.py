"""The throttle on an autonomous rollout, and when the next chunk is asked for.

Both are pure arithmetic that decides whether a robot moves, so both are tested
away from the robot. The step machine is the interesting one: one press must
execute one chunk and stop, whatever the chunk's length, and a mode change must
invalidate a plan made before it -- executing a plan drawn from a two-minute-old
photograph is the failure this design exists to prevent.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_policy_run
"""

import unittest

from actoris_harena.deploy.policy_run import RunControl, gate, prefetch_threshold


class TestGate(unittest.TestCase):
    def test_running_serves_whenever_there_is_something_to_serve(self):
        self.assertEqual(gate("run", 5, 0), ("serve", 0))
        self.assertEqual(gate("run", 0, 0), ("wait", 0))

    def test_holding_never_serves_however_full_the_queue(self):
        self.assertEqual(gate("hold", 32, 0), ("hold", 0))

    def test_a_step_takes_its_budget_from_the_chunk_that_landed(self):
        # Nothing queued yet: the press has been made, the chunk is coming.
        self.assertEqual(gate("step", 0, 0), ("wait", 0))
        # A 32-action chunk arrives; the budget is the rest of it.
        self.assertEqual(gate("step", 32, 0), ("serve", 31))
        self.assertEqual(gate("step", 31, 31), ("serve", 30))

    def test_a_step_waits_rather_than_ending_early_on_an_empty_queue(self):
        # Mid-chunk with nothing queued is a slow link, not the end of the plan.
        self.assertEqual(gate("step", 0, 7), ("wait", 7))


class TestPrefetchThreshold(unittest.TestCase):
    def test_a_slow_round_trip_raises_the_threshold(self):
        # 0.6 s at 30 Hz spends 18 actions; a third of a 32-action chunk is 10,
        # which is why a diffusion rollout stutters on the default.
        self.assertEqual(prefetch_threshold(10, None, 0.6, 30.0, 32), 27)

    def test_a_fast_policy_keeps_its_static_default(self):
        self.assertEqual(prefetch_threshold(33, None, 0.018, 30.0, 100), 33)

    def test_it_never_reaches_the_chunk_length(self):
        # A request would otherwise be in flight permanently.
        self.assertEqual(prefetch_threshold(10, None, 9.0, 30.0, 32), 31)

    def test_an_explicit_prefetch_is_obeyed_in_both_directions(self):
        self.assertEqual(prefetch_threshold(10, 4, 9.0, 30.0, 32), 4)
        self.assertEqual(prefetch_threshold(10, 40, 0.0, 30.0, 32), 40)

    def test_no_measurement_yet_leaves_the_default_alone(self):
        self.assertEqual(prefetch_threshold(10, None, 0.0, 30.0, 32), 10)


class TestRunControl(unittest.TestCase):
    def test_a_run_serves_every_tick(self):
        control = RunControl()
        self.assertEqual([control.decide(9) for _ in range(3)], ["serve"] * 3)
        self.assertEqual(control.mode, "run")

    def test_one_step_executes_one_chunk_and_then_holds(self):
        control = RunControl()
        control.request("step")
        depths = [0, 3, 2, 1, 9]  # the chunk lands on the second tick
        served = [control.decide(d) for d in depths]
        self.assertEqual(served, ["wait", "serve", "serve", "serve", "hold"])
        self.assertEqual(control.mode, "hold")

    def test_leaving_a_pause_marks_the_queue_stale_exactly_once(self):
        # The plan queued before a pause was drawn from a picture of the world
        # that may be minutes old. Resuming is where that is thrown away.
        control = RunControl()
        control.request("hold")
        control.request("run")
        self.assertTrue(control.queue_stale())
        self.assertFalse(control.queue_stale())

    def test_previewing_a_plan_does_not_throw_it_away(self):
        # The whole point of preview is that the plan on the screen is the plan
        # that executes; dropping it would make the inspection meaningless.
        control = RunControl()
        control.request("preview")
        self.assertFalse(control.queue_stale())
        control.request("step")
        self.assertFalse(control.queue_stale())

    def test_a_preview_serves_nothing_however_full_the_queue(self):
        control = RunControl(mode="preview")
        self.assertEqual([control.decide(9) for _ in range(3)], ["hold"] * 3)

    def test_a_step_taken_from_preview_returns_to_preview(self):
        # So that inspect-then-execute is a cycle rather than a one-way trip.
        control = RunControl(mode="preview")
        control.request("step")
        for depth in (3, 2, 1):
            control.decide(depth)
        self.assertEqual(control.mode, "preview")

    def test_a_step_taken_from_a_pause_still_ends_in_a_pause(self):
        control = RunControl(mode="hold")
        control.request("step")
        for depth in (2, 1):
            control.decide(depth)
        self.assertEqual(control.mode, "hold")

    def test_asking_for_the_mode_already_in_force_changes_nothing(self):
        control = RunControl()
        control.request("run")
        self.assertFalse(control.queue_stale())

    def test_a_reset_is_asked_for_once_and_read_once(self):
        # The loop must not put the scene back on every tick after one press.
        control = RunControl()
        self.assertFalse(control.reset_requested())
        control.request("reset")
        self.assertTrue(control.reset_requested())
        self.assertFalse(control.reset_requested())

    def test_a_reset_holds_the_throttle_before_the_loop_has_read_it(self):
        # There are ticks between the flag going up and the loop seeing it, and
        # none of them may serve an action to a rig about to be released.
        control = RunControl(mode="run")
        self.assertEqual(control.request("reset"), "hold")
        self.assertEqual(control.mode, "hold")

    def test_a_reset_is_not_a_stop_and_does_not_invalidate_by_itself(self):
        control = RunControl()
        control.request("reset")
        self.assertFalse(control.stop_requested())
        self.assertFalse(control.queue_stale())

    def test_stop_ends_a_trial_rather_than_latching(self):
        # Not a latch: the run stays up, so a second press minutes later is a
        # second request and not the first one still being true.
        control = RunControl()
        self.assertEqual(control.request("stop"), "hold")
        self.assertTrue(control.stop_requested())
        self.assertFalse(control.stop_requested())
        self.assertEqual(control.mode, "hold")

    def test_a_stop_does_not_ask_for_the_scene_back(self):
        control = RunControl()
        control.request("stop")
        self.assertFalse(control.reset_requested())

    def test_a_step_after_a_release_starts_from_hold_not_from_preview(self):
        # Otherwise one step of the NEXT trial would drop back into a preview
        # belonging to the attempt that has just ended.
        control = RunControl(mode="preview")
        control.request("stop")
        control.request("step")
        control.decide(depth=1)
        self.assertEqual(control.mode, "hold")

    def test_consent_can_be_withdrawn_when_the_arms_are_let_go(self):
        control = RunControl()
        control.request("arm")
        self.assertTrue(control.armed)
        control.disarm()
        self.assertFalse(control.armed)
        # And given again, because the next trial asks the same question.
        control.request("arm")
        self.assertTrue(control.armed)

    def test_an_unknown_mode_is_refused(self):
        control = RunControl()
        with self.assertRaises(ValueError):
            control.request("faster")

    def test_the_snapshot_carries_the_mode_with_whatever_was_published(self):
        control = RunControl()
        control.publish(tick=7, queue=12)
        control.request("hold")
        snapshot = control.snapshot()
        self.assertEqual(snapshot["tick"], 7)
        control.publish(tick=8)
        self.assertEqual(control.snapshot()["mode"], "hold")


if __name__ == "__main__":
    unittest.main()
