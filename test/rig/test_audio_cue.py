"""Unit tests for actoris_harena.recording.audio_cue (best-effort record cues).

No sound is actually played: playback is patched out. Exercises the pure name/
player resolution and that AudioCue never blocks or raises regardless of whether
a player or sound file is available.
"""

import unittest
from unittest import mock

from actoris_harena.recording import audio_cue
from actoris_harena.recording.audio_cue import AudioCue, resolve_player, resolve_sound


class TestResolveSound(unittest.TestCase):
    def test_shipped_cues_resolve(self):
        # The two packaged cues exist next to the module.
        self.assertIsNotNone(resolve_sound("start.wav"))
        self.assertIsNotNone(resolve_sound("stop.wav"))

    def test_missing_name_is_none(self):
        self.assertIsNone(resolve_sound(""))
        self.assertIsNone(resolve_sound("does_not_exist.wav"))

    def test_absolute_nonexistent_is_none(self):
        self.assertIsNone(resolve_sound("/nope/missing.wav"))


class TestResolvePlayer(unittest.TestCase):
    def test_returns_argv_prefix_or_none(self):
        result = resolve_player()
        self.assertTrue(result is None or (isinstance(result, list) and result))

    def test_none_when_no_player_on_path(self):
        with mock.patch.object(audio_cue.shutil, "which", return_value=None):
            self.assertIsNone(resolve_player())


class TestAudioCue(unittest.TestCase):
    def test_disabled_is_silent_noop(self):
        # Disabled: play() must not touch the player at all.
        with mock.patch.object(audio_cue.subprocess, "Popen") as popen:
            AudioCue(enabled=False, start_sound="start.wav").play("start")
            popen.assert_not_called()

    def test_play_launches_player_when_available(self):
        with mock.patch.object(
            audio_cue, "resolve_player", return_value=["/usr/bin/paplay"]
        ):
            cue = AudioCue(enabled=True, start_sound="start.wav", stop_sound="stop.wav")
            with mock.patch.object(audio_cue.subprocess, "Popen") as popen:
                cue.play("start")
                popen.assert_called_once()

    def test_falls_back_to_bell_without_player(self):
        with mock.patch.object(audio_cue, "resolve_player", return_value=None):
            cue = AudioCue(enabled=True, start_sound="start.wav")
            with mock.patch.object(audio_cue.subprocess, "Popen") as popen:
                cue.play("start")  # no player -> bell, never Popen
                popen.assert_not_called()

    def test_unknown_event_never_raises(self):
        cue = AudioCue(enabled=True, start_sound="start.wav")
        cue.play("nonsense")  # must not raise

    def test_popen_failure_swallowed(self):
        with mock.patch.object(
            audio_cue, "resolve_player", return_value=["/usr/bin/paplay"]
        ):
            cue = AudioCue(enabled=True, start_sound="start.wav")
            with mock.patch.object(
                audio_cue.subprocess, "Popen", side_effect=OSError("boom")
            ):
                cue.play("start")  # must not raise


if __name__ == "__main__":
    unittest.main()
