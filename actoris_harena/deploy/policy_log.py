"""What one autonomous rollout did, written down while it happens.

A failed grasp is over in a second and cannot be paused mid-air, so the run has
to leave enough behind to be examined afterwards: what the policy was shown,
what it planned from that, and what the arms actually did with the plan.

Three files, because they are written at three different rates.
``ticks.parquet`` has one row per control tick -- the measured joints, the goal
written, the mode, the queue depth -- and is what a tracking plot is drawn from.
``chunks.jsonl`` has one line per *plan*, appended the moment it lands, so a run
that dies mid-episode still leaves its plans on disk; a chunk is thirty-two
actions and a state, and the sequence numbers tie it back to the ticks that
executed it. ``trials.jsonl`` has one line per *attempt*, appended when somebody
says how it went: a run is a sequence of attempts at the same rig, and without a
verdict against each trial number a comparison between two checkpoints lives in
somebody's notebook rather than on disk.

Images are not stored by default: they are the largest part of an observation by
far, and the questions this log usually answers are about joints. They are the
one thing an attribution study cannot do without, though -- an attribution needs
the pixels the plan was drawn from -- so ``save_frames`` writes the observation
behind each *chunk* (not each tick: a chunk covers seconds) under
``frames/<seq>/<camera>.jpg``. Off unless asked for. Written by
``tool/run_policy.py``; read by ``common.analysis``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from actoris_harena.outputs import output_root

RUNS_SUBDIR = "policy_runs"
TICK_COLUMNS = ("t", "trial", "mode", "queue", "served", "state", "commanded")

#: What an attempt can be judged to have been. ``discard`` is not a third
#: outcome between the other two -- it says the attempt should not be counted at
#: all (the scene was wrong, somebody bumped the table, the tunnel dropped), and
#: a rate computed over trials must leave those out rather than score them as
#: failures.
OUTCOMES = ("success", "failure", "discard")

#: JPEG quality for a saved observation. High: these frames are read back by an
#: attribution study, and compression artefacts are indistinguishable from the
#: fine texture a tactile camera exists to capture.
FRAME_QUALITY = 95


def runs_root() -> Path:
    """Where run logs live: beside every other output of a session."""
    return output_root() / RUNS_SUBDIR


class RunLog:
    """One directory per rollout. Create it, feed it, close it."""

    def __init__(self, root: Path, save_frames: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ticks: "list[dict]" = []
        self._chunks = open(self.root / "chunks.jsonl", "a", encoding="utf-8")
        self._trials = open(self.root / "trials.jsonl", "a", encoding="utf-8")
        self._last_seq = 0
        #: Whether to keep the pixels each plan was drawn from. Off by default:
        #: see the module docstring for the one thing that needs them.
        self.save_frames = bool(save_frames)
        self._started = time.time()
        #: Which attempt each tick belongs to. A run outlives its trials -- the
        #: arms are released and taken up again between them -- so without this
        #: column two attempts at the same scene read as one long series with an
        #: unexplained pause in the middle.
        self.trial = 0

    @classmethod
    def create(
        cls,
        task: str,
        source: str,
        hz: float,
        root: "Path | None" = None,
        save_frames: bool = False,
        checkpoint: str = "",
        policy_type: str = "",
    ) -> "RunLog":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log = cls(Path(root or runs_root()) / stamp, save_frames=save_frames)
        (log.root / "meta.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "source": source,
                    "hz": hz,
                    "started": stamp,
                    "frames": bool(save_frames),
                    # WHICH weights were scored. Without this a directory of
                    # runs records that something was tried, not what.
                    "checkpoint": str(checkpoint),
                    "policy_type": str(policy_type),
                },
                indent=2,
            )
        )
        return log

    def tick(
        self,
        t: float,
        state: np.ndarray,
        commanded: "np.ndarray | None",
        mode: str,
        queue: int,
    ) -> None:
        self._ticks.append(
            {
                "t": round(float(t), 4),
                "trial": int(self.trial),
                "mode": mode,
                "queue": int(queue),
                "served": commanded is not None,
                "state": np.asarray(state, dtype=np.float32),
                "commanded": (
                    None if commanded is None else np.asarray(commanded, np.float32)
                ),
            }
        )

    def new_trial(self) -> int:
        """Begin counting another attempt. Returns the trial now in force."""
        self.trial += 1
        return self.trial

    def trial_outcome(self, outcome: str, notes: str = "") -> dict:
        """Write down how the attempt now in force went.

        Appended the moment it is given, like a chunk: a verdict typed while the
        rig is still in front of you is worth more than one reconstructed later,
        and a run that dies afterwards keeps it either way. Called more than once
        for the same trial, the later line wins -- somebody changing their mind
        having looked again is the normal case, and rewriting the file to remove
        the earlier line would lose that they did.
        """
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        record = {
            "trial": int(self.trial),
            "outcome": outcome,
            "notes": str(notes),
            "t": round(time.time() - self._started, 3),
        }
        self._trials.write(json.dumps(record) + "\n")
        self._trials.flush()
        return record

    def _write_frames(self, seq: int, images: "dict | None") -> int:
        """Save the observation behind one plan. Returns how many were written.

        Best effort by design: this is a diagnostic, and a run must not die
        because a disk filled or a frame arrived in a shape cv2 will not encode.
        """
        if not images:
            return 0
        import cv2

        out = self.root / "frames" / f"{int(seq):06d}"
        out.mkdir(parents=True, exist_ok=True)
        written = 0
        for name, frame in images.items():
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                continue
            # The pipeline carries RGB (LeRobot's preprocess_observation expects
            # it); cv2 writes BGR, so this swap is not optional.
            ok = cv2.imwrite(
                str(out / f"{name}.jpg"),
                array[:, :, ::-1],
                [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_QUALITY],
            )
            written += int(bool(ok))
        return written

    def note_chunk(self, source) -> bool:
        """Append the source's latest plan, if it is one this log has not seen."""
        seq = int(getattr(source, "last_chunk_seq", 0) or 0)
        chunk = getattr(source, "last_chunk", None)
        if seq <= self._last_seq or chunk is None:
            return False
        self._last_seq = seq
        sent = getattr(source, "last_sent", lambda: None)()
        frames = 0
        if self.save_frames and sent is not None:
            try:
                frames = self._write_frames(seq, sent[1])
            except Exception as exc:  # noqa: BLE001 -- never kill a rollout
                print(f"⚠️  could not save frames for chunk {seq}: {exc}")
        self._chunks.write(
            json.dumps(
                {
                    "seq": seq,
                    "trial": int(self.trial),
                    "frames": frames,
                    "at": round(float(getattr(source, "last_chunk_at", 0.0)), 4),
                    "state": (
                        None if sent is None else np.asarray(sent[0]).round(4).tolist()
                    ),
                    "actions": np.asarray(chunk, dtype=float).round(4).tolist(),
                    "round_trip_s": round(float(source.round_trip_s), 4),
                    "server_infer_s": round(
                        float(getattr(source, "server_infer_s", 0.0)), 4
                    ),
                }
            )
            + "\n"
        )
        self._chunks.flush()
        return True

    def close(self) -> Path:
        """Write the ticks and return the directory. Safe to call twice."""
        import pandas as pd

        if not self._chunks.closed:
            self._chunks.close()
        if not self._trials.closed:
            self._trials.close()
        path = self.root / "ticks.parquet"
        frame = pd.DataFrame(self._ticks, columns=list(TICK_COLUMNS))
        frame.to_parquet(path, index=False)
        return self.root


def read_jsonl(path: Path) -> "list[dict]":
    """Every well-formed line of a JSONL file; a truncated last line is dropped.

    These files are appended to while a rollout runs, so a run killed mid-write
    leaves a partial final line. That is the normal way for one to end, not a
    corruption, and reading it must not raise.
    """
    out: "list[dict]" = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a partial trailing line: the run was killed mid-append
    return out


def trial_verdicts(root: Path) -> "dict[int, dict]":
    """The final verdict per trial number: the LAST line wins (see trial_outcome)."""
    verdicts: "dict[int, dict]" = {}
    for record in read_jsonl(Path(root) / "trials.jsonl"):
        try:
            verdicts[int(record["trial"])] = record
        except (KeyError, TypeError, ValueError):
            continue
    return verdicts


def score(root: Path) -> "dict[str, int | float | None]":
    """Trials, successes and success rate for one run directory.

    ``discard`` is excluded from the denominator rather than counted against the
    policy -- an attempt nobody thinks should count is not a failure. A run with
    nothing but discards has no rate, and says ``None`` rather than zero.
    """
    verdicts = trial_verdicts(root)
    counted = [v for v in verdicts.values() if v.get("outcome") != "discard"]
    wins = sum(1 for v in counted if v.get("outcome") == "success")
    return {
        "trials": len(counted),
        "successes": wins,
        "discarded": len(verdicts) - len(counted),
        "rate": (wins / len(counted)) if counted else None,
    }
