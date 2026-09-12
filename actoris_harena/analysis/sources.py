"""Where the observations to analyse come from: a dataset, or a real rollout.

Two sources, one shape. Both yield ``(index, state, images, action)`` where
``action`` is the ground truth if there is one and ``None`` if there is not --
which is the whole difference between them. A recorded episode has the action a
human teleoperated, so attribution can be read beside how far the policy's plan
was from it. A rollout has no ground truth at all; what it has instead is what
actually happened on the rig.

A rollout is only analysable if it was recorded with ``--log-frames``. It is off
by default because the frames are the largest part of an observation, so this
says plainly what is missing rather than producing an empty study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _to_hwc_uint8(array: np.ndarray) -> np.ndarray:
    """A dataset frame (CHW float in [0,1]) as the HWC uint8 the rig produces."""
    array = np.asarray(array)
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3)
        and array.shape[0] < array.shape[-1]
    ):
        array = array.transpose(1, 2, 0)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    return array


def parse_range(text: str) -> "list[int]":
    """``'0-9,12'`` -> ``[0..9, 12]``. Empty means every episode."""
    out: "list[int]" = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


class DatasetSource:
    """Recorded episodes, through LeRobot's own reader."""

    def __init__(self, root: str, episodes: str = "", every: int = 20, cameras=None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = str(Path(root).expanduser().resolve())
        self.dataset = LeRobotDataset(
            repo_id="local/analysis", root=self.root, video_backend="pyav"
        )
        self.keys = [
            k for k in self.dataset.meta.features if k.startswith("observation.images")
        ]
        self.cameras = [k.split(".")[-1] for k in self.keys]
        if cameras:
            missing = [c for c in cameras if c not in self.cameras]
            if missing:
                raise SystemExit(
                    f"❌ this dataset has no camera(s) {', '.join(missing)}; it has "
                    f"{', '.join(self.cameras)}"
                )
        self.every = max(1, int(every))
        wanted = parse_range(episodes)
        self.episodes = wanted or list(range(self.dataset.num_episodes))
        self.fps = float(self.dataset.meta.fps)

    def rows(self, episode: int) -> "list[int]":
        meta = self.dataset.meta.episodes
        lo = int(meta["dataset_from_index"][episode])
        hi = int(meta["dataset_to_index"][episode])
        return list(range(lo, hi, self.every))

    def observations(self, episode: int):
        for row in self.rows(episode):
            sample = self.dataset[row]
            images = {
                key.split(".")[-1]: _to_hwc_uint8(sample[key]) for key in self.keys
            }
            yield (
                row,
                np.asarray(sample["observation.state"], dtype=np.float32),
                images,
                np.asarray(sample["action"], dtype=np.float32),
            )

    def actions(self, episode: int) -> np.ndarray:
        """Every action of the episode, for the phase segmentation."""
        meta = self.dataset.meta.episodes
        lo = int(meta["dataset_from_index"][episode])
        hi = int(meta["dataset_to_index"][episode])
        return np.stack(
            [
                np.asarray(self.dataset[i]["action"], dtype=np.float32)
                for i in range(lo, hi)
            ]
        )

    def describe(self) -> str:
        return (
            f"dataset {Path(self.root).name}: {self.dataset.num_episodes} episode(s), "
            f"{len(self.cameras)} camera(s) at {self.fps:g} fps"
        )


class RunSource:
    """One real rollout, from the frames its run log kept."""

    def __init__(self, root: str, cameras=None):
        import cv2

        self.cv2 = cv2
        self.root = Path(root).expanduser().resolve()
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.frames_dir = self.root / "frames"
        if not self.frames_dir.is_dir():
            raise SystemExit(
                f"❌ {self.root} kept no frames, so there is nothing to attribute a "
                "plan to. Record the rollout again with --log-frames "
                "(tool/run_policy.py); it is off by default because the frames are "
                "the largest part of an observation."
            )
        from actoris_harena.deploy.policy_log import read_jsonl

        self.chunks = {int(c["seq"]): c for c in read_jsonl(self.root / "chunks.jsonl")}
        self.plans = sorted(
            int(p.name) for p in self.frames_dir.iterdir() if p.name.isdigit()
        )
        first = self.frames_dir / f"{self.plans[0]:06d}" if self.plans else None
        self.cameras = (
            sorted(p.stem for p in first.glob("*.jpg"))
            if first and first.is_dir()
            else []
        )
        self.fps = float(self.meta.get("hz") or 30.0)
        self.episodes = [0]

    def observations(self, _episode: int = 0):
        for seq in self.plans:
            directory = self.frames_dir / f"{seq:06d}"
            images = {}
            for path in sorted(directory.glob("*.jpg")):
                frame = self.cv2.imread(str(path))
                if frame is not None:
                    images[path.stem] = frame[:, :, ::-1].copy()
            record = self.chunks.get(seq) or {}
            state = record.get("state")
            if state is None:
                continue
            yield seq, np.asarray(state, dtype=np.float32), images, None

    def actions(self, _episode: int = 0) -> np.ndarray:
        """What the policy actually planned, chunk by chunk: the phase source here.

        A rollout has no ground truth, so the phases are segmented from the
        policy's OWN commands. That is the honest thing to segment by -- it is
        what the grippers were told to do -- but it is not the same quantity as a
        dataset's teleoperated action, and a figure comparing the two should say
        so.
        """
        planned = [
            np.asarray(self.chunks[seq]["actions"], dtype=np.float32)[0]
            for seq in self.plans
            if seq in self.chunks and self.chunks[seq].get("actions")
        ]
        return np.stack(planned) if planned else np.zeros((0, 12), dtype=np.float32)

    def describe(self) -> str:
        return (
            f"rollout {self.root.name}: {len(self.plans)} plan(s) with frames, "
            f"{len(self.cameras)} camera(s), task {self.meta.get('task', '')!r}"
        )
