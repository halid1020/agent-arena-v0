import json
import time
import numpy as np
from actoris_harena.utilities.logger.logger_interface import Logger

import wandb
import os
import json
from pathlib import Path
import re

class WandbLogger(Logger):

    @staticmethod
    def _find_previous_run_id(logdir):
        """
        Returns run_id by reading folders named 'run-<id>' in <logdir>/wandb.
        Picks the most recent one if multiple exist.
        """
        logdir = Path(logdir)
        wandb_dir = logdir / "wandb"

        print("wandb_dir:", wandb_dir)

        if not wandb_dir.exists():
            return None

        runs = []
        for item in wandb_dir.iterdir():
            if item.is_dir() and item.name.startswith("run-"):
                # split: run-abc123 → ["run", "abc123"] → get "abc123"
                run_id = item.name.split("-")[-1]
                runs.append((item.stat().st_mtime, run_id))

        if not runs:
            return None

        # return the newest run
        runs.sort(reverse=True)
        latest_run_id = runs[0][1]
        print("Auto-resume run ID:", latest_run_id)
        return latest_run_id

    def __init__(self, logdir, project, name, config, run_id=None):

        self._logdir = Path(logdir)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}

        self.project = project
        self.name = name
        self.config = config

        # ---------------------------------------------------------------------
        # 🔍 AUTO-RESUME LOGIC
        # ---------------------------------------------------------------------
        auto_resume_id = None
        if run_id is None:
            auto_resume_id = self._find_previous_run_id(logdir)

        effective_run_id = run_id or auto_resume_id
        effective_resume = "allow" if effective_run_id else "never"

        print(f"[WandbLogger] run_id={effective_run_id}, resume={effective_resume}")

        # ---------------------------------------------------------------------
        # Initialize W&B
        # ---------------------------------------------------------------------
        self.wandb = wandb.init(
            project=project,
            name=name,
            config=config,
            id=effective_run_id,
            resume=effective_resume,
            dir=str(logdir),
        )
        self.step = 0

    def _compute_fps(self, step):
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            self._last_step = step
            return 0.0
        fps = (step - self._last_step) / (now - self._last_time)
        self._last_time = now
        self._last_step = step
        return fps

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        # Expect HWC or CHW image array
        self._images[name] = np.array(value)

    def video(self, name, value):
        # Expect B T H W C or T H W C (adjust as needed)
        self._videos[name] = np.array(value)


    def log(self, metrics, step=None, fps=10) -> None:
        """
        Logs scalars, images, and videos (supports numpy arrays or file paths).
        """
        if self.wandb is None:
            return
        
        if step == None:
            step = self.step
        self.step = step
        
        processed_metrics = {}
        for key, value in metrics.items():
            # 1. Handle numpy arrays (e.g., video frames)
            if isinstance(value, np.ndarray):
                # Expected shape: (T, H, W, C)
                if value.ndim == 4 and value.dtype in [np.uint8, np.float32, np.float64]:
                    # Ensure dtype uint8 in [0,255]
                    if value.dtype != np.uint8:
                        value = np.clip(value * 255, 0, 255).astype(np.uint8)
                    # FIX: Use wandb.Video, not self.wandb.Video
                    processed_metrics[key] = wandb.Video(value, fps=fps, format="mp4")
                else:
                    processed_metrics[key] = value
                
            # 2. Handle file paths (e.g., image or video files)
            elif isinstance(value, str) and os.path.exists(value):
                ext = os.path.splitext(value)[-1].lower()
                if ext in [".mp4", ".avi", ".mov"]:
                    # FIX: Use wandb.Video
                    processed_metrics[key] = wandb.Video(value, fps=fps, format="mp4")
                elif ext in [".gif", ".png", ".jpg", ".jpeg"]:
                    # FIX: Use wandb.Image (This caused your specific error)
                    processed_metrics[key] = wandb.Image(value)
                else:
                    processed_metrics[key] = value
            
            # 3. Scalars or other numeric values
            else:
                processed_metrics[key] = value

        self.wandb.log(processed_metrics, step=step)

    def log_frames(self, frames, key="video", step=None, fps=100, format="mp4"):
        """
        Logs video to WandB.

        Final logged shape is always: T×C×H×W

        Accepts:
            - list of H×W×C or C×H×W frames
            - numpy T×H×W×C
            - numpy T×C×H×W
            - single frame in either format (expanded to T=1)
        """

        if self.wandb is None or frames is None:
            return

        # --------------------------------------------------------
        # 1. Convert input to numpy array
        # --------------------------------------------------------
        if isinstance(frames, list):
            if len(frames) == 0:
                return

            processed = []

            for f in frames:
                f = np.asarray(f)

                # CHW → HWC
                if f.ndim == 3 and f.shape[0] in [1, 3]:
                    f = np.transpose(f, (1, 2, 0))

                processed.append(f)

            video = np.stack(processed, axis=0)  # Now T×H×W×C

        else:
            video = np.asarray(frames)

            # Single frame: expand T
            if video.ndim == 3:
                # CHW → HWC
                if video.shape[0] in [1, 3]:
                    video = np.transpose(video, (1, 2, 0))
                video = video[None]  # T=1

        # --------------------------------------------------------
        # 2. Validate now (should be T×H×W×C)
        # --------------------------------------------------------
        if video.ndim != 4:
            raise ValueError(
                f"log_frames(): Expected 4D input (T,H,W,C), got {video.shape}"
            )

        T, H, W, C = video.shape

        if C not in [1, 3]:
            raise ValueError(
                f"log_frames(): expected channel dim 1 or 3, got {C}"
            )

        # --------------------------------------------------------
        # 3. Convert to T×C×H×W
        # --------------------------------------------------------
        video = np.transpose(video, (0, 3, 1, 2))  # (T,H,W,C) → (T,C,H,W)

        # --------------------------------------------------------
        # 4. Float → uint8 normalization
        # --------------------------------------------------------
        if video.dtype != np.uint8:
            video = np.clip(video * 255, 0, 255).astype(np.uint8)

        # --------------------------------------------------------
        # 5. Log to WandB
        # --------------------------------------------------------
        self.wandb.log(
            {
                key: wandb.Video(video, fps=fps, format=format)
            },
            step=step
        )

    def write(self, fps=False, step=False):
        if not step:
            step = self.update_step

        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))

        # Print summary
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))

        # Write json metrics file
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")

        # Log scalars
        wandb_logs = {k: v for k, v in scalars}

        # Log images
        for name, img in self._images.items():
            wandb_logs[name] = wandb.Image(img)

        # Log videos
        for name, vid in self._videos.items():
            value = vid
            # Convert floats to uint8 video
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)

            # wandb accepts (T, H, W, C)
            if value.ndim == 5:  # B T H W C — take first batch
                value = value[0]

            value = np.transpose(value, (0, 3, 1, 2))  # (T,H,W,C) → (T,C,H,W)

            wandb_logs[name] = wandb.Video(value, fps=16, format="mp4")

        # Commit step
        wandb.log(wandb_logs, step=step)

        # Clear buffers
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)