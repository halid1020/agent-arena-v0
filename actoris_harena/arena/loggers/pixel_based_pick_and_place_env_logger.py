import os
import cv2
import numpy as np
from .video_logger import VideoLogger
import matplotlib.pyplot as plt
from .draw_utils import *


class PixelBasedPickAndPlaceEnvLogger(VideoLogger):

    def __call__(self, episode_config, result, filename=None, wandb_logger=None):
        super().__call__(episode_config, result, filename=filename, wandb_logger=None)

        eid = episode_config["eid"]
        frames = [info["observation"]["rgb"] for info in result["information"]]
        #actions = result["actions"]

        H, W = 512, 512

        if filename is None:
            filename = "manipulation"

        out_dir = os.path.join(self.log_dir, filename, "performance_visualisation")
        os.makedirs(out_dir, exist_ok=True)

        

        images = []

        for i in range(len(frames)-1):
            img = frames[i].copy()
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

            # -------------------------------
            # Step text
            # -------------------------------
            step_text = f"Step {i + 1}: Pick and Place"
            primitive_color = PRIMITIVE_COLORS.get("norm-pixel-pick-and-place", PRIMITIVE_COLORS["default"])
            draw_text_with_bg(
                img,
                step_text,
                (10, TEXT_Y_STEP),
                primitive_color
            )
            # -------------------------------
            # Extract action
            # -------------------------------
            # act must contain exactly:
            # pick_0, pick_1, place_0, place_1
            #print('results keys', result.keys())
            # Inside your Logger's __call__ method loop:
            applied_action = result["information"][i+1]['applied_action']

            # Extract values assuming [pick_y, pick_x, place_y, place_x, theta]
            # Convert normalized to pixel first
            px, py = norm_to_px(applied_action[:2], W, H)
            dx, dy = norm_to_px(applied_action[2:4], W, H)
            pick_theta = None
            if len(applied_action) == 5:
                place_theta = applied_action[4] # The orientation
            elif len(applied_action) == 6:
                pick_theta = applied_action[4]
                place_theta = applied_action[5] # The orientation

            # We swap them because your norm_to_px returns (x,y) 
            # and your drawing functions expect (y,x) to then 'swap' them back.
            img = draw_pick_and_place_with_orient(
                img, 
                (px, py), # Sending as (y, x) to satisfy your 'swap' function
                (dx, dy), 
                pick_orient=pick_theta, 
                place_orient=place_theta
            )
            

            images.append(img)

        # Add final frame (no action)
        images.append(cv2.resize(frames[-1], (W, H)))

        # ===========================================
        #  Matplotlib grid visualization
        # ===========================================
        MAX_COLS = 6
        num_images = len(images)
        num_rows = (num_images + MAX_COLS - 1) // MAX_COLS

        fig, axes = plt.subplots(
            num_rows, MAX_COLS,
            figsize=(3 * MAX_COLS, 3 * num_rows),
        )

        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        idx = 0
        for r in range(num_rows):
            for c in range(MAX_COLS):
                ax = axes[r][c]
                if idx < num_images:
                    img_rgb = images[idx] #cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                ax.axis("off")
                idx += 1

        plt.tight_layout()

        save_path = os.path.join(out_dir, f"episode_{eid}_trajectory.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        if wandb_logger is not None:
            wandb_logger.log(
                {f"trajectory/episode_{eid}": save_path})
