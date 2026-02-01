import argparse
from dotmap import DotMap
import agent_arena as ag_ar
import math
import os
import matplotlib
import numpy as np

# Force non-interactive backend
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def format_action_text(action):
    """Formats the vectorized action (14-dim array) for display."""
    # We now strictly expect a list or numpy array of length 14
    # Structure: [Pick Pos (3), Pick Rot (4), Place Pos (3), Place Rot (4)]
    if isinstance(action, (np.ndarray, list)):
        if len(action) >= 14:
            p0 = action[0:3]  # Pick XYZ
            p1 = action[7:10] # Place XYZ
            return f"Pick:({p0[0]:.2f},{p0[1]:.2f})\nPlace:({p1[0]:.2f},{p1[1]:.2f})"
        
    return "Invalid Action"

def main():
    parser = argparse.ArgumentParser()
    # Defaulting to stack-block-pyramid as per your previous snippet
    parser.add_argument('--arena', default='raven')
    parser.add_argument('--task', default='stack-block-pyramid') 
    parser.add_argument('--eid', default=0, type=int)
    parser.add_argument('--disp', default=0, type=int)
    args = parser.parse_args()
    
    print(f'\nInitialising Environment {args.arena} with task {args.task}')
    disp = args.disp == 1
    
    log_dir = './tmp/test_random'

    # 1. Build Arena with specific configuration
    # Note: 'action_horizon' determines how many steps the episode runs
    arena = ag_ar.build_arena(
        args.arena,
        DotMap({
            'disp': disp,
            'task': args.task,
            'view_mode': 'top_down',
            'img_res': 128,
            'action_horizon': 3
        }),
        save_dir=log_dir,
        project_name='test_raven_random',
        exp_name='test_raven_random'
    )
    arena.set_eval()

    # 2. Build the Random Agent
    # This agent produces the vectorized 14-dim action
    print("Building Random Agent...")
    agent = ag_ar.build_agent(
        'raven-random',
        DotMap({}), 
        log_dir, 
        project_name='test_raven_random',
        exp_name='test_raven_random')

    # 3. Run Episode
    # The arena adapter (RavenEnvAdapter) will receive the vector from the agent
    print(f"Running episode {args.eid}...")
    _, res = ag_ar.run(agent, arena, 'eval',
        episode_config={
            'eid': args.eid, 
            'save_video': True, 
        },
        checkpoint=-1)
    
    # 4. Plotting Logic
    if 'information' in res and len(res['information']) > 0:
        infos = res['information']
        actions = res.get('actions', [])
        total_steps = len(infos)
        
        # Determine subsampling to fit the image
        MAX_COLS = 10 
        step = max(1, math.ceil(total_steps / MAX_COLS))
        plot_indices = range(0, total_steps, step)
        num_cols = len(plot_indices)
        
        print(f"Total steps: {total_steps}. Subsampling with step {step}.")
        print(f"Plotting {num_cols} columns (Color, Depth, Mask)...")

        # Create Grid: 3 Rows x N Columns
        fig, axes = plt.subplots(3, num_cols, figsize=(3 * num_cols, 10))
        
        # Handle case where num_cols=1 (axes is 1D array)
        if num_cols == 1:
            axes = axes.reshape(3, 1)

        for col_idx, step_idx in enumerate(plot_indices):
            info = infos[step_idx]
            
            # --- FIX: Extract observation dictionary first ---
            obs = info.get('observation', {}) 

            # --- Row 1: RGB Color ---
            ax_rgb = axes[0, col_idx]
            # Check for 'rgb' (single frame) or 'color' (tuple of frames)
            if 'rgb' in obs:
                rgb = np.array(obs['rgb'])
                ax_rgb.imshow(rgb)
                ax_rgb.set_title(f'Step {step_idx} (RGB)')
                
                # Overlay Action Text
                if step_idx < len(actions):
                    action_str = format_action_text(actions[step_idx])
                    ax_rgb.text(5, 10, action_str, color='white', fontsize=8, 
                                verticalalignment='top',
                                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
            elif 'color' in obs: # Fallback if 'rgb' isn't there
                rgb = np.array(obs['color'])
                if rgb.ndim == 4: rgb = rgb[0]
                ax_rgb.imshow(rgb)
                ax_rgb.set_title(f'Step {step_idx} (RGB)')
            else:
                ax_rgb.text(0.5, 0.5, 'No RGB', ha='center')

            # --- Row 2: Depth ---
            ax_depth = axes[1, col_idx]
            if 'depth' in obs:
                depth = np.array(obs['depth'])
                # Handle tuple of cameras (Batch, H, W) -> (H, W)
                if depth.ndim == 3: depth = depth[0]
                
                ax_depth.imshow(depth, cmap='plasma')
                ax_depth.set_title(f'Depth')
            else:
                ax_depth.text(0.5, 0.5, 'No Depth', ha='center')

            # --- Row 3: Segmentation Mask ---
            ax_segm = axes[2, col_idx]
            if 'mask' in obs:
                segm = np.array(obs['mask'])
                # Handle tuple of cameras
                if segm.ndim == 3: segm = segm[0]
                
                ax_segm.imshow(segm, cmap='tab20', interpolation='nearest')
                ax_segm.set_title(f'Mask')
            else:
                ax_segm.text(0.5, 0.5, 'No Mask', ha='center')

            # Turn off axes ticks for all
            ax_rgb.axis('off')
            ax_depth.axis('off')
            ax_segm.axis('off')

        plt.tight_layout()
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f'episode_{args.eid}_random_summary.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        print("No information/steps found in results to plot.")

if __name__ == '__main__':
    main()