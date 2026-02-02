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
    """Formats the action (vectorized) for display based on dimensions."""
    if isinstance(action, (np.ndarray, list)):
        # --- CASE 1: World Action (14-dim) ---
        # [Pick Pos (3), Pick Rot (4), Place Pos (3), Place Rot (4)]
        if len(action) == 14:
            p0 = action[0:3]  # Pick XYZ
            p1 = action[7:10] # Place XYZ
            return f"World Action\nPick:({p0[0]:.2f}, {p0[1]:.2f})\nPlace:({p1[0]:.2f}, {p1[1]:.2f})"
        
        # --- CASE 2: Pixel Action (5-dim) ---
        # [Pick U, Pick V, Place U, Place V, Theta] (Normalized -1 to 1)
        elif len(action) == 5:
            pick_uv = action[0:2]
            place_uv = action[2:4]
            theta = action[4]
            return f"Pixel Action (Norm)\nPick:({pick_uv[0]:.2f}, {pick_uv[1]:.2f})\nPlace:({place_uv[0]:.2f}, {place_uv[1]:.2f})\nTheta:{theta:.2f}"

    return "Invalid Action"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', default='raven', help="Base arena name")
    parser.add_argument('--task', default='stack-block-pyramid', help="Task name")
    parser.add_argument('--eid', default=0, type=int, help="Episode ID")
    parser.add_argument('--disp', default=0, type=int, help="Display/GUI mode (0 or 1)")
    
    # --- NEW: Switch Arguments ---
    parser.add_argument('--space', default='world', choices=['world', 'pixel'], 
                        help="Action space type: 'world' (14-dim) or 'pixel' (5-dim)")
    parser.add_argument('--policy', default='random', choices=['random', 'oracle'], 
                        help="Policy type: 'random' or 'oracle'")
    
    args = parser.parse_args()
    
    # 1. Determine Arena and Agent Names based on arguments
    if args.space == 'pixel':
        # Assumes you have registered 'raven-pixel' -> RavenPixelEnvAdapter
        arena_key = 'raven-pixel' 
        
        if args.policy == 'random':
            agent_key = 'raven-pixel-random'
        else:
            agent_key = 'raven-pixel-oracle'
    else:
        # Standard World Space
        arena_key = 'raven'
        
        if args.policy == 'random':
            agent_key = 'raven-random'
        else:
            # Assumes 'raven-oracle' maps to the standard RavenOraclePolicyAdapter
            agent_key = 'raven-oracle' 

    print(f'\n--- Configuration ---')
    print(f'Arena Type:   {args.space.upper()} ({arena_key})')
    print(f'Policy Type:  {args.policy.upper()} ({agent_key})')
    print(f'Task:         {args.task}')
    print(f'Episode ID:   {args.eid}')
    print(f'---------------------')

    disp = args.disp == 1
    log_dir = f'./tmp/test_{args.space}_{args.policy}'

    # 2. Build Arena
    # Note: 'action_horizon' determines how many steps the episode runs
    print(f"\nBuilding Arena: {arena_key}...")
    arena = ag_ar.build_arena(
        arena_key,
        DotMap({
            'disp': disp,
            'task': args.task,
            'view_mode': 'top_down',
            'img_res': 128,
            'action_horizon': 5,
            # Pass debug config if using pixel arena to save internal debug plots
            'debug': True, 
            'debug_dir': os.path.join(log_dir, 'internal_debug')
        }),
        save_dir=log_dir,
        project_name=f'test_raven_{args.space}',
        exp_name=f'test_{args.policy}'
    )
    arena.set_eval()

    # 3. Build Agent
    print(f"Building Agent: {agent_key}...")
    agent = ag_ar.build_agent(
        agent_key,
        DotMap({}), 
        log_dir, 
        project_name=f'test_raven_{args.space}',
        exp_name=f'test_{args.policy}'
    )

    # 4. Run Episode
    print(f"Running episode {args.eid}...")
    _, res = ag_ar.run(agent, arena, 'eval',
        episode_config={
            'eid': args.eid, 
            'save_video': True, 
        },
        checkpoint=-1)
    
    # 5. Plotting Logic
    if 'information' in res and len(res['information']) > 0:
        infos = res['information']
        actions = res.get('actions', [])
        total_steps = len(infos)
        
        # Determine subsampling
        MAX_COLS = 10 
        step = max(1, math.ceil(total_steps / MAX_COLS))
        plot_indices = range(0, total_steps, step)
        num_cols = len(plot_indices)
        
        print(f"Total steps: {total_steps}. Subsampling with step {step}.")
        print(f"Plotting {num_cols} columns (Color, Depth, Mask)...")

        fig, axes = plt.subplots(3, num_cols, figsize=(3 * num_cols, 10))
        
        if num_cols == 1:
            axes = axes.reshape(3, 1)

        for col_idx, step_idx in enumerate(plot_indices):
            info = infos[step_idx]
            obs = info.get('observation', {}) 

            # --- Row 1: RGB Color ---
            ax_rgb = axes[0, col_idx]
            
            # Helper to extract RGB safely
            rgb = None
            if 'rgb' in obs:
                rgb = np.array(obs['rgb'])
            elif 'color' in obs:
                rgb = np.array(obs['color'])
                if rgb.ndim == 4: rgb = rgb[0]

            if rgb is not None:
                ax_rgb.imshow(rgb)
                ax_rgb.set_title(f'Step {step_idx} (RGB)')
                
                # Overlay Action Text
                if step_idx < len(actions):
                    action_str = format_action_text(actions[step_idx])
                    ax_rgb.text(5, 10, action_str, color='white', fontsize=7, 
                                verticalalignment='top', fontfamily='monospace',
                                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
            else:
                ax_rgb.text(0.5, 0.5, 'No RGB', ha='center')

            # --- Row 2: Depth ---
            ax_depth = axes[1, col_idx]
            if 'depth' in obs:
                depth = np.array(obs['depth'])
                if depth.ndim == 3: depth = depth[0]
                ax_depth.imshow(depth, cmap='plasma')
                ax_depth.set_title(f'Depth')
            else:
                ax_depth.text(0.5, 0.5, 'No Depth', ha='center')

            # --- Row 3: Segmentation Mask ---
            ax_segm = axes[2, col_idx]
            # Use 'mask' (binary) or 'segm' (ids) depending on what's available
            # Prioritize 'mask' as it's cleaner for visualization usually
            if 'mask' in obs:
                segm = np.array(obs['mask'])
                if segm.ndim == 3: segm = segm[0]
                ax_segm.imshow(segm, cmap='gray')
                ax_segm.set_title(f'Binary Mask')
            elif 'segm' in obs:
                segm = np.array(obs['segm'])
                if segm.ndim == 3: segm = segm[0]
                ax_segm.imshow(segm, cmap='tab20', interpolation='nearest')
                ax_segm.set_title(f'ID Mask')
            else:
                ax_segm.text(0.5, 0.5, 'No Mask', ha='center')

            ax_rgb.axis('off')
            ax_depth.axis('off')
            ax_segm.axis('off')

        plt.tight_layout()
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f'episode_{args.eid}_summary.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        print("No information/steps found in results to plot.")

if __name__ == '__main__':
    main()