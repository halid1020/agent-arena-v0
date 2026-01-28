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
    """Formats the action dictionary into a concise string."""
    if isinstance(action, dict):
        # Ravens actions usually have 'pose0' (pick) and 'pose1' (place)
        # We'll extract just the position (x, y) for brevity
        text_parts = []
        if 'pose0' in action:
            p0 = action['pose0'][0]
            text_parts.append(f"P0:({p0[0]:.2f},{p0[1]:.2f})")
        if 'pose1' in action:
            p1 = action['pose1'][0]
            text_parts.append(f"P1:({p1[0]:.2f},{p1[1]:.2f})")
        return "\n".join(text_parts) if text_parts else str(action)
    return str(action)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', default='raven|task:block-insertion')
    parser.add_argument('--eid', default=0, type=int)
    parser.add_argument('--disp', default=0, type=int)

    args = parser.parse_args()
    
    # Environment
    print()
    print('Initialising Environment {}'.format(args.arena))

    disp = args.disp == 1
    
    # 1. Define the config string with top_down mode and 128 resolution
    arena_config = f"{args.arena},disp:{disp},view_mode:top_down,img_res:128"
    
    print(f"Building arena with config: {arena_config}") 
    arena = ag_ar.build_arena(arena_config, ray=False)
    
    arena.set_eval()

    # Initialise Expert Policy
    agent = ag_ar.build_agent(
        'raven-oracle', 
        DotMap({}))

    log_dir = './tmp/test_arena'
    arena.set_log_dir(log_dir, project_name='test_proj', exp_name='test_exp')
    agent.set_log_dir(log_dir, project_name='test_proj', exp_name='test_exp')

    _, res = ag_ar.run(agent, arena, 'eval',
        episode_config={
            'eid': args.eid, 
            'save_video': True, 
        },
        checkpoint=-1)
    
    ### Plotting Logic
    if 'information' in res and len(res['information']) > 0:
        infos = res['information']
        actions = res.get('actions', []) # Retrieve actions list
        total_steps = len(infos)
        
        MAX_PLOT_FRAMES = 60 
        step = max(1, math.ceil(total_steps / MAX_PLOT_FRAMES))
        
        plot_indices = range(0, total_steps, step)
        num_plots = len(plot_indices)

        cols = 6
        rows = math.ceil(num_plots / cols)
        
        print(f"Total steps: {total_steps}. Subsampling with step {step}.")
        print(f"Plotting {num_plots} images from 'color' field in a {rows}x{cols} grid...")

        fig, axes = plt.subplots(rows, cols, figsize=(20, 3.5 * rows))
        
        if rows * cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, idx in enumerate(plot_indices):
            ax = axes[i]
            try:
                # --- Image Handling ---
                image_data = infos[idx]['color']
                image_data = np.array(image_data) 
                
                if image_data.ndim == 4:
                    image_data = image_data[0]

                ax.imshow(image_data)
                ax.set_title(f'Step {idx}')

                # --- TODO RESOLVED: Action Text Overlay ---
                # Check if an action exists for this index
                if idx < len(actions):
                    action_str = format_action_text(actions[idx])
                    
                    # Place text at top-left (x=5, y=10) in image coordinates
                    # bbox creates the black background with white text
                    ax.text(5, 10, action_str, 
                            color='white', 
                            fontsize=8, 
                            verticalalignment='top',
                            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
                            
            except KeyError:
                print(f"Warning: 'color' key missing at step {idx}")
                ax.text(0.5, 0.5, 'No Data', ha='center')
            ax.axis('off')

        for i in range(num_plots, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f'episode_{args.eid}_color_summary.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        print("No information/steps found in results to plot.")

if __name__ == '__main__':
    main()