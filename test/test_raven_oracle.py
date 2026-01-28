import argparse
from dotmap import DotMap
import agent_arena as ag_ar
import math
import os
import matplotlib
import numpy as np # Added for shape checking

# Force non-interactive backend
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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
    arena = ag_ar.build_arena(f"{args.arena},disp:{disp}", ray=False)
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
        total_steps = len(infos)
        
        # Subsample if necessary
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
                image_data = infos[idx]['color']
                
                # --- FIX STARTS HERE ---
                # Check if image_data has 4 dimensions (e.g., 3 cameras, H, W, C)
                # If so, pick the first image [0]
                image_data = np.array(image_data) # Ensure it's a numpy array
                if image_data.ndim == 4:
                    image_data = image_data[0]
                # --- FIX ENDS HERE ---

                ax.imshow(image_data)
                ax.set_title(f'Step {idx}')
            except KeyError:
                print(f"Warning: 'color' key missing at step {idx}")
                ax.text(0.5, 0.5, 'No Data', ha='center')
            
            ax.axis('off')

        # Clean up unused axes
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