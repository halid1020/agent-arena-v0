import argparse
from dotmap import DotMap
import actoris_harena as athar
import math
import os
import matplotlib
import numpy as np

# Force non-interactive backend for server environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def format_action_text(action):
    """Formats the action for PushT (typically [x, y] or [x, y, orientation])."""
    if isinstance(action, (np.ndarray, list)):
        action = np.array(action).flatten()
        if len(action) >= 2:
            return f"Action Pos: ({action[0]:.2f}, {action[1]:.2f})"
    return "Invalid Action"

def main():
    parser = argparse.ArgumentParser()
    # Keeping your original default but allowing overrides
    parser.add_argument('--arena', default='openAI-gym', help="Arena name")
    parser.add_argument('--eid', default=0, type=int, help="Episode ID")
    parser.add_argument('--disp', default=0, type=int, help="Display mode (0 or 1)")
    parser.add_argument('--policy', default='random', help="Policy type")
    
    args = parser.parse_args()
    
    disp = args.disp == 1
    log_dir = f'./tmp/test_pushT_{args.policy}'
    project_name = 'test_pushT'
    exp_name = f'run_{args.policy}_eid_{args.eid}'

    print(f'\n--- Configuration ---')
    print(f'Arena:      {args.arena}')
    print(f'Policy:     {args.policy}')
    print(f'Episode ID: {args.eid}')
    print(f'Log Dir:    {log_dir}')
    print(f'---------------------')

    # 1. Build Arena using the DotMap format
    print(f"\nBuilding Arena: {args.arena}...")
    arena = athar.build_arena(
        f"{args.arena}", 
        DotMap({
            'ray': False,
            'domain': 'pushT',
            'use_default_goal_cam': True,
            'disp': disp
            # Add any specific pushT domain params here if needed
        }),
        save_dir=log_dir,
        project_name=project_name,
        exp_name=exp_name
    )
    arena.set_eval()

    # 2. Build Agent
    print(f"Building Agent: {args.policy}...")
    agent = athar.build_agent(
        args.policy,
        DotMap({}), 
        log_dir, 
        project_name=project_name,
        exp_name=exp_name
    )

    # 3. Run Episode
    print(f"Running episode {args.eid}...")
    _, res = athar.run(agent, arena, 'eval',
        episode_config={
            'eid': args.eid, 
            'save_video': True, 
        },
        env_success_stop=False,
        policy_terminate=False,
        checkpoint=-1)
    
if __name__ == '__main__':
    main()