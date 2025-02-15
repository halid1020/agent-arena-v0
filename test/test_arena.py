import argparse

import agent_arena as ag_ar

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', default='openAI-gym|domain:pushT')
    parser.add_argument('--eid', default=0, type=int)
    parser.add_argument('--disp', default=0, type=int)

    args = parser.parse_args()
    
    print('args', args)

    # Environment
    print()
    print('Initialising Environment {}'.format(args.arena))

    disp = args.disp == 1  # Convert disp to boolean directly

    arena = ag_ar.build_arena(f"{args.arena},disp:{disp}", ray=False)

    arena.set_eval()

    # Initialise Expert Policy
    action_space = arena.get_action_space()
    print('action space', action_space)
    agent = ag_ar.build_agent('random')

    log_dir = './tmp/test_arena'
    arena.set_log_dir(log_dir)
    agent.set_log_dir(log_dir)

    ag_ar.run(agent, arena, 'eval',
        episode_config={
            'eid': args.eid, 
            'save_video': True,
        },
        checkpoint=-1)

if __name__ == '__main__':
    main()