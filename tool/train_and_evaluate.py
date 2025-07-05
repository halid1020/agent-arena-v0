import sys
import argparse
import os

import agent_arena as ag_ar
from agent_arena.utilities.utils import create_message_logger

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', default='softgym|domain:mono-square-fabric,initial:crumpled,action:pixel-pick-and-place(1),task:flattening')
    parser.add_argument('--agent', default='planet-clothpick')
    parser.add_argument('--config', default='RGB2RGB')
    parser.add_argument('--log_dir', default='test_results')
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)

    return parser.parse_args()

def main():
    args = parse_arguments()

    disp = args.disp == 1  # Convert disp to boolean directly

    config = ag_ar.retrieve_config(args.agent, args.arena, args.config)
    # create_message_logger(config.save_dir, args.verbose)
    save_dir = os.path.join(args.log_dir, args.arena, args.agent, args.config)
    
   

    arena = ag_ar.build_arena(f"{args.arena},disp:{disp}")
    agent = ag_ar.build_agent(args.agent, config=config)
    
    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)
    #logger = ag_ar.build_logger(config.logger_name, config.save_dir)

    validation_interval = config.validation_interval
    total_update_steps = config.total_update_steps


    ag_ar.train_and_evaluate(agent, arena,
        validation_interval, total_update_steps, args.eval_checkpoint)

if __name__ == '__main__':
    main()