import argparse
import os

import agent_arena.api as ag_ar
from agent_arena.utilities.utils import create_message_logger

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', 
        default="softgym|domain:mono-square-fabric,initial:crumple,"\
            "action:pixel-pick-and-place(1),task:flattening")
    parser.add_argument('--agent', 
        default="oracle-rect-fabric|action:pixel-pick-and-place(1),"
                "strategy:expert,task:flattening")
    parser.add_argument('--config',  default='default')
    parser.add_argument('--log_dir', default='tmp')
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval_checkpoint', default='-1', type=int)

    args = parser.parse_args()

    if args.disp == 1:
        disp = True
    elif args.disp == 0:
        disp = False

    config = ag_ar.retrieve_config(
        args.agent, 
        args.arena, 
        args.config)

    save_dir = os.path.join(args.log_dir, args.arena, args.agent, args.config)
    
   
    #create_message_logger(save_dir, args.verbose)

    arena = ag_ar.build_arena(args.arena + ',disp:{}'.format(disp))
    agent = ag_ar.build_agent(args.agent, config)
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)
    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)

    ag_ar.evaluate(agent, arena, args.eval_checkpoint)
    
if __name__ == '__main__':
    main()