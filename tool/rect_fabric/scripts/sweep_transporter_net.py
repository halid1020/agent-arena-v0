import argparse

import api as ag_ar
from utilities.utils import create_message_logger

def main():

    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', 
        default="softgym|domain:mono-square-fabric,initial:crumple,"\
            "action:pixel-pick-and-place(1),task:flattening")
    parser.add_argument('--config',  default='default')
    parser.add_argument('--log_dir', default='tmp')
    parser.add_argument('--verbose', default='silence', type=str)
    parser.add_argument('--disp', default=0, type=int)

    args = parser.parse_args()

    agent_name = "transporter"

    if args.disp == 1:
        disp = True
    elif args.disp == 0:
        disp = False

    config = ag_ar.retrieve_config(
        agent_name, 
        args.arena, 
        args.config, 
        args.log_dir)
    
   
    create_message_logger(config.save_dir, args.verbose)

    arena = ag_ar.build_arena(args.arena + ',gui:{}'.format(disp))
    agent = ag_ar.build_agent(agent_name, arena, config)
    

    checkpoints = [1001, 5001, 10001, 15001, 20001, 25001, 30001, 35001]
    for checkpoint in checkpoints:
        agent.load_checkpoint(checkpoint)
        save_dir = config.save_dir + "/manupilation_checkpoint_{}".format(checkpoint)
        logger = ag_ar.build_logger(arena.logger_name, save_dir)
        ag_ar.evaluate(agent, arena, logger)
    
if __name__ == '__main__':
    main()