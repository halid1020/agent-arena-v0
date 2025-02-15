import argparse
import api as ag_ar
from utilities.utils import create_message_logger
from agent_arena import TrainableAgent

def main():

    ### Argument Definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', 
        default="softgym|domain:mono-square-fabric,initial:crumple,"\
            "action:pixel-pick-and-place(1),task:flattening")
    parser.add_argument('--agent', 
        default="oracle-rect-fabric|action:pixel-pick-and-place(1),"
        "strategy:expert,task:flattening")
    # parser.add_argument('--logger', 
    #     default="standard_logger")
    parser.add_argument('--log_dir', default='tmp')
    parser.add_argument('--config',  default='default')
    
    parser.add_argument('--eid', default=1, type=int)
    parser.add_argument('--disp', default=0, type=int)
    parser.add_argument('--eval', default=1, type=int)
    parser.add_argument('--save_video', default=0, type=int)
    parser.add_argument('--verbose', default='silence', type=str)
    
    ### TODO: put eval
    
    args = parser.parse_args()

    if args.disp == 1:
        disp = True
    elif args.disp == 0:
        disp = False
    else:
        raise NotImplementedError
    
    if args.eval == 1:
        eval = True
    elif args.disp == 0:
        eval = False
    else:
        raise NotImplementedError
    
    
    config = ag_ar.retrieve_config(
        args.agent, 
        args.arena, 
        args.config, 
        args.log_dir)

    
    create_message_logger(config.save_dir, args.verbose)
    

    ### Initialise arena
    arena = ag_ar.build_arena(args.arena + ',disp:{}'.format(disp))

    ### Initialise Policy
    agent = ag_ar.build_agent(args.agent, config=config)

    ### Initialise Logger
    logger = ag_ar.build_logger(arena.logger_name, config.save_dir)
    
    if eval:
        arena.set_eval()
        if isinstance(agent, TrainableAgent):
            agent.load()
            agent.set_eval()
           
    else:
        arena.set_train()

    
    ag_ar.run(agent, 
        arena, 
        'eval' if eval else 'train',
        episode_config={
            'eid': args.eid, 
            'tier': 0,
            'save_video': (True if args.save_video == 1 else False)
        },
        logger=logger, checkpoint=-1)
    


if __name__ == '__main__':
    main()
