import os
import typing
from typing import Optional
from multipledispatch import dispatch
import logging


from dotmap import DotMap
import ruamel.yaml as yaml
from tqdm import tqdm
from pathlib import Path

from agent_arena.agent.oracle.builder import OracleBuilder
from agent_arena.arena.builder import ArenaBuilder
from agent_arena.agent.register import AGENT_NEEDS_CONFIG, AGENT_NO_CONFIG
from agent_arena.utilities.transform.register import DATA_TRANSFORMER
from agent_arena.utilities.transform.transform import Transform
from agent_arena.registration.logger import LOGGER
from agent_arena.utilities.perform_single import perform_single
from agent_arena import TrainableAgent, Agent, Arena
from agent_arena.utilities.logger.logger_interface import Logger
from agent_arena.utilities.verbose import Verbose

# Create Enum for Verbose

def load_yamls(file_path: str) -> typing.Dict[str, typing.Any]:
    return yaml.safe_load(Path(file_path).read_text())

@dispatch(str)
def retrieve_config(config_path: str) -> typing.Dict[str, typing.Any]:
    configs = load_yamls(config_path)

    return DotMap(configs)

@dispatch(str, str, str)
def retrieve_config(agent_name: str, arena_name:str, 
        config_name:str, config_dir: Optional[str]=None,
        ) -> typing.Dict[str, typing.Any]:
    
    config = DotMap()
    
    config_path = '{}/{}/{}.yaml'.format(agent_name, arena_name, config_name)
    if config_dir == None: ## If None load from agent-arena trajecotry
        config_path = os.path.join(os.environ['AGENT_ARENA_PATH'], 'configuration', 'train_and_evaluate', config_path)
        if agent_name in AGENT_NEEDS_CONFIG.keys():
            config = retrieve_config(config_path)
            config.oracle = False
        elif agent_name in AGENT_NO_CONFIG.keys():
            pass
        else:
            
            config.oracle = True
    else: ## load from customised trajectory
        config_path = os.path.join(config_dir, config_path)
        config = retrieve_config(config_path)

    
    
    #config.save_dir = os.path.join(log_dir, arena_name, agent_name, config_name)
    return config


def build_transform(name: str, params: DotMap) -> Transform:
    return DATA_TRANSFORMER[name](params)

def build_arena(name: str, ray=False) -> Arena:
    return ArenaBuilder.build(name, ray=ray)


# find ways to get read of arena.
def build_agent(
        name: str, 
        config: Optional[DotMap] = None) -> Agent:
    
    if config is not None and config.oracle:
        return OracleBuilder.build(name)
    
    # if arena is not None:
    #     config.action_space = arena.get_action_space()
    if name in AGENT_NEEDS_CONFIG.keys():
        return AGENT_NEEDS_CONFIG[name](config)
    else:
        return AGENT_NO_CONFIG[name](config)

def build_logger(name: str, save_dir: str) -> Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = LOGGER[name](save_dir)
    return logger

def run(agent: Agent, arena: Arena, mode:str, 
        episode_config: dict, checkpoint: int) -> bool:
    
    filename = 'manupilation_{}'.format(checkpoint)
    print('episode_config', episode_config)
    
    if mode == 'eval' and arena.logger.check_exist(episode_config, filename):
        return
   
    res = perform_single(arena, agent, mode=mode, episode_config=episode_config,
                collect_frames=episode_config['save_video'])
    
    if mode == 'eval':
        agent.logger(episode_config, res, filename)
        arena.logger(episode_config, res, filename)

    return True

def evaluate(agent: Agent, arena: Arena, checkpoint: int) -> bool:

    #arena.set_eval()
    
    logging.info('[ag_ar.evaluate] Start evaluating Agent "{}" on\n     Arena "{}"'.\
            format(agent.get_name(), arena.get_name()))
    
    env_eval_configs = arena.get_eval_configs()
    #print('checkpoint', checkpoint)
    if isinstance(agent, TrainableAgent):
        if checkpoint >= 0:
            print('load_checkpoint', checkpoint)
            agent.load_checkpoint(checkpoint)
        else:
            checkpoint = agent.load()

    for episode_config in tqdm(env_eval_configs):
        #print('checkpoint', checkpoint)
        run(agent, arena, 'eval', episode_config, checkpoint=checkpoint)
    
    return True


def validate(agent, arena, update_step):
    '''
        Validate the agent's current performance on the slected validation intial configuration of the arena.

        This method requires:
            * The agent has `get_writer` method to log the validation results.
            * The arena has `set_val` and `get_val_configs` methods to set the 
              arena to validation mode and get the validation configurations.
    '''

    val_configs = arena.get_val_configs() 
                
    for episode_config in tqdm(val_configs):
        run(agent, arena, 'val', episode_config, checkpoint=update_step)
    
def train_and_evaluate(agent: TrainableAgent, arena: Arena,
                       validation_interval: int, total_update_steps: int, eval_checkpoint: int) -> bool:
    '''
        Train the agent on the selected arena and evaluate the agent's performance on the selected 
        evaluation configurations of the arena.

        This method requires:
            * The agent has `train` method with `arena` and `update_steps` as arguments to train the agent,
            * The agent has `load` method to load the agent's model.
            * The agent has `load_checkpoint` method to load the agent's model from a checkpoint.
            * The agent has `get_name` method to get the agent's name.
            * The agent has `get_writer` method to log the validation results.
            * The arena has `set_eval` and `get_eval_configs` methods to set the 
              arena to evaluation mode and get the evaluation configurations.
            * The arena has `set_val` and `get_val_configs` methods to set the 
              arena to validation mode and get the validation configurations.
    '''

    logging.info('\n[ag_ar.train_and_evaluate] Training Agent "{}"'.format(agent.get_name()))
    print('action horizon', arena.get_action_horizon())
    
    #validate(agent, arena, 0)

    if validation_interval > 0:
        assert total_update_steps > 0, 'Total update steps must be greater than 0'                    
        start_update_step = agent.load()

        if eval_checkpoint >= 0:
            total_update_steps = min(total_update_steps, eval_checkpoint)
    
        for u in range(start_update_step, total_update_steps-1, validation_interval):
            agent.train(validation_interval, arena)
            agent.save()
            validate(agent, arena, u)

    else:
        agent.train(arena)

    
    logging.info('\n[ag_ar.train_and_evaluate] Finished training Agent "{}"'.format(agent.get_name()))

    # if eval_checkpoint >= 0:
    #     logging.info('\n[ag_ar.train_and_evaluate] Load checkpoint {}'.format(eval_checkpoint))
    #     agent.load_checkpoint(eval_checkpoint)

    evaluate(agent, arena, checkpoint=eval_checkpoint)