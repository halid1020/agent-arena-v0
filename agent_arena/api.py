import os
import typing
from typing import Optional, List
from multipledispatch import dispatch
import logging
import json
import shutil

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
from agent_arena.utilities.visual_utils import save_video, save_numpy_as_gif

# Create Enum for Verbose

def load_yamls(file_path: str) -> typing.Dict[str, typing.Any]:
    return yaml.safe_load(Path(file_path).read_text())

def retrieve_config_from_path(config_path: str) -> typing.Dict[str, typing.Any]:
    configs = load_yamls(config_path)
    return DotMap(configs)

def retrieve_config(agent_name: str, arena_name:str, 
        config_name:str, config_dir: Optional[str]=None,
        ) -> typing.Dict[str, typing.Any]:
    
    config = DotMap()
    
    config_path = '{}/{}/{}.yaml'.format(agent_name, arena_name, config_name)
    if config_dir == None: ## If None load from agent-arena trajecotry
        config_path = os.path.join(os.environ['AGENT_ARENA_PATH'], 'configuration', 'train_and_evaluate', config_path)
        if agent_name in AGENT_NEEDS_CONFIG.keys():
            config = retrieve_config_from_path(config_path)
            config.oracle = False
        elif agent_name in AGENT_NO_CONFIG.keys():
            pass
        else:
            
            config.oracle = True
    else: ## load from customised trajectory
        config_path = os.path.join(config_dir, config_path)
        config = retrieve_config_from_path(config_path)

    
    
    #config.save_dir = os.path.join(log_dir, arena_name, agent_name, config_name)
    return config

def register_agent(name: str, class_):
    AGENT_NEEDS_CONFIG[name] = class_

def build_transform(name: str, params: DotMap) -> Transform:
    return DATA_TRANSFORMER[name](params)

def build_arena(name: str, ray=False) -> Arena:
    return ArenaBuilder.build(name, ray=ray)


# find ways to get read of arena.
def build_agent(
        name: str, 
        config: Optional[DotMap] = None) -> Agent:
    
    if config is not None and config.get("oracle", False):
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

def save_best_results(results, save_dir, checkpoint):
    """
    Save the best results (a list of dicts) to a JSON file.

    Args:
        results (list[dict]): List of result dictionaries to save.
        save_dir (str): Directory where to save the results.
    """

    save_dir = save_dir or os.getcwd()
    best_path = os.path.join(save_dir, 'best')
    os.makedirs(best_path, exist_ok=True)

    json_file = os.path.join(best_path, "best_results.json")
    print('Saving best results to', best_path)

    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)

    folder_name_to_copy = f'val_checkpoint_{checkpoint}'
    src_path = os.path.join(save_dir, folder_name_to_copy)
    dst_path = best_path

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Checkpoint folder not found: {src_path}")

    # Copy contents safely for Python < 3.8 (no dirs_exist_ok)
    for item in os.listdir(src_path):
        s = os.path.join(src_path, item)
        d = os.path.join(dst_path, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

def load_best_results(save_dir):
    """
    Load the best results from a JSON file.

    Args:
        save_dir (str): Directory where the best_results.json is stored.
                        Defaults to current working directory.

    Returns:
        list[dict]: Loaded list of result dictionaries, or empty list if none found.
    """
    save_dir = save_dir or os.getcwd()
    load_path = os.path.join(save_dir, 'best', "best_results.json")

    if not os.path.exists(load_path):
        return []

    with open(load_path, "r") as f:
        results = json.load(f)
    print(f"Loaded best results from {load_path}")
    return results


def run(agent: Agent, arena: Arena, mode:str, 
        episode_config: dict, checkpoint: int):
    
    
    print(f'run {mode} episode_config', episode_config)
    
    eval_filename = 'eval_checkpoint_{}'.format(checkpoint)
    if mode == 'eval' and arena.logger.check_exist(episode_config, eval_filename):
        return
   
    res = perform_single(arena, agent, mode=mode, episode_config=episode_config,
                collect_frames=episode_config['save_video'])
    
    if mode == 'eval':
        filename = 'eval_checkpoint_{}'.format(checkpoint)
        agent.logger(episode_config, res, filename)
        arena.logger(episode_config, res, filename)
    
    if mode == 'val':
        filename = 'val_checkpoint_{}'.format(checkpoint)
        agent.logger(episode_config, res, filename)
        arena.logger(episode_config, res, filename)

    return True, res

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
        Validate the agent's current performance on the selected validation initial configuration of the arena.

        This method requires:
            * The agent has `get_writer` method to log the validation results.
            * The arena has `set_val` and `get_val_configs` methods to set the 
              arena to validation mode and get the validation configurations.
    '''
    val_configs = arena.get_val_configs()
    print('val configs len', len(val_configs))
    results = []          
    for episode_config in tqdm(val_configs, desc="Validating controller in the arena..."):
        _, res = run(agent, arena, 'val', episode_config, checkpoint=update_step)
        results.append(res['evaluation'])
    return results


def compare_results(results_1, results_2, compare_func):
    return compare_func(results_1, results_2)
    
def train_and_evaluate_single(agent: TrainableAgent, arena: Arena,
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
        start_update_step = agent.load() #If no checkpint, it will return 0 --> no training

        if eval_checkpoint >= 0:
            total_update_steps = min(total_update_steps, eval_checkpoint)

        #print('total_update_steps', total_update_steps)
        for u in range(start_update_step, total_update_steps, validation_interval):
            #print('u', u)
            agent.train(validation_interval, [arena])
            agent.save()
            
            results = validate(agent, arena, u + validation_interval)
            #print('results', results)
            best_results = load_best_results(agent.save_dir)
            if len(best_results) == 0 or compare_results(results, best_results, arena.compare) > 0:
                agent.save_best()
                save_best_results(results, agent.save_dir,  u + validation_interval)

    else:
        agent.train([arena])

    
    logging.info('\n[ag_ar.train_and_evaluate] Finished training Agent "{}"'.format(agent.get_name()))

    evaluate(agent, arena, checkpoint=eval_checkpoint)


def train_plural_eval_single(
        agent: TrainableAgent, train_arenas: List[Arena], eval_arena: Arena, val_arena: Arena,
        validation_interval: int, total_update_steps: int, eval_checkpoint: int) -> bool:
    '''
        Train a single agent on the list of arenas and evaluate the agent's performance on the single selected 
        evaluation configurations of the arena.
    '''

    if validation_interval > 0:
        assert total_update_steps > 0, 'Total update steps must be greater than 0'                    
        start_update_step = agent.load() #If no checkpint, it will return 0 --> no training

        if eval_checkpoint >= 0:
            total_update_steps = min(total_update_steps, eval_checkpoint)

        #print('total_update_steps', total_update_steps)
        for u in range(start_update_step, int(total_update_steps), validation_interval):
            #print('u', u)
            agent.train(validation_interval, train_arenas)
            agent.save()
            
            results = validate(agent, val_arena, u + validation_interval)
            #print('results', results)
            best_results = load_best_results(agent.save_dir)
            if len(best_results) == 0 or compare_results(results, best_results, val_arena.compare) > 0:
                agent.save_best()
                save_best_results(results, agent.save_dir,  u + validation_interval)

    else:
        agent.train(train_arenas)

    evaluate(agent, eval_arena, checkpoint=eval_checkpoint)