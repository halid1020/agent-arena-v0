import os
import typing
from typing import Optional, List
import json
import shutil
import numpy as np

from dotmap import DotMap
import ruamel.yaml as yaml
from tqdm import tqdm
from pathlib import Path

from agent_arena.agent.oracle.builder import OracleBuilder
from agent_arena.arena.builder import ArenaBuilder

from agent_arena.agent.registration import AGENTS
from agent_arena.arena.registration import ARENAS

from agent_arena.utilities.transform.register import DATA_TRANSFORMER
from agent_arena.utilities.transform.transform import Transform
from agent_arena.registration.logger import LOGGER
from agent_arena.utilities.perform_single import perform_single
from agent_arena import TrainableAgent, Agent, Arena
from agent_arena.utilities.logger.logger_interface import Logger


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
    AGENTS[name] = class_

def register_arena(name: str, class_):
    ARENAS[name] = class_

def build_transform(name: str, params: DotMap) -> Transform:
    return DATA_TRANSFORMER[name](params)

def build_arena(name: str, ray=False) -> Arena:
    return ArenaBuilder.build(name, ray=ray)

def build_arena(
        name: str,
        config: Optional[DotMap] = None,
        save_dir: Optional[str] = None,
        project_name: str = 'agent_arena',
        exp_name: str = 'tmp') -> Arena:
    """
    Constructs and configures an environment arena.

    This factory function initializes an arena instance based on the provided
    registry name, applies the given configuration, and sets up the directory
    structure for logging.

    Args:
        name: The registry key string identifying the arena class to instantiate.
            Must be a key present in the global `ARENAS` registry.
        config: A DotMap containing configuration parameters for the arena.
            Defaults to None.
        save_dir: The root directory path where logs and artifacts will be saved.
            If None, the logger may use a default temporary location or
            suppress file output depending on implementation. Defaults to None.
        project_name: The name of the project, used for organizing logs within
            the save directory. Defaults to 'agent_arena'.
        exp_name: The specific experiment name, creating a subdirectory under
            the project folder. Defaults to 'tmp'.

    Returns:
        Arena: An initialized instance of the requested Arena subclass, ready
            for interaction.

    Raises:
        KeyError: If `name` is not found in the `ARENAS` registry.
    """
    arena = ARENAS[name](config)
    arena.set_log_dir(save_dir, project_name, exp_name)
    return arena


def build_agent(
        name: str,
        config: Optional[DotMap] = None,
        save_dir: Optional[str] = None,
        project_name: str = 'agent_arena',
        exp_name: str = 'tmp') -> Agent:
    """
    Constructs and configures an agent instance.

    This factory function handles the instantiation of agents, including special
    handling for oracle agents and those requiring specific configuration injections.
    It also initializes the agent's logging system.

    Args:
        name: The registry key string identifying the agent class to instantiate.
            Must be present in either `AGENT_NEEDS_CONFIG` or `AGENT_NO_CONFIG`,
            or handled by `OracleBuilder`.
        config: A DotMap containing hyperparameters and initialization settings
            for the agent. Defaults to None.
        save_dir: The root directory path where the agent's logs and model
            checkpoints will be saved. Defaults to None.
        project_name: The name of the project, used for hierarchical logging.
            Defaults to 'agent_arena'.
        exp_name: The specific experiment name, used for hierarchical logging.
            Defaults to 'tmp'.

    Returns:
        Agent: An initialized instance of the requested Agent subclass.

    Raises:
        KeyError: If `name` is not found in the supported agent registries.
    """
    if config is not None and config.get("oracle", False):
        agent = OracleBuilder.build(name)
    
    # if name in AGENT_NEEDS_CONFIG.keys():
        
    # else:
    #     agent = AGENT_NO_CONFIG[name](config)

    agent = AGENTS[name](config)
    
    agent.set_log_dir(save_dir, project_name, exp_name)
    return agent


def build_logger(name: str, save_dir: str) -> Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = LOGGER[name](save_dir)
    return logger

def save_best_results(results, save_dir, checkpoint):
    """
    Save the best results (a list of dicts) and checkpoint id.

    Args:
        results (list[dict]): List of result dictionaries to save.
        save_dir (str): Directory where to save the results.
        checkpoint (int or str): Checkpoint identifier.
    """

    save_dir = save_dir or os.getcwd()
    best_path = os.path.join(save_dir, 'best')
    os.makedirs(best_path, exist_ok=True)

    # ---- save results ----
    json_file = os.path.join(best_path, "best_results.json")
    print('[agent-arena] Saving best results to', best_path)

    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)

    # ---- save checkpoint id ----
    checkpoint_file = os.path.join(best_path, "checkpoint.txt")
    with open(checkpoint_file, "w") as f:
        f.write(str(checkpoint) + "\n")

    # ---- copy checkpoint folder ----
    folder_name_to_copy = f'val_checkpoint_{checkpoint}'
    src_path = os.path.join(save_dir, folder_name_to_copy)
    dst_path = best_path

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Checkpoint folder not found: {src_path}")

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


def run(agent: Agent, arena: Arena, mode: str, 
        episode_config: dict, checkpoint: int,
        policy_terminate: bool=True, env_success_stop: bool=True):
    """
    Executes an episode (or check if it already exists) and logs the results.

    Args:
        agent (Agent): The agent instance to be evaluated or trained.
        arena (Arena): The environment instance where the agent interacts.
        mode (str): Execution mode. Options: 'train', 'eval', or 'val'.
        episode_config (dict): Configuration for the specific episode (e.g., 'eid', 'save_video').
        checkpoint (int): The checkpoint iteration number associated with the agent's weights.
        policy_terminate (bool, optional): If True, allows the agent to decide when to stop the episode. Defaults to True.
        env_success_stop (bool, optional): If True, the episode ends immediately upon environment success. Defaults to True.

    Returns:
        tuple: (bool, dict or None)
            - bool: True if the episode was run, False if it was skipped (e.g., log already exists).
            - dict: The result dictionary 'res' containing frames, actions, and evaluation metrics. Returns None if skipped.
    """
    
    print(f'[agent-arena, run] Run mode {mode} on episode_config', episode_config)
    
    eval_filename = 'eval_checkpoint_{}'.format(checkpoint)
    
    
    print(f'[agent-arena, run] Run mode {mode} on episode_config', episode_config)
    
    eval_filename = 'eval_checkpoint_{}'.format(checkpoint)
    if mode == 'eval' and arena.logger.check_exist(episode_config, eval_filename):
        return False, None
   
    res = perform_single(arena, agent, mode=mode, episode_config=episode_config,
                collect_frames=episode_config['save_video'], save_info=True,
                policy_terminate=policy_terminate, env_success_stop=env_success_stop)
    
    if mode == 'eval':
        filename = 'eval_checkpoint_{}'.format(checkpoint)
        agent.logger(episode_config, res, filename)
        arena.logger(episode_config, res, filename)
    
    if mode == 'val':
        filename = 'val_checkpoint_{}'.format(checkpoint)
        agent.logger(episode_config, res, filename)
        arena.logger(episode_config, res, filename, agent.logger)

    return True, res

def evaluate(agent: Agent, arena: Arena, checkpoint: int, 
             policy_terminate: bool = True,
             env_success_stop: bool = True) -> bool:
    """
    Evaluates a specific checkpoint of an agent within a given arena.

    This function loads the specified model checkpoint into the agent (if the agent 
    is trainable) and iterates through the arena's evaluation configurations to 
    assess performance.

    Args:
        agent (Agent): The agent instance to be evaluated.
        arena (Arena): The environment or arena instance where the evaluation 
            takes place.
        checkpoint (int): The specific model checkpoint to load. 
            Accepted special values:
            * -1: Load the most recent checkpoint.
            * -2: Load the best-performing checkpoint (if supported).
            * >= 0: Load the checkpoint corresponding to this specific step.
        policy_terminate (bool, optional): If True, allows the policy/agent to 
            decide when to terminate an episode (e.g., stopping signal). 
            Defaults to True.
        env_success_stop (bool, optional): If True, the environment will 
            terminate the episode immediately upon satisfying the success condition. 
            Defaults to True.

    Returns:
        bool: Returns True upon the successful completion of the evaluation loop.
    """
    
    print('[agent-arena, evaluate] Start evaluating Agent "{}" on\n     Arena "{}"'.\
            format(agent.get_name(), arena.get_name()))
    
    env_eval_configs = arena.get_eval_configs()
    if isinstance(agent, TrainableAgent):
        if checkpoint == -2:
            checkpoint = agent.load_best()
        elif checkpoint >= 0:
            agent.load_checkpoint(checkpoint)
        else:
            checkpoint = agent.load() # load the last one.
        
    print('[agent-arena, evaluate] Load_checkpoint', checkpoint) #-2 represent best

    for episode_config in tqdm(env_eval_configs):
        run(agent, arena, 'eval', episode_config, checkpoint=checkpoint, 
            policy_terminate=policy_terminate, env_success_stop=env_success_stop)
    
    return True

def log_validation_metrics(results, agent, step):
    """
    results: list of dicts.
        each dict: { metric_name -> list_of_values_over_steps }
    """

    # ----- EPISODE LEVEL METRICS -----

    # success: assume metric 'success' is 0/1 at end of episode
    success_values = []
    episode_lengths = []

    for res in results:
        # last entry per episode
        success_values.append(int(res['success'][-1]))
        # episode length as number of steps
        episode_lengths.append(len(next(iter(res.values()))))

    success_rate = float(np.mean(success_values)) if success_values else 0.0
    avg_ep_len = float(np.mean(episode_lengths))
    std_ep_len = float(np.std(episode_lengths))

    # log episode metrics
    agent.logger.log({
        "val/success_rate": success_rate,
        "val/avg_episode_length": avg_ep_len,
        "val/std_episode_length": std_ep_len,
    }, step=step)

    # ----- STEP-WISE METRIC STATS -----

    # for each metric, compute avg/std of the last step values across episodes
    last_step_stats = {}

    for metric_name in results[0].keys():  # assume all have same keys
        vals = []
        for res in results:
            if metric_name in res:
                vals.append(res[metric_name][-1])

        if len(vals) > 0:
            last_step_stats[f"val/{metric_name}_last_mean"] = float(np.mean(vals))
            last_step_stats[f"val/{metric_name}_last_std"] = float(np.std(vals))

    agent.logger.log(last_step_stats, step=step)

def validate(agent, arena, update_step, policy_terminate=True, env_success_stop=True):
    '''
        Validate the agent's current performance on the selected validation initial configuration of the arena.

        This method requires:
            * The agent has `get_writer` method to log the validation results.
            * The arena has `set_val` and `get_val_configs` methods to set the 
              arena to validation mode and get the validation configurations.
    '''
    val_configs = arena.get_val_configs()
    print(f'[agent-arena, validate] Validation episode configs size {len(val_configs)} at checkpoint {update_step}')
    results = []          
    for episode_config in tqdm(val_configs, desc="[agent-arena] Validating controller in the arena..."):
        _, res = run(agent, arena, 'val', episode_config, checkpoint=update_step, 
                     policy_terminate=policy_terminate, env_success_stop=env_success_stop)
        results.append(res['evaluation'])
    log_validation_metrics(results, agent, update_step)
    return results


def compare_results(results_1, results_2, compare_func):
    return compare_func(results_1, results_2)
    
def train_and_evaluate_single(
    agent: TrainableAgent, arena: Arena,
    validation_interval: int, total_update_steps: int, 
    eval_last_check: bool = False, eval_best_check: bool = True,
    policy_terminate: bool = True, env_success_stop: bool = True) -> bool:
    """
    Trains an agent on a single arena, performs periodic validation, and runs 
    final evaluations.

    This function manages the full training lifecycle: loading existing checkpoints, 
    running the training loop, performing validation checks at set intervals to 
    identify the best model, and running final evaluations on the last and/or 
    best checkpoints.

    Args:
        agent (TrainableAgent): The agent to be trained. The agent is expected 
            to implement `train`, `load`, `save`, `save_best`, and `load_best`.
        arena (Arena): The environment wrapper used for training and validation.
        validation_interval (int): The frequency (in update steps) at which 
            validation is performed. If 0, validation is skipped during training.
        total_update_steps (int): The target total number of update steps for 
            the training session.
        eval_last_check (bool, optional): If True, runs the `evaluate` function 
            on the final checkpoint after training concludes. Defaults to False.
        eval_best_check (bool, optional): If True, runs the `evaluate` function 
            on the best-performing checkpoint (saved during validation) after 
            training concludes. Defaults to True.
        policy_terminate (bool, optional): If True, allows the policy to trigger 
            episode termination during validation/evaluation steps. Defaults to True.
        env_success_stop (bool, optional): If True, validation/evaluation episodes 
            end immediately upon success. Defaults to True.

    Returns:
        bool: Returns True (implicitly) when the training and evaluation pipeline 
        completes.

    Raises:
        AssertionError: If `validation_interval` is set (>0) but `total_update_steps` 
        is invalid (<=0).
    """

    print('\n[agent-arena, train_and_evaluate_single] Training "{}" agent ...'.format(agent.get_name()))
    #validate(agent, arena, 0)
    if validation_interval > 0:
        assert total_update_steps > 0, 'Total update steps must be greater than 0'                    
        start_update_step = agent.load() #If no checkpint, it will return 0 --> no training

        for u in range(start_update_step, int(total_update_steps), validation_interval):
            arena.set_train()
            agent.train(validation_interval, [arena])
            agent.save()
            
            results = validate(agent, arena, u + validation_interval, 
                               policy_terminate=policy_terminate, env_success_stop=env_success_stop)

            best_results = load_best_results(agent.save_dir)
            if len(best_results) == 0 or compare_results(results, best_results, arena.compare) > 0:
                agent.save_best()
                save_best_results(results, agent.save_dir,  u + validation_interval)

    else:
        agent.train([arena])

    
    print('\n[agent-arena, train_and_evaluate_single] Finished training Agent "{}"'.format(agent.get_name()))

    if eval_last_check:
        evaluate(agent, arena, checkpoint=-1, 
                policy_terminate=policy_terminate, env_success_stop=env_success_stop)

    if eval_best_check:
        evaluate(agent, arena, checkpoint=-2, 
                policy_terminate=policy_terminate, env_success_stop=env_success_stop)
    
    return True

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

        for u in range(start_update_step, int(total_update_steps), validation_interval):
            agent.train(validation_interval, train_arenas)
            agent.save()
            
            results = validate(agent, val_arena, u + validation_interval)

            best_results = load_best_results(agent.save_dir)
            if len(best_results) == 0 or compare_results(results, best_results, val_arena.compare) > 0:
                agent.save_best()
                save_best_results(results, agent.save_dir,  u + validation_interval)

    else:
        agent.train(train_arenas)

    evaluate(agent, eval_arena, checkpoint=eval_checkpoint)