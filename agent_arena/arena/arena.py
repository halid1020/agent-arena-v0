from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

from ..utilities.logger.dummy_logger import DummyLogger
from ..utilities.types import ActionType, InformationType, ActionSpaceType


 
class Arena(ABC):
    """
        Abstract class for defining an arena in a control problem.
    """

    def __init__(self):
        self.name = "arena"
        self.mode = "train"
        self.setup_ray(id=0)
        self.disp = False
        self.random_reset = True
        self.logger = DummyLogger()
        self.eid = 0

        from .dummy_task import DummyTask
        self.task = DummyTask()
        from .dummy_action_tool import DummyActionTool
        self.action_tool = DummyActionTool()

    def set_log_dir(self, logdir: str):
        """
        Set the log directory for the logger.

        Args:
            logdir: The path to the log directory.
        """
        self.logger.set_log_dir(logdir)
        print("Log directory for the arena is set to {}".format(logdir))

    ##### The following is used by api #####
    def get_name(self) -> str:
        """
        Get the name of the arena.

        Returns:
            str: The name of the arena.
        """

        return self.name
    
    def set_disp(self, flg: bool):
        """
        Set the display flag for GUI demonstration.

        Args:
            flg (bool): True to enable display, False to disable.
        """

        self.disp = flg

    
    @abstractmethod
    def get_eval_configs(self) -> List[Dict[str, Any]]:
        """
        Get configurations for evaluation episodes.

        Returns:
            List of configurations for evaluation episodes.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_val_configs(self) -> List[Dict[str, Any]]:
        """
        Get configurations for validation episodes.

        Returns:
            List of configurations for validation episodes.
        """
        raise NotImplementedError

   
    # Core arena methods
    @abstractmethod
    def reset(self, episode_config: Optional[Dict[str, Any]] = None) -> InformationType:
        """
        Reset the arena for a new trial.

        Args:
            episode_config (Optional[Dict[str, Any]]): Configuration for the episode.
            
            if episode_config is None, 
                it set episode_config to {'eid': <random>, 'save_video': False}
            
            if eid is not given in episode_config and `random_reset` is True, 
                then sample a random eid;

        Returns:
            InformationType: Information about the arena state after reset.
        """
        
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action: ActionType) -> InformationType:
        """
            This method take an `action` to the environment using its action_tool,
            and return `information` about the arena along with task-oriented information using task.
        """
        raise NotImplementedError

    @abstractmethod
    def get_frames(self) -> List[np.ndarray]:
        """
        Get the list of frames collected so far.

        Returns:
            List[np.ndarray]: List of frames.
        """
        raise NotImplementedError
    
    @abstractmethod
    def clear_frames(self):
        """
        Clear the list of collected frames.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_goal(self) -> InformationType:
        """
        Get the goal of the current episode.

        Returns:
            InformationType: Information about the current goal.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_action_space() -> ActionSpaceType:
        """
        Get the action space of the arena.

        Returns:
            ActionSpaceType: The action space defined using gym.spaces.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_random_action():
        """
        Sample a random action from the action space.

        Returns:
            A uniformly sampled action from the action space.
        """
        raise NotImplementedError
    
    def set_train(self):
        """
        Set the arena to sample only training episodes.
        """
        self.mode = "train"

    def set_eval(self):
        """
        Set the arena to sample only evaluation episodes.
        """
        self.mode = "eval"

    def set_val(self):
        """
        Set the arena to sample only validation episodes.
        """
        self.mode = "val"
    
    @abstractmethod
    def get_no_op(self) -> ActionType:
        """
        Get the no-op action (action with no effect on the environment).

        Returns:
            ActionType: The no-op action.
        """
        raise NotImplementedError
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the arena and return metrics.

        Returns:
            Dict[str, Any]: A dictionary of evaluated metrics.
        """
        return self.task.evaluate(self, metrics={})
    
    @abstractmethod
    def get_action_horizon(self) -> int:
        """
        Get the action horizon (length of an episode) of the arena.

        Returns:
            int: The action horizon.
        """
        raise NotImplementedError
    
    def get_num_episodes(self) -> int:
        """
        Get the number of possible episodes in the arena under the current mode.

        Returns:
            int: The number of possible episodes, or -1 if undefined.
        """
        return -1

    def set_task(self, task):
        """
        Set the task for the arena.

        Args:
            task: The task to be set.
        """
        self.task = task

    def set_action_tool(self, action_tool):
        """
        Set the action tool for the arena.

        Args:
            action_tool: The action tool to be set.
        """
        self.action_tool = action_tool
    
    def success(self):
        return self.task.success(self)
    
    def setup_ray(self, id):
        """
            This method sets up the ray handle for the arena for multi-processing.
        """
        self.id = id
        self.ray_handle = {"val": id}

    def get_mode(self):
        return self.mode
    
    def get_episode_id(self):
        return self.eid