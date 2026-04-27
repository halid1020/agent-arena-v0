from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

from ..utilities.logger.dummy_logger import DummyLogger
from ..utilities.types import ActionType, InformationType, ActionSpaceType

class Arena(ABC):
    """
    Abstract base class for defining an arena (environment wrapper) in a control problem.
    This class handles the boilerplate for episode tracking, seeding, logging, 
    and defining the expected interfaces for concrete implementations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Arena with default properties and configurations.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing trial limits, 
                                     action horizons, and display settings.
        """
        self.name = "arena"               # Default name of the arena for logging
        self.mode = "train"               # Default mode is training
        self.setup_ray(id=0)              # Initialize multi-processing handle
        self.disp = False                 # GUI display flag
        self.random_reset = True          # Flag to allow random episode resets
        self.logger = DummyLogger()       # Fallback logger
        self.eid = 0                      # Current Episode ID
        self.action_horizon = config.get('action_horizon', -1)

        # Lazy imports for dummy placeholders to avoid circular dependencies
        from .dummy_task import DummyTask
        from .dummy_action_tool import DummyActionTool
        
        self.task = DummyTask()
        self.action_tool = DummyActionTool()
        self.video_frames = []            # Buffer to store frames for video rendering
        self.aid = 0                      # Arena ID
        self.action_space = None          # To be defined by child classes (gym.Space)
        
        # Trial caps based on modes
        self.num_eval_trials = config.get('num_eval_trials', 30)
        self.num_train_trials = config.get('num_train_trials', 1000)
        self.num_val_trials = config.get('num_val_trials', 10)

    def set_id(self, id: int):
        """Set the unique identifier for this arena instance."""
        self.aid = id
    
    def get_id(self) -> int:
        """Return the unique identifier of this arena instance."""
        return self.aid

    def set_log_dir(self, logdir: str, project_name: str, exp_name: str):
        """
        Set the logging directory and configure the active logger.

        Args:
            logdir (str): Base path to the log directory.
            project_name (str): Name of the overarching project.
            exp_name (str): Specific name of the experiment run.
        """
        self.logger.set_log_dir(logdir, project_name, exp_name)
        print("Log directory for the arena is set to {}".format(logdir))
    
    def get_name(self) -> str:
        """Return the name of the arena."""
        return self.name
    
    def set_disp(self, flg: bool):
        """
        Toggle the display flag for GUI demonstration (e.g., cv2.imshow).

        Args:
            flg (bool): True to enable GUI display, False to disable.
        """
        self.disp = flg

    def get_num_episodes(self) -> np.int32:
        """
        Get the total number of episodes allocated for the current operating mode.

        Returns:
            np.int32: The number of trials for 'eval', 'val', or 'train'.
        """
        if self.mode == 'eval':
            return self.num_eval_trials
        elif self.mode == 'val':
            return self.num_val_trials
        elif self.mode == 'train':
            return self.num_train_trials
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported.")

    def get_eval_configs(self) -> List[Dict[str, Any]]:
        """Generate configurations for all evaluation episodes."""
        return [{'eid': eid, 'tier': 0, 'save_video': True} for eid in range(self.num_eval_trials)]

    def get_train_configs(self) -> List[Dict[str, Any]]:
        """Generate configurations for all training episodes."""
        return [{'eid': eid, 'tier': 0, 'save_video': getattr(self, 'config', {}).get('save_video', False)} 
                for eid in range(self.num_train_trials)]

    def get_val_configs(self) -> List[Dict[str, Any]]:
        """Generate configurations for all validation episodes."""
        return [{'eid': eid, 'tier': 0, 'save_video': True} for eid in range(self.num_val_trials)]
   
    # Core arena methods
    @abstractmethod
    def reset(self, episode_config: Optional[Dict[str, Any]] = None) -> InformationType:
        """
        Reset the arena for a new trial.

        Args:
            episode_config (Optional[Dict[str, Any]]): Configuration for the episode.
            If None, defaults to {'eid': <random>, 'save_video': False}.
            If 'eid' is omitted and `random_reset` is True, a random eid is sampled.

        Returns:
            InformationType: A dictionary containing the observation and arena state.
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action: ActionType) -> InformationType:
        """
        Execute an action within the environment.

        Args:
            action (ActionType): The action to be executed, provided by the agent.

        Returns:
            InformationType: Arena state and task-oriented info (rewards, done flags).
        """
        raise NotImplementedError

    def get_frames(self) -> List[np.ndarray]:
        """Return the list of RGB frames collected during the episode. This is for video saving."""
        return self.video_frames
    
    def clear_frames(self):
        """Clear the frame buffer."""
        self.video_frames.clear()
    
    def get_action_space(self) -> ActionSpaceType:
        """Return the action space (gym.Space) of the arena."""
        return self.action_space
    
    def sample_random_action(self) -> ActionType:
        """Return a uniformly sampled action from the arena's action space."""
        return self.action_space.sample()
    
    def set_train(self):
        """Set the arena to sample only training episodes."""
        self.mode = "train"

    def set_eval(self):
        """Set the arena to sample only evaluation episodes."""
        self.mode = "eval"

    def set_val(self):
        """Set the arena to sample only validation episodes."""
        self.mode = "val"
    
    def get_no_op(self) -> ActionType:
        """
        Get a zeroed-out action, representing no movement or effect.

        Returns:
            ActionType: A zero array matching the action space shape.
        """
        return np.zeros(self.action_space.shape, dtype=np.float32)
    
    @abstractmethod
    def compare(self, result_1: List[Dict], result_2: List[Dict]) -> int:
        """
        Compare validation results from two different policies.

        Args:
            result_1: List of information dictionaries from policy 1.
            result_2: List of information dictionaries from policy 2.

        Returns:
            int: 1 if result_1 > result_2, -1 if result_1 < result_2, 0 if tied/similar.
        """
        raise NotImplementedError
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the arena's current task status and return metrics."""
        return self.task.evaluate(self, metrics={})
    
    def get_action_horizon(self) -> int:
        """Return the maximum number of steps (action horizon) per episode."""
        return self.action_horizon

    def set_task(self, task):
        """Inject a specific task evaluation metric tool into the arena."""
        self.task = task

    def set_action_tool(self, action_tool):
        """Inject a specific action transformation tool into the arena."""
        self.action_tool = action_tool

    def get_goal(self):
        """Fetch the current goal from the underlying task."""
        return self.task.get_goal()
    
    def success(self) -> bool:
        """Check if the underlying task considers the current state a success."""
        return self.task.success(self)
    
    def setup_ray(self, id: int):
        """
        Set up the ray handle to facilitate multi-processing.

        Args:
            id (int): Ray worker ID.
        """
        self.id = id
        self.ray_handle = {"val": id}

    def get_mode(self) -> str:
        """Return the current mode ('train', 'eval', or 'val')."""
        return self.mode
    
    def get_episode_id(self) -> int:
        """Return the current Episode ID (eid)."""
        return self.eid