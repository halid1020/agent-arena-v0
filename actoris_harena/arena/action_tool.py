
from abc import ABC, abstractmethod
from gym.spaces import Space
from typing import Dict, Any
from ..utilities.types import InformationType, ActionType, ActionSpaceType
from .arena import Arena


class ActionTool(ABC):
    """
    Abstract base class for action tools in a control problem.
    """

    def __init__(self):
        self.action_space = None

    def get_action_space(self) -> ActionSpaceType:
        """
        Get the action space of the tool.

        Returns:
            ActionSpaceType: The action space.
        """

        return self.action_space
    
    def sample_random_action(self) -> ActionType:
        """
        Sample a random action from the action space.

        Returns:
            ActionType: A randomly sampled action.
        """
        return self.action_space.sample()

    def get_action_horizon(self) -> int:
        """
        Get the action horizon (default is 0).

        Returns:
            int: The action horizon.
        """

        return 0

    def get_no_op(self) -> ActionType:
        """
        Get the no-op (no operation) action.

        Returns:
            ActionType: The no-op action.
        """
         
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, arena: Arena) -> InformationType:
        """
        Reset the action tool for a new episode.

        Args:
            arena (Arena): The arena in which the action is being performed.

        Returns:
            InformationType: Information about the reset state.
        """

        raise NotImplementedError
    
    @abstractmethod
    def step(self, arena: Arena, action: ActionType) -> InformationType:
        
        """
        Perform a step in the arena using the given action.

        Args:
            arena (Arena): The arena in which the action is being performed.
            action (ActionType): The action to be taken.

        Returns:
            InformationType: Information about the result of the action.
        """

        raise NotImplementedError

