from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..utilities.types import InformationType
from .arena import Arena

class Task(ABC):
    """
    Abstract base class defining the interface for tasks in an arena.
    """

    @staticmethod
    @abstractmethod
    def reset(arena: Arena) -> InformationType:
        """
        Reset the task for a new episode.

        Args:
            arena: The arena in which the task is being performed.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def success(arena: Arena) -> bool:
        """
        Check if the task has been successfully completed.

        Args:
            arena: The arena in which the task is being performed.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def evaluate(arena: Arena, metrics: List[str]) -> Dict[str, float]:
        """
        Evaluate the task performance and update metrics.

        Args:
            arena: The arena in which the task is being performed.
            metrics: List of metrics to evaluate.

        Returns:
            Dict[str, float]: The required evaluated metrics in a dictionary.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reward(arena: Any) -> Dict[str, float]:
        """
        Calculate the rewards for the current state of the task.

        Args:
            arena: The arena in which the task is being performed.

        Returns:
            Dict[str, float]: The rewards for the current state of the task.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_goal(arena: Arena) -> InformationType:
        """
        Get the current goal of the task for its current episode.

        Args:
            arena: The arena in which the task is being performed.

        Returns:
            InformationType: The current goal of the task.
        """
        raise NotImplementedError
