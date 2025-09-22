from abc import abstractmethod
from typing import Optional, List
from ..arena.arena import Arena
from ..utilities.utils import TrainWriter
from .agent import Agent

class TrainableAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.name: str = "trainable-agent"
        self.mode = 'train'
        self.train_writer: TrainWriter = TrainWriter()
        self.loaded = False

    def train(self, update_steps: int, arenas: Optional[List[Arena]] = None) -> bool:
        """
        Train the agent, optionally on the provided arenas.

        Args:
            update_steps: Number of update steps to perform.
            arenas: Optional list of arenas to train on.

        Returns:
            bool: True if the training is successful, False otherwise.
        """
        return False

    def load(self, path: Optional[str] = None) -> int:
        """
        Load the latest agent checkpoint from the specified path or the logger's log directory.

        Args:
            path: Optional path to load the checkpoint from.

        Returns:
            int: The checkpoint number that was loaded, or -1 if loading was unsuccessful.
        """
        return -1

    def load_checkpoint(self, checkpoint: int) -> bool:
        """
        Load the agent from a specific checkpoint in its logger's log directory.

        Args:
            checkpoint: The checkpoint number to load.

        Returns:
            bool: True if the loading is successful, False otherwise.
        """
        return False

    @abstractmethod
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save the current agent checkpoint to the specified path.

        Args:
            path: Optional path to save the checkpoint to, 
                  if path is None, save to the logger's log directory.

        Returns:
            bool: True if the saving is successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def set_train(self) -> None:
        """Set the agent to training mode."""
        raise NotImplementedError

    @abstractmethod
    def set_eval(self) -> None:
        """Set the agent to evaluation mode."""
        raise NotImplementedError

    def get_train_writer(self) -> TrainWriter:
        """
        Get the writer for logging training data.

        Returns:
            TrainWriter: The train writer object.
        """
        return self.train_writer