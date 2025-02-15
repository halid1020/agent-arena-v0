# from abc import abstractmethod
# from typing import Optional, List

# from ..arena.arena import Arena
# from ..utilities.utils import TrainWriter
# from .agent import Agent

# class TrainableAgent(Agent):

#     def __init__(self, config):
#         super().__init__(config)
#         self.name = "trainable-agent"
#         self.train_writer = TrainWriter()
        

#     @abstractmethod
#     def train(self, update_steps: int, arena: List[Arena] = None) -> bool:
#         """
#             This method trains the agent optionaly on the arenas.
#             It returns True if the training is successful.
#         """
#         raise NotImplementedError
    
#     def load(self, path: Optional[str] = None) -> int:
#         """
#             This method loads the lastest agent's checkpoint from the path.
#             It the path is not provided, it loads the lastest checkpoint from its logger's log directory.
#             It return the checkpoint number it loaded, and returns -1 if 
#             the loading is unsuccessful.
#         """
#         return -1

#     def load_checkpoint(self, checkpoint: int) -> bool:
#         """
#             This method loads the agent from a specific checkpoint in its logger's log directory.
#             It returns True if the loading is successful.
#         """
#         return False
    
#     @abstractmethod
#     def save(self, path: Optional[str] = None) -> bool:
#         """
#             This method saves the agent to the path.
#             Save the current checkpoin.
#         """
#         raise NotImplementedError
    
#     @abstractmethod
#     def set_train(self):
#         """
#             This method set the agent to train mode.
#         """
#         raise NotImplementedError
    
#     @abstractmethod
#     def set_eval(self):
#         """
#             This method set the agent to eval mode.
#         """
#         raise NotImplementedError

#     def get_train_writer(self) -> TrainWriter:
#         """
#             This method returns a writer for logging training data.
#         """
#         return self.train_writer

from abc import abstractmethod
from typing import Optional, List
from ..arena.arena import Arena
from ..utilities.utils import TrainWriter
from .agent import Agent

class TrainableAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.name: str = "trainable-agent"
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