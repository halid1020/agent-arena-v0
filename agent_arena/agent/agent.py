# from abc import ABC, abstractmethod
# from agent_arena.utilities.utils import SummaryWriter
# from agent_arena.utilities.logger.dummy_logger import DummyLogger
 

# class Agent(ABC):

#     def __init__(self, config):
#         self.config = config
#         self.name = "agent"
#         self.internal_states = {} # collection of states for each interacting arena.
#         self.logger = DummyLogger()

#     def get_name(self) -> str:
#         """
#             This method return the name of the agent
#         """
#         return self.name
    
#     def set_log_dir(self, logger):
#         self.logger.set_log_dir(logger)
    
#     def reset(self, arena_ids) -> dict:
#         """
#             This method reset the agent before a new trial regarind the arena_ids.
#             It is used by perform method in utils.py.
#             It returns collection of booleans if the resets are successful.
#         """
#         for arena_id in arena_ids:
#             self.internal_states[arena_id] = {}
    
#     def init(self, informations) -> dict:
#         """
#             This method initialise the agent's internal state given the initial information.
#             It returns True if the initialisation is successful.
#         """
#         pass
    
#     def update(self, informations, actions) -> dict:
#         """
#             This method updates the agent's internal state given the current information and action.
#             It is used by perform method in utils.py.
#             It returns True if the update is successful.
#         """
#         pass

#     @abstractmethod
#     def act(self, informations, update=False) -> dict:
#         """
#             This method produces an action given the current information.
#             This method does not update the agent's internal state.
#             It is used by perform method in utils.py.
#         """
#         raise NotImplementedError
    
#     def success(self):
#         return {arena_id: False for arena_id in self.internal_states.keys()}

#     def terminate(self):
#         return {arena_id: False for arena_id in self.internal_states.keys()}

#     def get_phase(self):
#         return {arena_id: 'none' for arena_id in self.internal_states.keys()}

#     def get_state(self) -> dict:
#         """
#             This method returns intermdiate state after applying act, init or update method.
#         """
#         return self.internal_states

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dotmap import DotMap

from ..utilities.logger.dummy_logger import DummyLogger
from ..utilities.types import ActionType, InformationType, \
    ArenaIdType, ActionPhaseType


class Agent(ABC):
    def __init__(self, config: DotMap):
        self.config: DotMap = config
        self.name: str = "agent"
        self.internal_states: Dict[ArenaIdType, InformationType] = {}
        self.logger = DummyLogger()

    def get_name(self) -> str:
        """Return the name of the agent. This will be used but not limited for logging."""
        return self.name

    def set_log_dir(self, logdir: Any) -> None:
        """
        Set the log directory for the logger.
.name, 
        Args:
            logdir: The path to the log directory.
        """
        self.logger.set_log_dir(logdir)
        print("Log directory for the agent is set to {}".format(logdir))

    def reset(self, arena_ids: List[ArenaIdType]) -> List[bool]:
        """
        Reset the agent before a new trial for the given arena_ids.
        
        Args:
            arena_ids: List of arena identifiers.
        
        Returns:
            A list of booleans indicating if the resets were successful for each arena.
        """
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {}
        return [True for _ in arena_ids]

    def init(self, info_list: List[Dict[str, Any]]) -> List[bool]:
        """
        Initialise the agent's internal state given the initial information.
        
        Args:
            informations: Initial information for the agent from the reset arenas.
        
        Returns:
            A list indicating if the initialization was successful.
        """
        return [True for _ in info_list]

    def update(self, info_list: List[InformationType], actions: List[ActionType]) -> List[bool]:
        """
        Update the agent's internal state given the current information and action.
        
        Args:
            informations: Current list of informations for the agent from the arenas.
            actions: Actions taken by the agent.
        
        Returns:
            A list indicating if the update was successful.
        """
        return [True for _ in info_list]

    @abstractmethod
    def act(self, info_list: List[InformationType], update: bool = False) -> List[ActionType]:
        """
        Produce actions given the current informations from the arena, update the internal state if required.
        
        Args:
            info_list: Current information for the agent.
            update: Whether to update the agent's internal state; 
                    do not update if called `update` and/or 'init' method before.
        
        Returns:
            A list containing the agent's action.
        """
        raise NotImplementedError

    def success(self) -> Dict[ArenaIdType, bool]:
        """Check if the agent thinks it succeeded in each arena."""
        return {arena_id: False for arena_id in self.internal_states.keys()}

    def terminate(self) -> Dict[ArenaIdType, bool]:
        """Check if the agent thinks it should terminate in each arena."""
        return {arena_id: False for arena_id in self.internal_states.keys()}

    def get_phase(self) -> Dict[ArenaIdType, ActionPhaseType]:
        """Get the current action phase for each arena."""
        return {arena_id: 'none' for arena_id in self.internal_states.keys()}

    def get_state(self) -> Dict[ArenaIdType, InformationType]:
        """
        Return intermediate state after applying act, init or update method.
        
        Returns:
            A dictionary containing the internal states for each arena.
        """
        return self.internal_states
