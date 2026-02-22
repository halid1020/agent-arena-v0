from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dotmap import DotMap

from ..utilities.logger.dummy_logger import DummyLogger
from ..utilities.types import ActionType, InformationType, \
    ArenaIdType, ActionPhaseType
from .wandb_logger import WandbLogger

class Agent(ABC):
    def __init__(self, config: DotMap):
        self.config: DotMap = config
        self.name = "agent"
        self.internal_states: Dict[ArenaIdType, InformationType] = {}
        
        

    def get_name(self) -> str:
        """Return the name of the agent. This will be used but not limited for logging."""
        return self.name
    
    def set_log_dir(self, logdir, project_name="actoris_harena", exp_name="tmp", disable_wandb=False):
        self.save_dir = logdir
        self.logger = WandbLogger(logdir,
            project_name,
            name=exp_name,
            config=dict(self.config),
            disable_wandb=disable_wandb)


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

    
    def act(self, info_list: List[InformationType], updates: List[bool]) -> List[ActionType]:
        """
        Produce actions given the current informations from the arena, update the internal state if required.
        
        Args:
            info_list: Current information for the agent.
            update: Whether to update the agent's internal state; 
                    do not update if called `update` and/or 'init' method before.
        
        Returns:
            A list containing the agent's action.
        """
        actions = []
        for info, upd in zip(info_list, updates):
            actions.append(self.single_act(info, upd))
        return actions

    @abstractmethod
    def single_act(self, info, update=False) -> ActionType:
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
