from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dotmap import DotMap

from ..utilities.types import ActionType, InformationType, \
    ArenaIdType, ActionPhaseType
from .wandb_logger import WandbLogger

class Agent(ABC):
    """
    Abstract base class for defining an Agent capable of interacting with 
    one or more arenas (environments) simultaneously. 
    
    This class manages the agent's configuration, logging setup, and 
    maintains independent internal states for multi-arena batch processing.
    """

    def __init__(self, config: DotMap):
        """
        Initialise the Agent with a given configuration.

        Args:
            config (DotMap): Configuration parameters for the agent.
        """
        self.config: DotMap = config
        self.name = "agent"
        # Stores the internal state (e.g., memory, hidden states) for each active arena
        self.internal_states: Dict[ArenaIdType, InformationType] = {}

    def get_name(self) -> str:
        """
        Return the name of the agent. 
        
        Returns:
            str: The agent's name, primarily used for logging and identification.
        """
        return self.name
    
    def set_log_dir(self, logdir: str, project_name: str = "actoris_harena", 
                    exp_name: str = "tmp", disable_wandb: bool = False):
        """
        Set the logging directory and initialise the Weights & Biases (wandb) logger.

        Args:
            logdir (str): Base directory path for saving local logs.
            project_name (str): Name of the wandb project.
            exp_name (str): Name of the specific experiment run.
            disable_wandb (bool): If True, disables wandb syncing (useful for debugging).
        """
        self.save_dir = logdir
        self.logger = WandbLogger(
            logdir,
            project_name,
            name=exp_name,
            config=dict(self.config),
            disable_wandb=disable_wandb
        )

    def reset(self, arena_ids: List[ArenaIdType]) -> List[bool]:
        """
        Reset the agent's internal state for a new trial across the given arenas.
        
        Args:
            arena_ids (List[ArenaIdType]): List of arena identifiers to reset.
        
        Returns:
            List[bool]: A list of booleans indicating if the reset was successful 
                        for each requested arena.
        """
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {}
        return [True for _ in arena_ids]

    def init(self, info_list: List[Dict[str, Any]]) -> List[bool]:
        """
        Initialise the agent's internal state using the initial observation/info 
        from newly reset arenas.
        
        Args:
            info_list (List[Dict[str, Any]]): Initial information dicts from the arenas.
        
        Returns:
            List[bool]: A list indicating if initialisation was successful for each arena.
        """
        return [True for _ in info_list]

    def update(self, info_list: List[InformationType], actions: List[ActionType]) -> List[bool]:
        """
        Update the agent's internal state given the current environmental information 
        and the last actions taken.
        
        Args:
            info_list (List[InformationType]): Current information from the arenas.
            actions (List[ActionType]): The previous actions taken by the agent.
        
        Returns:
            List[bool]: A list indicating if the state update was successful for each arena.
        """
        return [True for _ in info_list]

    def act(self, info_list: List[InformationType], updates: List[bool]) -> List[ActionType]:
        """
        Process a batch of environment information to produce actions for multiple arenas.
        
        Args:
            info_list (List[InformationType]): Current state information from the arenas.
            updates (List[bool]): Flags indicating whether to update the agent's internal 
                                  state for each arena. (Pass False if state was already 
                                  handled by an explicit `update()` or `init()` call).
        
        Returns:
            List[ActionType]: A list containing the predicted action for each arena.
        """
        actions = []
        for info, upd in zip(info_list, updates):
            actions.append(self.single_act(info, update=upd))
        return actions

    @abstractmethod
    def single_act(self, info: InformationType, update: bool = False) -> ActionType:
        """
        Produce a single action for a specific arena. Must be implemented by subclasses.

        Args:
            info (InformationType): The current state/observation from a single arena.
            update (bool): Whether to update the agent's internal state for this specific interaction.

        Returns:
            ActionType: The action to be executed in the environment.
        """
        raise NotImplementedError

    def success(self) -> Dict[ArenaIdType, bool]:
        """
        Check if the agent evaluates its current state as a success in each arena.
        
        Returns:
            Dict[ArenaIdType, bool]: A dictionary mapping arena IDs to success flags.
        """
        return {arena_id: False for arena_id in self.internal_states.keys()}

    def terminate(self) -> Dict[ArenaIdType, bool]:
        """
        Check if the agent determines the episode should terminate early in each arena.
        
        Returns:
            Dict[ArenaIdType, bool]: A dictionary mapping arena IDs to termination flags.
        """
        return {arena_id: False for arena_id in self.internal_states.keys()}

    def get_phase(self) -> Dict[ArenaIdType, ActionPhaseType]:
        """
        Get the current action phase (e.g., approaching, grasping, moving) for each arena.
        
        Returns:
            Dict[ArenaIdType, ActionPhaseType]: A dictionary mapping arena IDs to their current phase.
        """
        return {arena_id: 'none' for arena_id in self.internal_states.keys()}

    def get_state(self) -> Dict[ArenaIdType, InformationType]:
        """
        Retrieve the agent's current internal states across all tracked arenas.
        This is typically called after applying the act, init, or update methods.
        
        Returns:
            Dict[ArenaIdType, InformationType]: A dictionary containing the internal states.
        """
        return self.internal_states