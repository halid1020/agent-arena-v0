from agent_arena import TrainableAgent

class RLAgent(TrainableAgent):
    """
    Base class for RL agents.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def set_reward_processor(self, reward_processor):
        """
        Set the reward processor for the agent for training.
        Put these between sampled data from dataset and before feeding into the agent.
        Args:
            reward_processor: The reward processor to be set.
        """
        self.reward_processor = reward_processor