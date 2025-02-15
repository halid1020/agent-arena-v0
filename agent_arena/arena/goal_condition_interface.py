from abc import ABC, abstractmethod

class GoalConditionInterface(ABC):

    @abstractmethod
    def get_goal(self):
        """
            This method returns the goal of the current episode.
        """
        raise NotImplementedError