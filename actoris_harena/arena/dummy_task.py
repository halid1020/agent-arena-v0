from .task import Task

class DummyTask(Task):

    def __init__(self):
        pass

    def reset(self, arena):
        pass

    def success(self, arena):
        return False
    
    def evaluate(self, arena, metrics):
        return {
            'dummy_metric': 0
        }

    def reward(self, arena):
        return 0
    
    def get_goal(self, arena):
        return None