
from .action_tool import ActionTool

class DummyActionTool(ActionTool):
    

    def __init__(self):
        self.action_space = None

    def get_no_op(self):
         
        return self.action_space.sample()
    
    def reset(self, arena):
        pass

    def step(self, arena, action):
        
        return arena.step(action)
