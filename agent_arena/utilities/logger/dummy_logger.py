
from agent_arena.utilities.logger.logger_interface import Logger

class DummyLogger(Logger):
    def __init__(self):
        super().__init__()

    def __call__(self, episode_config, result, filename):
        pass
    
    def check_exist(self, episode_config, *args, **kwargs):
        return False