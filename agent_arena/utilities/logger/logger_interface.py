class Logger():

    def __init__(self):
        self.log_dir = './tmp'
    
    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def __call__(self, episode_config, result, filename=None):
        pass
    
    def check_exist(self, episode_config, filename=None):
        pass