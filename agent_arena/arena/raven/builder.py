from .raven_env_adapter import RavenEnvAdapter
from dotmap import DotMap

class RavenBuilder():

    def build(config_str):

        config = RavenBuilder.parse_config_str(config_str)
        return RavenBuilder.build_from_config(**config)
    
    def build_from_config(task, disp, **kwargs):
        if disp == 'True':
            disp = True
        else:
            disp = False
        config = DotMap({
            'task': task,
            'disp': disp
        })
        env = RavenEnvAdapter(config)
        return env

    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v

        return config