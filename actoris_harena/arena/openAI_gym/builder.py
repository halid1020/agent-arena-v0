from dotmap import DotMap

class OpenAIGymBuilder():


    def build(config_str):

        config = OpenAIGymBuilder.parse_config_str(config_str)
        config = DotMap(config)
        return OpenAIGymBuilder.build_from_config(config)
    
    def build_from_config(config):
        from ..openAI_gym.wrapper \
            import OpenAIGymArena
        
        config.pixel_observation = True
        
        return OpenAIGymArena(config)
    

    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v

        return config