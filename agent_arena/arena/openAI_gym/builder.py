class OpenAIGymBuilder():


    def build(config_str):

        config = OpenAIGymBuilder.parse_config_str(config_str)
        return OpenAIGymBuilder.build_from_config(**config)
    
    def build_from_config(domain, **kwargs):
        from ..openAI_gym.wrapper \
            import OpenAIGymArena
        
        kwargs['pixel_observation'] = True
        
        return OpenAIGymArena(domain, **kwargs)
    

    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v

        return config