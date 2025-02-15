class DM_ControlBuilder():


    def build(config_str):

        config = DM_ControlBuilder.parse_config_str(config_str)
        return DM_ControlBuilder.build_from_config(**config)
    
    def build_from_config(domain, task, **kwargs):
        from arena.dm_control.wrapper \
            import DM_ControlSuiteArena
        
        kwargs['pixel_observation'] = True
        
        return DM_ControlSuiteArena(domain, task, **kwargs)
    

    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v

        return config