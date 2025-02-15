from arena.deformable_raven.wrapper import DeformableRavenWrapper

class DeformbaleRavenBuilder():

    def build(config_str):

        config = DeformbaleRavenBuilder.parse_config_str(config_str)
        return DeformbaleRavenBuilder.build_from_config(**config)
    
    def build_from_config(task, gui, **kwargs):
        if gui == 'True':
            gui = True
        else:
            gui = False
        env = DeformableRavenWrapper(task, gui)
        return env

    def parse_config_str(config_str):
        config = {}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v

        return config