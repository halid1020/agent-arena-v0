class ArenaBuilder():

    def build(config_str, ray=False):
        target_builder = config_str.split('|')[0]

        if target_builder == 'softgym':
            from .softgym.builder \
                import SoftGymBuilder
            return SoftGymBuilder.build(config_str, ray=ray)
        
        # elif target_builder == 'rainbow-rect-fabric-sim':
        #     from .softgym.builders.rect_fabric_env_builder \
        #         import RectFabricEnvBuilder
        #     return RectFabricEnvBuilder.build(config_str)
        
        # elif 'garment-sim' in target_builder:
        #     from .softgym.builders.garment_env_builder \
        #         import GarmentEnvBuilder
        #     return GarmentEnvBuilder.build(config_str)
        
        elif target_builder == 'raven':
            from .raven.builder \
                import RavenBuilder
            return RavenBuilder.build(config_str)
        
        elif target_builder == 'deformable-raven':
            from .deformable_raven.bulider \
                import DeformbaleRavenBuilder
            return DeformbaleRavenBuilder.build(config_str)
        
        elif target_builder == 'dm-control-suite':
            from .dm_control.builder \
                import DM_ControlBuilder
            return DM_ControlBuilder.build(config_str)
        elif target_builder == 'openAI-gym':
            from .openAI_gym.builder \
                import OpenAIGymBuilder
            return OpenAIGymBuilder.build(config_str)
        
        else:
            print("EnvBuilder: target_builder <{}> does not support".format(target_builder))
            raise NotImplementedError
