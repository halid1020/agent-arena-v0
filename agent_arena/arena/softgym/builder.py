import numpy as np
import logging

# from ..softgym.envs.fabric_velocity_control_env import FabricVelocityControlEnv
# from utilities.utils import print_dict_tree

class SoftGymBuilder():

    # Return a built environment from a configuration string
    # config_str example "mono-rect-fabric|task:flattening,observation:RGB,action:pick-and-place,initial:crumple"
    def build(config_str, ray=False):
        
        # Parse the config string and call the build function
        #print('helllooo')
        config = SoftGymBuilder.parse_config_str(config_str)
        if 'fabric' in config['domain']:
            from ..softgym.builders.fabric_domain_builder \
                import FabricDomainBuilder
            return FabricDomainBuilder.build_from_config(**config, ray=ray)
        elif 'towel' in config['domain']:
            from ..softgym.builders.fabric_domain_builder \
                import FabricDomainBuilder
            return FabricDomainBuilder.build_from_config(**config, ray=ray)
        elif 'Tshirt' in config['domain']:
            from ..softgym.builders.garment_domain_builder \
                import GarmentDomainBuilder
            return GarmentDomainBuilder.build_from_config(**config)
        elif 'clothfunnels' in config['domain']:
            from ..softgym.builders.cloth_funnel_domain_builder \
                import ClothFunnelDomainBuilder
            return ClothFunnelDomainBuilder.build_from_config(**config, ray=ray)
        elif 'ur3e' in config['domain']:
            from ..softgym.builders.ur3e_domain_builder \
                import UR3eDomainBuilder
            return UR3eDomainBuilder.build_from_config(**config, ray=ray)
        else:
            print("SoftGymBuilder: domain <{}> does not support".format(config['domain']))
            raise NotImplementedError

    def parse_config_str(config_str):
        config = {'domain': config_str.split('|')[0]}
        config_str = config_str.split('|')[1]
        items = config_str.split(',')

        for i in items:
            k, v = i.split(':')
            config[k] = v

        if 'action' not in config:
            return config

        config['num_picker'] = 2
        if '(' in config['action']:
            config['num_picker'] = int(config['action'].split('(')[1].split(')')[0])
        
        config['org_action'] = config['action']
        config['action'] = config['action'].split('(')[0]
        

        #print('config', config)

        return config