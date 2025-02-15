import numpy as np
from utilities.utils import print_dict_tree

## action:pixel-pick-and-place, initial:flatten, domain:mono-single-tshirt, task:flattening


class GarmentDomainBuilder():
    
        
    def build_from_config(domain, initial, action, org_action, num_picker,task, 
                          disp=None, seed=None, horizon=None, 
                          save_control_step_info=False):
        
        if num_picker > 2:
            raise ValueError("num_picker should be less than 2")

        ## Put the default config here
        config = {
            "headless": False,
            "random_seed": 0,
            "use_cached_states": True,
            "save_cached_states": False,
            "recolour_config": False,
            "context": {}, ## TODO: If not give, context should be fixed.
            # "picker_low": [[-0.62, 0.02, -0.62], [-0.62, 0.02, -0.62]],
            # "picker_high": [[0.62, 1.0, 0.62], [0.62, 1.0, 0.62]],

            "picker_low": [[-1, 0.02, -1], [-1, 0.02, -1]],
            "picker_high": [[1, 1.0, 1], [1, 1.0, 1]],
            'save_image_dim': (256, 256),
            "picker_initial_pos": [[0.2, 0.2, 0.2], [-0.2, 0.2, 0.2]],
            "control_horizon": 1000,
            "num_picker": 2,
            "action_horizon": 30,
            'camera_params':{
                'default_camera':{
                    'pos': np.array([-0.0, 1.5, 0]),
                    'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                    'width': 720,
                    'height': 720},
            }
        }

        
        obj = domain.split('-')[-1]
        colour = domain.split('-')[0]
        number = domain.split('-')[1]
        config.update(GarmentDomainBuilder.return_config_from_initial_state(initial, domain))
        config.update(GarmentDomainBuilder.return_config_from_target_object(obj))
        GarmentDomainBuilder.return_config_from_context(colour, obj, config)
        config.update(GarmentDomainBuilder.return_config_from_action(action, num_picker))
        
        print('Softgym Garment Domain Config:')
        print_dict_tree(config)

        if disp is not None:
            disp = True if disp == 'True' else False
            config['headless'] = not disp

        if seed is not None:
            config['random_seed'] = int(seed)


        return GarmentDomainBuilder.build_env(config, domain, initial, action, task)

        
        
    def build_env(config, domain, initial, action, task):
        from arena.softgym.envs.garment_velocity_control_env \
            import GarmentVelocityControlEnv

        env = GarmentVelocityControlEnv(config)


        ## Wrap Action
        if action == 'pixel-pick-and-place':

            kwargs = {
                'motion_trajectory': 'rectangular',
                'pick_height': 0.028,
                'place_height':  0.1,
                'pick_lower_bound': [-1, -1],
                'pick_upper_bound': [1, 1],
                'place_lower_bound': [-1, -1],
                'place_upper_bound': [1, 1],
                "release_height": 0.05,
                "prepare_height": 0.1,
                'fix_pick_height': True,
                'fix_place_height': True,
                'velocity': 0.05,
                'action_dim': config['num_picker']
            }
            kwargs['action_horizon'] = config['action_horizon']
            from arena.softgym.picker_action_wrappers.pixel_pick_and_place_wrapper \
                  import PixelPickAndPlaceWrapper
            env = PixelPickAndPlaceWrapper(env, **kwargs)
        
        else:
            raise NotImplementedError

        ## Wrap Task
        if task == 'flattening':
            from arena.softgym.task_wrappers.rect_fabric.garment_flattening_wrapper \
                import GarmentFlatteningWrapper
            env = GarmentFlatteningWrapper(env, domain=domain, initial=initial)
        else:
            raise NotImplementedError

        
        return env
    
    def return_config_from_context(context, target_object, config):
        #print('color_mode', context)
        if 'mono' in context:
            pass
            #pass
            #config['context'].remove('colour')
        elif 'rainbow' in context:
            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'inside_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                }
            }
        else:
            raise ValueError('Invalid color mode')
    
        if target_object == 'Tshirt':
            config['context']['garment'] = ['Tshirt']
        else:
            raise NotImplementedError
        return config
        
    
    def return_config_from_target_object(target_object):
        config = {}
        if target_object == 'Tshirt':
            config['num_variations'] = 1000
            config['garment'] = ['Tshirt']
            config['eval_tiers'] = {
                0: [i for i in range(30)]
            }
            config['video_episodes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            config['val_episodes'] = [31, 32, 33]

        elif target_object == 'all':
            config['eval_tiers'] = {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            }
            config['video_episodes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            raise NotImplementedError
        
        return config
    
    def return_config_from_initial_state(initial_state, domain):
        
        config = {'initial_state': initial_state, 'context': {}}

        if initial_state == 'crumple':
            config['use_cached_states'] = True
            config['save_cached_states'] = True
            config['context']['position'] = 0.6
            config['context']['rotation'] = True
            config['context']['state'] = True
            config['context']['flip_face'] = 0.5
            config["cached_states_path"] = "{}.pkl".format(domain)

        else:
            raise NotImplementedError
        
        return config
    
    def return_config_from_action(action, num_picker):
        config = {}
        if action in ['pixel-pick-and-place', 'pixel-pick-and-place-z', 'world-pick-and-place']:
            config.update({
                "picker_radius": 0.015,
                "picker_initial_pos": [[0.55, 0.2, 0.55], [-0.55, 0.2, 0.55]],
                "action_dim": num_picker
            })
        else:
            ### Raise Error telling that such action is invalid
            print("Action <{}> not supported".format(action))
            raise NotImplementedError
        
        return config