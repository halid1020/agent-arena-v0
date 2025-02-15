import numpy as np
import logging

from ..envs.fabric_env \
    import FabricEnv, FabricEnvRay
from agent_arena.utilities.utils import print_dict_tree

class FabricDomainBuilder():
        
    def build_from_config(domain, initial, action, org_action, num_picker,task, 
                          disp=None, seed=None, horizon=None, 
                          save_control_step_info=False, ray=False):
        if num_picker > 2:
            raise ValueError("num_picker should be less than 2")

        ## Put the default config here
        config = {
            "headless": False,
            "random_seed": 0,
            "use_cached_states": True,
            "save_cached_states": False,
            "recolour_config": False,
            "num_picker": 2,
            'save_control_step_info': save_control_step_info,
            'save_image_dim': (256, 256),
            'control_horizon': 2000,
            'context': {},
            "picker_low": [[-0.62, 0.02, -0.62], [-0.62, 0.02, -0.62]],
            "picker_high": [[0.62, 1.0, 0.62], [0.62, 1.0, 0.62]],
            "picker_initial_pos": [[0.2, 0.2, 0.2], [-0.2, 0.2, 0.2]],
            'action_horizon': 30,
        }


        
        config.update(FabricDomainBuilder.return_config_from_domain(domain))
        config.update(FabricDomainBuilder.return_config_from_initial_state(domain, initial))
        config.update(FabricDomainBuilder.return_config_from_action(action, num_picker))
        config.update(FabricDomainBuilder.return_config_from_task(task, initial, domain))
        config['action_horizon'] = int(horizon) if horizon is not None else config['action_horizon']

        print('Softgym Fabric Domain Config:')
        print_dict_tree(config)

        if disp is not None:
            disp = True if disp == 'True' else False
            config['headless'] = not disp

        if seed is not None:
            config['random_seed'] = int(seed)
        
        #print('config keys', config)
        
         #### Define base env
        
        if ray:
            env = FabricEnvRay.remote(config)
        else:
            env = FabricEnv(config)
       
        
        ### Put Action Wrappers
        if action == 'pixel-pick-and-place':
            kwargs = {}
            kwargs['action_horizon'] = config['action_horizon']
            from ...softgym.picker_action_wrappers.pixel_pick_and_place \
                  import PixelPickAndPlace
            action_tool = PixelPickAndPlace(**kwargs)
            #env.set_action_tool(action_tool)

        elif action == 'pixel-pick-and-fling':
            kwargs = {}
            kwargs['action_horizon'] = config['action_horizon']
            from ...softgym.picker_action_wrappers.pixel_pick_and_fling \
                import PixelPickAndFling
            action_tool = PixelPickAndFling(**kwargs)
            #env.set_action_tool(action_tool)
        elif action == 'hybrid-action-primitive':
            kwargs = {}
            kwargs['action_horizon'] = 2 #config['action_horizon']
            from ...softgym.picker_action_wrappers.hybrid_action_primitive \
                import HybridActionPrimitive
            action_tool = HybridActionPrimitive(**kwargs)
            #env.set_action_tool(action_tool)

        elif action == 'velocity-grasp':
            if num_picker == 1:
                pass
            else:
                raise NotImplementedError
            
            from ...softgym.picker_action_wrappers.velocity_grasp \
                import VelocityGrasp
            env = VelocityGrasp(env, action_repeat=10, 
                                max_interactive_step=1000)
        else:
            ## Raise with error message
            print("Action {} is not supported".format(action))
            raise NotImplementedError
        
        
        
        #print('here here')
        ### Put Task Wrapper
        if task == 'flattening':
            from ..tasks.rect_fabric.towel_flattening \
                import TowelFlatteningTask
            task = TowelFlatteningTask()
            #env.set_task(task)
        elif task == 'canonical-flattening':
            from ...softgym.task_wrappers.rect_fabric.flattening_wrapper \
                import FlatteningWrapper
            env = FlatteningWrapper(env, canonical=True, domain=domain, initial=initial)
        elif task == 'one-step-folding':
            from ...softgym.task_wrappers.rect_fabric.one_step_folding_wrapper \
                import OneStepFoldingWrapper
            env = OneStepFoldingWrapper(env, domain=domain, initial=initial)
        elif task == 'diagonal-folding':
            #print('hello')
            from ...softgym.task_wrappers.rect_fabric.diagonal_folding_wrapper \
                import DiagonalFoldingWrapper
            env = DiagonalFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'corners-edge-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.corners_edge_inward_folding_wrapper \
                import CornersEdgeInwardFoldingWrapper
            env = CornersEdgeInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-diagonal-folding':
            from ...softgym.task_wrappers.rect_fabric.diagonal_folding_wrapper \
                import DiagonalFoldingWrapper
            env = DiagonalFoldingWrapper(env, canonical=True, domain=domain, initial=initial)

        elif task == 'diagonal-cross-folding':
            from ...softgym.task_wrappers.rect_fabric.diagonal_cross_folding_wrapper \
                import DiagonalCrossFoldingWrapper
            env = DiagonalCrossFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        
        elif task == 'canonical-diagonal-cross-folding':
            from ...softgym.task_wrappers.rect_fabric.diagonal_cross_folding_wrapper \
                import DiagonalCrossFoldingWrapper
            env = DiagonalCrossFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
        elif task == 'cross-folding':
            from ...softgym.task_wrappers.rect_fabric.cross_folding_wrapper \
                import CrossFoldingWrapper
            env = CrossFoldingWrapper(env, domain=domain, initial=initial, action=org_action)


       

        elif task == 'one-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.one_corner_inward_folding_wrapper \
                import OneCornerInwardFoldingWrapper
            env = OneCornerInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-one-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.one_corner_inward_folding_wrapper \
                import OneCornerInwardFoldingWrapper
            env = OneCornerInwardFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
        
        elif task == 'double-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.double_corner_inward_folding_wrapper \
                import DoubleCornerInwardFoldingWrapper
            env = DoubleCornerInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-double-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.double_corner_inward_folding_wrapper \
                import DoubleCornerInwardFoldingWrapper
            env = DoubleCornerInwardFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
        
        elif task == 'all-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.all_corner_inward_folding_wrapper \
                import AllCornerInwardFoldingWrapper
            env = AllCornerInwardFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-all-corner-inward-folding':
            from ...softgym.task_wrappers.rect_fabric.all_corner_inward_folding_wrapper \
                import AllCornerInwardFoldingWrapper
            env = AllCornerInwardFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
        
        
        ## Following tasks are for both square and rectangular fabrics.
        elif task == 'rectangular-folding':
            from ...softgym.task_wrappers.rect_fabric.rectangular_folding_wrapper \
                import RectangularFoldingWrapper
            env = RectangularFoldingWrapper(env, domain=domain, initial=initial, action=org_action)

        elif task == 'canonical-rectangular-folding':
            from ...softgym.task_wrappers.rect_fabric.rectangular_folding_wrapper \
                import RectangularFoldingWrapper
            env = RectangularFoldingWrapper(env, canonical=True, domain=domain, initial=initial)

        elif task == 'side-folding':
            from ...softgym.task_wrappers.rect_fabric.side_folding_wrapper \
                import SideFoldingWrapper
            env = SideFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-side-folding':
            from ...softgym.task_wrappers.rect_fabric.side_folding_wrapper \
                import SideFoldingWrapper
            env = SideFoldingWrapper(env, canonical=True, domain=domain, initial=initial)

        elif task == 'double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, domain=domain, initial=initial, action=org_action)
        elif task == 'canonical-double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, canonical=True, domain=domain, initial=initial)

        elif task == 'double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, domain=domain, initial=initial)
        elif task == 'canonical-double-side-folding':
            from ...softgym.task_wrappers.rect_fabric.double_side_folding_wrapper \
                import DoubleSideFoldingWrapper
            env = DoubleSideFoldingWrapper(env, canonical=True, domain=domain, initial=initial)
       

        else:
            logging.error("[softgym builder] task {} is not supported".format(task))
            raise NotImplementedError
        
        if ray:
            env.set_action_tool.remote(action_tool)
            env.set_task.remote(task)
        else:
            env.set_action_tool(action_tool)
            env.set_task(task)
        
        return env
        
    
    def return_config_from_domain(domain):
        config = {}

        if domain == 'mono-square-fabric':
            config['num_variations'] = 1000
            config['cached_states_path'] = "mono-square-fabric.pkl"
            config['eval_tiers'] = {
                4: [0, 10, 15, 20, 22, 25, 26, 27, 30, 31, 34, 35, 38, 39, 41, 42, 53, 56, 59, 60, 72, 75, 81, 87, 88, 90, 94],
                
                3: [1, 3, 16, 32, 45, 50, 54, 55, 57, 64, 65, 67, 71, 74, 79, 83, 86, 89, 91, 97, 98, 99],
                
                2: [5, 11, 21, 23, 28, 46, 52, 69, 70, 92],
                
                1: [36, 66, 96],
                
                0: [33, 49, 76]
                
            }
            config['video_episodes'] = [0, 10, 1, 3, 5, 11, 23, 36, 33]
            config['val_episodes'] = [2, 4, 6]
        
        
        elif domain == 'rainbow-rectangular-fabrics':
            config['num_variations'] = 5000
            config['cached_states_path'] = "rainbow-rectangular-fabrics.pkl"
            config['eval_tiers'] = {
                
                
                7: [6, 109, 197, 243, 320, 438, 465] ,
                
                6: [1, 22, 28, 30, 44, 49, 50] ,
                
                5: [0, 136, 159, 166, 187, 193, 208],

                4: [21, 68, 150, 189, 198, 249, 307] ,
                
                3: [8, 206, 212, 277, 323, 360, 474],
                
                2: [37, 55, 65, 69, 104, 112, 140],
                
                1: [26, 40, 43, 54, 56, 64, 75],
                
                0: [177, 216, 286, 329, 375, 443, 446]
                
            }
            config['val_episodes'] = [6, 109, 197]
            # first two eps of each tier
            config['video_episodes'] = [8, 206, 37, 55, 177, 216, 6, 109, 1, 22, 0, 136, 21, 68]
        
        elif domain == 'rainbow-square-fabrics':
            config['num_variations'] = 2000
            config['cached_states_path'] = "rainbow-square-fabrics.pkl"
            config['eval_tiers'] = {
                

                3: [10, 15, 28, 29, 36, 37, 46],
                
                2:  [33, 38, 41, 47, 49, 60, 66],
                
                1:   [4, 6, 8, 13, 18, 32, 44],
                
                0: [9, 11, 22, 31, 40, 55, 58]
                
            }

            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        
        elif domain == 'vcd-rect-fabric':
            config['num_variations'] = 30
            config['cached_states_path'] = "vcd_rect_fabric.pkl"
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == 'vcd-square-fabric':
            config['num_variations'] = 30
            config['cached_states_path'] = "vcd_square_fabric.pkl"
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]

        elif domain in ['realadapt-towels', 'realadapt-towels-sq', 'realadapt-rainbow-towels']:
            config['num_variations'] = 1000
            config['cached_states_path'] = "{}.pkl".format(domain)
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i+30 for i in range(3)]
        
        elif domain in ['realadapt-nomisgrasp-towels', 'realadapt-nomultilayer-towels']:
            config['num_variations'] = 1000
            config['cached_states_path'] = "realadapt-towels.pkl".format(domain)
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i+30 for i in range(3)]
        
        elif domain == "ffn-rect-fabric":
            config['num_variations'] = 30
            config['cached_states_path'] = "vcd_square_fabric.pkl"
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        elif domain == "ffn-square-fabric":
            config['num_variations'] = 30

            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffnm-rect-fabric":
            config['num_variations'] = 30

            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffnm-square-fabric":
            config['num_variations'] = 30
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffmr-square-fabric":
            config['num_variations'] = 40
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
        
        elif domain == "ffmr-rect-fabric":
            config['num_variations'] = 40
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
            
        else:
            print("Target object <{}> not supported".format(domain))
            raise NotImplementedError
        
        return config
    
    def return_config_from_initial_state(domain, initial_state):
        
        config = {'initial_state': initial_state, 'context': {}}

        if domain == 'rainbow-rectangular-fabrics':
            config['context']['rectangular'] = True
            config['context']['size'] = {
                'width': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                },

                'length': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                }
            }

            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
            }
        
        if domain == 'rainbow-square-fabrics': 
            config['context']['rectangular'] = False
            config['context']['size'] = {
                'width': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                },

                'length': {
                    'lower_bound': 0.2,
                    'upper_bound': 0.7
                }
            }

            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
            }
        #print('domain', domain)
        if domain in ['realadapt-towels', 'realadapt-towels-sq', 
                      'realadapt-rainbow-towels', 'realadap-nomisgrasp-towels',
                      'realadapt-nomultilayer-towels']:
            
            #print("HELELLELE")
            if 'sq' in domain:
                config['context']['rectangular'] = False
            else:
                config['context']['rectangular'] = True
            config['context']['size'] = {
                'width': {
                    'lower_bound': 0.15,
                    'upper_bound': 0.45
                },

                'length': {
                    'lower_bound': 0.15,
                    'upper_bound': 0.45
                }
            }
            
            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
            }
                
            config['context']['colour_mode'] = 'both_same'
            config['grasp_mode'] = {
                'around': 0.9,
                'miss': 0.1
            }
            #print('here')

        if domain == 'realadapt-rainbow-towels':
            
            
            config['context']['colour'] = {
                'front_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
                'back_colour': {
                    'lower_bound': [0.0, 0.0, 0.0],
                    'upper_bound': [1.0, 1.0, 1.0]
                },
            }
                
        if domain == 'realadapt-nomisgrasp-towels':
            config['grasp_mode'] = {
                'around': 1.0,
                'miss': 0.0
            }
        
        if domain == 'realadapt-nomultilayer-towels':
            config['grass_mode'] = {
                'closest': 0.9,
                'miss': 0.1
            }


        if initial_state == 'crumpled':
            config['use_cached_states'] = True
            config['save_cached_states'] = True
            config['context']['position'] = 0.6
            config['context']['rotation'] = True
            config['context']['state'] = True
            config['context']['flip_face'] = 0.5
            
            if domain == 'vcd-rect-fabric':
                config['context']['rectangular'] = True
                config['context']['size'] = {
                    'width': {
                        'lower_bound': 0.19,
                        'upper_bound': 0.21
                    },

                    'length': {
                        'lower_bound': 0.31,
                        'upper_bound': 0.34
                    }
                }
            
            if domain == 'vcd-square-fabric':
                config['context']['size'] = {
                    'width': {
                        'lower_bound': 0.25,
                        'upper_bound': 0.28
                    }
                }
                config['context']['rectangular'] = False

            


        elif initial_state == 'flattened':
            config['use_cached_states'] = False
            config['num_variations'] = 1000
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            #config['eval_tiers'][0][24] = 31
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [30+i for i in range(3)]
            #print('context', config['context'])
            config['context'].update({
                
                'position': 0.6,
                'rotation': True,
                'state': False,
                'flip_face': 0.5,
                'all_visible': True
            })
        
        elif initial_state == 'random_flattened':
            config['use_cached_states'] = False
            config['num_variations'] = 1000
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [30+i for i in range(3)]
            
            config['context'].update({
                
                'position': 0.6,
                'rotation': True,
                'state': False,
                'flip_face': 0.5,
                'all_visible': False
            })
           

            

        elif initial_state == 'canonical':
            config['context'] = {
                
                'position': 0,
                'rotation': False,
                'state': False,
                'flip_face': 0.5
            }
            config['eval_tiers'] = {
                0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29]
            }
            config['video_episodes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            config['use_cached_states'] = False
            config['num_variations'] = 100
        
        elif initial_state == 'centre-flattened':
            config['use_cached_states'] = False
            config['num_variations'] = 1000
            config['eval_tiers'] = { 0: [i for i in range(30)] }
            config['video_episodes'] = [i for i in range(10)]
            config['val_episodes'] = [i for i in range(3)]
            
            config['context'] = {
                
                'position': 0,
                'rotation': True,
                'state': False,
                'flip_face': 0
            }

            if domain == 'ffmr-rect-fabric':
                config['context']['rectangular'] = True
                config['context']['size'] = {
                    'length': {
                        'lower_bound': 0.32,
                        'upper_bound': 0.37
                    },

                    'width': {
                        'lower_bound': 0.22,
                        'upper_bound': 0.33
                    }
                }
            if domain == 'ffmr-square-fabric':
                config['context']['rectangular'] = False
                config['context']['size'] = {
                    

                    'width': {
                        'lower_bound': 0.32,
                        'upper_bound': 0.37
                    }
                }

        elif initial_state == 'crumpled':
            pass
        else:
            raise NotImplementedError
    
        return config
    
    def return_config_from_action(action, num_picker):
        config = {}
        if action in ['pixel-pick-and-place', 'pixel-pick-and-place-z', 'world-pick-and-place',
                      'pixel-pick-and-fling', 'hybrid-action-primitive']:
            config.update({
                "picker_radius": 0.015,
                "picker_initial_pos": [[0.55, 0.2, 0.55], [-0.55, 0.2, 0.55]],
                "action_dim": num_picker
            })
        elif action == 'velocity-grasp':
            config.update({
                "picker_radius": 0.015,
            })
        else:
            ### Raise Error telling that such action is invalid
            print("Action <{}> not supported".format(action))
            raise NotImplementedError
        
        return config
    
    def return_config_from_task(task, init, domain):
        config = {}
        if task in ['flattening', 'canonical-flattening']:
            #config['reward_mode'] = "hoque_ddpg"
            config['action_horizon'] = 30 # 30
            #pass
        elif task in ['one-step-folding']:
            config['action_horizon'] = 1
            
        elif 'folding' in task:
            #config['reward_mode'] = "normalised_particle_distance"
            config['action_horizon'] = 15
            if init == 'crumpled':
                config['action_horizon'] += 30
            pass
        elif task == 'all':
            #config['reward_mode'] = "normalised_particle_distance"
            #config['action_horizon'] = 25
            pass
        else:
            logging.error("[softgym builder] task {} is not supported".format(task))
            raise NotImplementedError
        
        if domain == 'ffn-square-fabric' :
                config['camera_params'] = {
                    'default_camera':{
                        'pos': np.array([-0.0, 0.65, 0]),
                        'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                        'width': 720,
                        'height': 720},
                }
                config['cloth_param'] = {
                    'size': [0.3, 0.3],
                    'pos': [-0.15, 0.0, 0.15],
                    'stiff': [2.0, 0.5, 1.0]
                }
                config['mass'] = 0.0054

        elif domain == 'ffn-rect-fabric':
           
            config['camera_params'] = {
                'default_camera':{
                    'pos': np.array([-0.0, 0.65, 0]),
                    'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                    'width': 720,
                    'height': 720},
            }
            config['cloth_param'] = {
                'size': [0.3, 0.2],
                'pos': [-0.15, 0.0, 0.15],
                'stiff': [2.0, 0.5, 1.0]
            }
            config['mass'] = 0.0054

        elif domain == 'ffnm-square-fabric' :
                config['camera_params'] = {
                    'default_camera':{
                        'pos': np.array([-0.0, 1, 0]),
                        'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                        'width': 720,
                        'height': 720},
                }
                config['cloth_param'] = {
                    'size': [0.3, 0.3],
                    'pos': [-0.15, 0.0, 0.15],
                    'stiff': [2.0, 0.5, 1.0]
                }
                config['mass'] = 0.0054
                
        elif domain == 'ffnm-rect-fabric':
           
            config['camera_params'] = {
                'default_camera':{
                    'pos': np.array([-0.0, 1, 0]),
                    'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                    'width': 720,
                    'height': 720},
            }
            config['cloth_param'] = {
                'size': [0.3, 0.2],
                'pos': [0, 0.0, 0],
                'stiff': [2.0, 0.5, 1.0]
            }
            config['mass'] = 0.0054
        
        elif domain == 'ffmr-rect-fabric':
            
            config['camera_params'] = {
                'default_camera':{
                    'pos': np.array([-0.0, 0.65, 0]),
                    'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                    'width': 720,
                    'height': 720},
            }
            config['cloth_param'] = {
                'size': [0.35, 0.25],
                'pos': [0, 0.0, 0],
                'stiff': [2.0, 0.5, 1.0]
            }
            config['mass'] = 0.0054

        elif domain == 'ffmr-square-fabric':
            
            config['camera_params'] = {
                'default_camera':{
                    'pos': np.array([-0.0, 0.65, 0]),
                    'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                    'width': 720,
                    'height': 720},
            }
            config['cloth_param'] = {
                'size': [0.35, 0.35],
                'pos': [-0.15, 0.0, 0.15],
                'stiff': [2.0, 0.5, 1.0]
            }
            config['mass'] = 0.0054
        
        elif domain in ['realadapt-towels', 'realadapt-towels-sq', 
                        'realadapt-rainbow-towels', 'realadapt-nomisgrasp-towels',
                        'realadapt-nomultilayer-towels']:
            
            config['camera_params'] = {
                'default_camera':{
                    'pos': np.array([-0.0, 0.65, 0]),
                    'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                    'width': 720,
                    'height': 720},
            }
            config['cloth_param'] = {
                'size': [0.3, 0.3],
                'pos': [-0.15, 0.0, 0.15],
                'stiff': [2.0, 0.5, 1.0]
            }
            config['mass'] = 0.0054
        
        
        return config