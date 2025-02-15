import os

from dotmap import DotMap
from ..envs.ur3e_one_picker_pick_and_place_env import *
from ..tasks.garments.garment_flattening import GarmentFlatteningTask
## action:pixel-pick-and-place, initial:flatten, domain:mono-single-tshirt, task:flattening

class UR3eDomainBuilder():
    
        
    def build_from_config(domain, task, disp, horizon=8, rayid=0, ray=False):
        object = domain.split('-')[-1]
        

        #TODO: set the correct config
        config = {
            'object': object,
            'picker_radius': 0.015, #0.015,
            'particle_radius': 0.00625,
            'picker_threshold': 0.007, # 0.05,
            'picker_low': (-5, 0, -5),
            'picker_high': (5, 5, 5),
            'grasp_mode': {'closest': 1.0},
            'init_state_path': os.path.join(
                os.environ['AGENT_ARENA_PATH'], '..', 'data', 'cloth_funnel', 'init_states'),
            'task': task,
            'disp': True if disp == 'True' else False,
            'ray_id': rayid,
            'horizon': int(horizon),
        }
        
        config['grasp_mode'] = {
            'around': 0.9,
            'miss': 0.1
        }

        config = DotMap(config)
        task = GarmentFlatteningTask()
        if ray:
            env = UR3eOnePickerPickAndPlaceEnvRay.remote(config)
            env.set_task.remote(task)
        else:
            env = UR3eOnePickerPickAndPlaceEnv(config)
            env.set_task(task)
        return env