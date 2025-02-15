import numpy as np
import random
import logging

from agent_arena.agent.utilities.utils import *
from agent_arena.agent.oracle.rect_fabric.oracle_towel_pnp_flattening \
    import OracleTowelPnPFlattening
from agent_arena.agent.oracle.rect_fabric.realadapt_pnp_flattening \
    import RealAdaptPnPFlattening
from agent_arena.arena.softgym.tasks.rect_fabric.towel_flattening \
    import TowelFlatteningTask


    
class RectFabricMultiStepFoldingExpertPolicy(RealAdaptPnPFlattening):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
        self._train = kwargs['train'] if 'train' in kwargs else False
        self.current_action_type = ''

        ### Default is double cross folding
        self.folding_pick_order = kwargs['folding_pick_order'] if 'folding_pick_order' in kwargs else np.asarray([[[0, 0]], [[1, 0]]])
        self.folding_place_order = kwargs['folding_place_order'] if 'folding_place_order' in kwargs else np.asarray([[[1, 1]], [[0, 1]]])
        #self.folding_scale_range = kwargs['folding_scale_range'] if 'folding_scale_range' in kwargs else [(1.0, 1.0), (1.0, 1.0)]
        self.flatten_threshold = kwargs['flatten_threshold'] if 'flatten_threshold' in kwargs else 0.96
        self.fold_steps = -1
        #self.over_ratio_ = -0.02

        self.random_folding_steps = kwargs['random_folding_steps'] if 'random_folding_steps' in kwargs else False
        print('random_folding_steps', self.random_folding_steps)
        if self.random_folding_steps:
            print('random_folding_steps', self.random_folding_steps)
            self.action_types.append('random_folding')
            self.pick_corner = kwargs['pick_corner'] if 'pick_corner' in kwargs else False

        self.flatten_noise = kwargs['flatten_noise'] if 'flatten_noise' in kwargs else False
        self.folding_noise = kwargs['folding_noise'] if 'folding_noise' in kwargs else False
        self.next_step_threshold = 0.06
        self._reset()
        self.phase = 'folding'
        self.action_type = 'folding'
    
    def get_phase(self):
        return self.phase
        
    
    def reset(self):
        super().reset()
        print('resetting folding policy')
        self.fold_steps = -1
        self.current_action_type = ''
        self.is_success = False
        if self.random_folding_steps:
            print('generate random folding steps !')
            steps = random.randint(*self.random_folding_steps)
            self.folding_pick_order = np.random.rand(steps, 1, 2)
            if self.pick_corner:
                self.folding_pick_order = np.random.randint(0, 2, (steps, 1, 2))
            self.folding_place_order = np.random.rand(steps, 1, 2)
            self.over_ratios = np.zeros((steps, )).astype(np.float32)
        
        # if info is not None:
        #     self.init_state(info)

    def finsihed(self):
        return self.fold_steps >= len(self.folding_pick_order)
        
    def _is_folding_case(self, arena):
        if self.fold_steps >= 0:
            return True
        return TowelFlatteningTask.success(arena)
    
    def _get_pick_and_place_particle_ids(self, arena):
        
        cloth_H, cloth_W = arena.get_cloth_size()

        pick_pixel_pos = self.folding_pick_order[self.fold_steps].copy()
        place_pixel_pos = self.folding_place_order[self.fold_steps].copy()
        #print('pick_pixel_pos', pick_pixel_pos)

        # logging.debug('[oracle, multi-step fold] self.folding_pick_order {}'.format(self.folding_pick_order))
        # logging.debug('[oracle, multi-step fold] pick_pixel_pos {}'.format(pick_pixel_pos))
        # logging.debug('[oracle, multi-step fold] place_pixel_pos {}'.format(place_pixel_pos))

        print('pick_pixel_pos', pick_pixel_pos)
        print('place_pixel_pos', place_pixel_pos)
        
        pick_pixel_pos[:, 0] *= (cloth_H-1)
        pick_pixel_pos[:, 1] *= (cloth_W-1)

        place_pixel_pos[:, 0] *= (cloth_H-1)
        place_pixel_pos[:, 1] *= (cloth_W-1)

           

        pick_particle_ids = pick_pixel_pos[:, 0].astype(np.int32) * cloth_W \
            + pick_pixel_pos[:, 1].astype(np.int32)
        place_particle_ids = place_pixel_pos[:, 0].astype(np.int32) * cloth_W \
            + place_pixel_pos[:, 1].astype(np.int32)
        
        ## remove if ids are negative
        logging.debug('[oracle, multi-step fold]  pick_particle_ids {} shape {} before filtering'\
                          .format(pick_particle_ids, pick_particle_ids.shape))
        pick_particle_ids = pick_particle_ids[pick_particle_ids >= 0]
        place_particle_ids = place_particle_ids[place_particle_ids >= 0]
        
        return pick_particle_ids, place_particle_ids
    
    def terminate(self):
        return self.fold_steps >= len(self.folding_pick_order)
    
    def act(self, info):
        arena = info['arena']
        action = np.clip(
                    self.no_op.astype(float).reshape(*self.action_dim), 
                    self.action_space.low,
                    self.action_space.high)\
        .reshape(self.action_dim[0], 2, -1)[:, :, :2]\
        .reshape(self.action_dim[0], -1)

        ## No-op
        if self.fold_steps >= len(self.folding_pick_order):
            print('no-oppping')
            self.is_success = True
            logging.debug('case fold no-op')
            self.phase = 'success'
            return {'norm_pixel_pick_and_place': np.clip(
                    self.no_op.astype(float).reshape(*self.action_dim), 
                    self.action_space.low,
                    self.action_space.high)}
        
        ### Folding
        logging.debug('[oracle, multi-step fold] self.fold_steps {}'.format(self.fold_steps))
        logging.debug('[oracle, multi-step fold]  normalised_coverage {}'.format(info['normalised_coverage']))
        if self._is_folding_case(arena):
            self.phase = "folding"
            logging.debug('[oracle, multi-step fold]  case fold')
            if self.fold_steps < 0:
                self.fold_steps = 0

            
            particles = arena.get_object_positions()
            _, projected_positions = arena.get_visibility(
                    particles, 
                    cameras="default_camera")
            projected_positions = projected_positions[0]
            

           
            
            pick_particle_ids, place_particle_ids = self._get_pick_and_place_particle_ids(arena)
            shape_0 = pick_particle_ids.shape[0]
            logging.debug('[oracle, multi-step fold]  pick_particle_ids {} shape {}'\
                          .format(pick_particle_ids, pick_particle_ids.shape))
            over_ratio = self.over_ratios[self.fold_steps]

            ## Check if the last folding is sucessful
            
            if np.linalg.norm(projected_positions[pick_particle_ids] - projected_positions[place_particle_ids], axis=1).max() < self.next_step_thresholds[self.fold_steps]\
                or self.random_folding_steps:

                self.fold_steps += 1
                

                if self.fold_steps >= len(self.folding_pick_order):
                    logging.debug('[oracle, multi-step fold] case fold no-op')
                    self.phase = 'success'
                    return  {'norm_pixel_pick_and_place': \
                                np.clip(self.no_op.astype(np.float32).\
                                reshape(*self.action_dim), self.action_space.low, self.action_space.high)}
                
                pick_particle_ids, place_particle_ids = self._get_pick_and_place_particle_ids(arena)
                shape_0 = pick_particle_ids.shape[0]
                over_ratio = self.over_ratios[self.fold_steps]

            
            elif self.fold_steps > 0:
                over_ratio += 0.1


            action[:shape_0, :2] = projected_positions[pick_particle_ids]        
            action[:shape_0, 2:] = projected_positions[pick_particle_ids]  + \
                (1 + over_ratio)*(projected_positions[place_particle_ids] - projected_positions[pick_particle_ids])
            

            action = self.action_noise(action.copy(), noise= self.folding_noise)

            return {'norm_pixel_pick_and_place': self.hueristic_z(arena, action.copy()).reshape(*self.action_dim)}

        ### Flattening
        logging.debug('[oracle, multi-step fold] case flatten')
        flatten_action = super().act(info)
        action[:1] = flatten_action
        self.phase = "flattening"

        return {
            'norm_pixel_pick_and_place': action.reshape(*self.action_dim)
        }
        
    
    # def get_action_type(self):
    #     return self.phase