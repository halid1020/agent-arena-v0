import numpy as np
import random 
from scipy.spatial import ConvexHull #, convex_hull_plot_2d
import logging
import gym

from ...oracle.random_pick_and_place_policy import RandomPickAndPlacePolicy
from agent_arena.agent.utilities.utils import *


class OracleTowelPnPFlattening(RandomPickAndPlacePolicy):
    """
    Always return one picker pick and place position, [1, 4]
    """
    def __init__(self, config):
        super().__init__(config)
        #self.action_dim = self.action_space.shape
        #print('hello self.action_dim', self.action_dim)
        self.boudary_2 = 0.9
        self.stop_threshold = 0.99
        config = config.toDict()
        self.no_op = np.asarray(config['no_op'])
        self.canonical = config['canonical'] if 'canonical' in config else False
        #print('canonical', self.canonical)
        if self.canonical:
            self.IoU_threshold = 0.9
        
        self.action_types = [
            'no-op',
            'drag-to-opposite-side',
            'reveal-hidden-pick_cornercorner',
            'untwist',
            'flatten',
            'reveal-edge-point']
        
        self.current_action_type = ''
        self.search_range = 0.5
        self.camera_height = 1.5
        self.action_space = gym.spaces.Box(
            low=np.asarray(config['action_low']), 
            high=np.asarray(config['action_high']), 
            shape=tuple(config['action_dim']), dtype=np.float64)

        self.over_ratio = 1.08
        self.search_interval = 0.01
        self.pick_z = config['pick_z'] if 'pick_z' in config else False
        self.place_z = config['pick_z'] if 'place_z' in config else False

        
        # if self.heuristic_pick_z and self.heuristic_place_z:
        #     self.no_op = np.asarray([1, 1, 0, 1, 1, 0])

        if self.pick_z:
            self.pick_offset = config['pick_offset'] if 'pick_offset' in config else 0.02

        if self.place_z:
            self.place_height = config['place_height'] if 'place_height' in config else 0.06
    
        self.verbose = config['verbose'] if 'verbose' in config else False
        # self.internal_states[arena_id]['revealing_corner']= False
        # self.internal_states[arena_id]['revealing_corner_id'] = None
        # self.internal_states[arena_id]['last_corner_id']= None
        # #self.last_hidden_points = []
        
        #self.flatten_threshold = kwargs['flatten_threshold'] if 'flatten_threshold' in kwargs else 0.99
        self.flatten_noise = config['flatten_noise'] if 'flatten_noise' in config else False
        if self.flatten_noise:
            #print('hello')
            self.action_types.extend(['noisy-' + n for n in self.action_types])

    def get_name(self):
        return 'Oracle Rectangular Fabric Pick and Place Expert Policy'

    def reset(self, arena_ids):
        super().reset(arena_ids)

        for arena_id in arena_ids:
            self.internal_states[arena_id]['last_action'] = None
            self.internal_states[arena_id]['revealing_corner'] = False
            self.internal_states[arena_id]['revealing_corner_id'] = None
            self.internal_states[arena_id]['last_corner_id'] = None
            self.internal_states[arena_id]['is_success'] = False
            if self.canonical:
                self.internal_states[arena_id]['to_canonical'] = False

        # self.internal_states[arena_id]['last_action'] = None
        # self.internal_states['revealing_corner']= False
        # self.internal_states['revealing_corner_id'] = None
        # self.internal_states['last_corner_id ']= None
        # #self.last_hidden_points = []
        # self.internal_states['is_success '] = False
        # if self.canonical:
        #     self.to_canonical = False

    def _search_best_pairs(self, corner_positions_2d, flatten_corner_world_positions_2d):
        N = corner_positions_2d.shape[0]

        #logging.debug('[oracle,rect-fabric,flattening, expert] flatten pos {}'.format(flatten_corner_world_positions_2d))

        min_sum_distance = 1000
        
        pick_corner_order = [0, 1, 3, 2]
        place_corner_order = [0, 1, 3, 2]

        for j in range(0, N):
            flatten_indexes = [place_corner_order[(i+j)%N] for i in range(N)]
            distance = np.linalg.norm(corner_positions_2d[pick_corner_order] - flatten_corner_world_positions_2d[flatten_indexes], axis=1)

            
            pairs = {pick_corner_order[i]: (place_corner_order[(i+j)%N], distance[i]) for i in range(N)}
            sum_distance = distance.sum()

            if sum_distance < min_sum_distance:
                min_sum_distance = sum_distance
                best_pairs = pairs

        
        place_corner_order = [1, 0, 2, 3]
        for j in range(0, N):
            flatten_indexes = [place_corner_order[(i+j)%N] for i in range(N)]
            distance = np.linalg.norm(corner_positions_2d[pick_corner_order] - flatten_corner_world_positions_2d[flatten_indexes], axis=1)

            
            pairs = {pick_corner_order[i]: (place_corner_order[(i+j)%N], distance[i]) for i in range(N)}
            sum_distance = distance.sum()

            if sum_distance < min_sum_distance:
                min_sum_distance = sum_distance
                best_pairs = pairs
            
        return best_pairs, min_sum_distance


    def _sweep_to_find_best_pairs(
            self,
            corner_world_positions_2d, 
            flatten_corner_world_positions_2d, 
            flatten_edge_world_positions_2d):
        
        min_sum_distance = 1000

        for dx in np.arange(-self.search_range, self.search_range+self.search_interval, self.search_interval):
                for dy in np.arange(-self.search_range, self.search_range+self.search_interval, self.search_interval):
                    target_corner_positions_2d_tmp = flatten_corner_world_positions_2d.copy()
                    target_edge_positions_2d_tmp = flatten_edge_world_positions_2d.copy()
                    target_corner_positions_2d_tmp[:, 0] += dx
                    target_corner_positions_2d_tmp[:, 1] += dy
                    target_edge_positions_2d_tmp[:, 0] += dx
                    target_edge_positions_2d_tmp[:, 1] += dy
                    pairs, sum_d = self._search_best_pairs(corner_world_positions_2d, target_corner_positions_2d_tmp)
                
                    
                    if sum_d < min_sum_distance:
                        min_sum_distance = sum_d
                        target_place_projected_corner_positions = \
                            (target_corner_positions_2d_tmp.copy()/(self.camera_height*self.camera_to_world)) 
                        
                        target_place_projected_edge_positions = \
                            (target_edge_positions_2d_tmp.copy()/(self.camera_height*self.camera_to_world))
                        
                        best_pairs = pairs.copy()
        
        return best_pairs, min_sum_distance, \
            target_place_projected_corner_positions, target_place_projected_edge_positions


    def _get_pairs(self, wcp_2d, fp_wcp_2d, fp_wep_2d, canonical=False):

        
        pairs_0, sum_distance_0 = self._search_best_pairs(
            wcp_2d, 
            fp_wcp_2d)

        ## rotation flatten_corner and edge_world_positions_2d around the center for 90 degree
        logging.debug('before rotation {}'.format(fp_wcp_2d))
        rfp_wcp_2d = np.zeros_like(fp_wcp_2d)
        rfp_wcp_2d[:, 0] = fp_wcp_2d[:, 1]
        rfp_wcp_2d[:, 1] = -fp_wcp_2d[:, 0]

        rfp_wep_2d = np.zeros_like(fp_wep_2d)
        rfp_wep_2d[:, 0] = fp_wep_2d[:, 1]
        rfp_wep_2d[:, 1] = -fp_wep_2d[:, 0]
        logging.debug('after rotation {}'.format(rfp_wcp_2d))
        
        pairs_1, sum_disntace_1 = self._search_best_pairs(
            wcp_2d, 
            rfp_wcp_2d)
        
        best_pairs = pairs_0.copy()
        smallest_sum_distance = sum_distance_0
        best_fp_pcp = \
            (fp_wcp_2d.copy()/(self.camera_height*self.camera_to_world))
        best_fp_pep = \
            (fp_wep_2d.copy()/(self.camera_height*self.camera_to_world))

        if sum_disntace_1 < sum_distance_0:
            best_pairs = pairs_1.copy()
            smallest_sum_distance = sum_disntace_1
            best_fp_pcp = \
                (rfp_wcp_2d.copy()/(self.camera_height*self.camera_to_world))
            best_fp_pep = \
                (rfp_wep_2d.copy()/(self.camera_height*self.camera_to_world))

        logging.debug('[oracle,rect-fabric,flattening, expert] best_pairs {}'.format(pairs_0))

        if canonical:
            return best_pairs, best_fp_pcp, best_fp_pep

        pairs_2, sum_distance_2, place_projected_corner_positions_2, place_projected_edge_positions_2 = \
            self._sweep_to_find_best_pairs(
                wcp_2d, 
                fp_wcp_2d, 
                fp_wep_2d)
        
        if sum_distance_2 < smallest_sum_distance:
            best_pairs = pairs_2.copy()
            smallest_sum_distance = sum_distance_2
            best_fp_pcp = place_projected_corner_positions_2.copy()
            best_fp_pep = place_projected_edge_positions_2.copy()
        
        pairs_3, sum_distance_3, place_projected_corner_positions_3, place_projected_edge_positions_3 = \
            self._sweep_to_find_best_pairs(
                wcp_2d, 
                rfp_wcp_2d, 
                rfp_wep_2d)
        
        if sum_distance_3 < smallest_sum_distance:
            best_pairs = pairs_3.copy()
            smallest_sum_distance = sum_distance_3
            best_fp_pcp = place_projected_corner_positions_3.copy()
            best_fp_pep = place_projected_edge_positions_3.copy()
          
            
        logging.debug('[oracle,rect-fabric,flattening, expert] not canniclisation, new best_pairs {}'.format(best_pairs))
        
        return best_pairs, best_fp_pcp, best_fp_pep
    
    
    def is_twist(self, corner_positions_2d, environment):
        corner_id_to_particle_id = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1)
        }

        hull = ConvexHull(corner_positions_2d)
        valid_edges = [
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (1, 3),
            (3, 1),
            (2, 3),
            (3, 2)
        ]
        if len(hull.simplices) < 4:
            return False, []

        for s in hull.simplices:
            if ((s[0], s[1]) not in valid_edges) and  ((s[1], s[0]) not in valid_edges):
                p0, p1 = s[0], s[1]
                p2 = 0
                for i in range(4):
                    if i != p0 and i != p1 and (((p0, p2) in hull.simplices) or ((p2, p0) in hull.simplices)):
                        p2 = i
                        break
                p3 = 6 - p0 - p1 - p2
                
                #print('p0, p1, p2, p3', p0, p1, p2, p3)

                ##################################
                ## If mid of p1 and p2 is higher than mid of p0 and p3
                ## Swap p0 to p1, p3 to p2
                particles = environment.get_object_positions()
                cloth_H, cloth_W = environment.get_cloth_size()
                # print('particles', particles.shape)
                # print('cloth_H, cloth_W', cloth_H, cloth_W)

                mid_p1_p2_x = (corner_id_to_particle_id[p1][0] + corner_id_to_particle_id[p2][0])/2.0
                mid_p1_p2_y = (corner_id_to_particle_id[p1][1] + corner_id_to_particle_id[p2][1])/2.0
                mid_p1_p2_x, mid_p1_p2_y = int(mid_p1_p2_x * (cloth_H-1)), int(mid_p1_p2_y * (cloth_W-1))
                mid_p1_p2_id = mid_p1_p2_x * cloth_W + mid_p1_p2_y      

                mid_p0_p3_x = (corner_id_to_particle_id[p0][0] + corner_id_to_particle_id[p3][0])/2.0
                mid_p0_p3_y = (corner_id_to_particle_id[p0][1] + corner_id_to_particle_id[p3][1])/2.0
                mid_p0_p3_x, mid_p0_p3_y = int(mid_p0_p3_x * (cloth_H-1)), int(mid_p0_p3_y * (cloth_W-1))
                mid_p0_p3_id = mid_p0_p3_x * cloth_W + mid_p0_p3_y

                # print('mid_p1_p2', particles[mid_p1_p2_id])
                # print('mid_p0_p3', particles[mid_p0_p3_id])

                if particles[mid_p1_p2_id][1] > particles[mid_p0_p3_id][1]:
                    p0, p1, p2, p3 = p1, p0, p3, p2


                ##################################
                if ((p2, p0) in valid_edges):
                    p1, p2 = p2, p1

                #print('after: p0, p1, p2, p3', p0, p1, p2, p3)
                return True, [p0, p1, p2, p3]
        if len(hull.simplices) < 4:
            return False, []

        return False, []
    
    def _reveal_hidden_corner(self, 
            arena, 
            action, 
            projected_corner_positions, 
            corner_id,
            pairs, 
            flatten_place_projected_corner_positions):
        arena_id = arena.id
        h, w = arena.get_cloth_dim()
        ## Scale ishalf of the cloth diagonal length
        scale = np.sqrt(h**2 + w**2)/2.0
    

        cloth_mask = arena._get_cloth_mask(resolution=(128, 128)).T
        H, W = cloth_mask.shape

        cloth_countor = np.zeros_like(cloth_mask)
        for i in range(1, cloth_mask.shape[0]-1):
            for j in range(1, cloth_mask.shape[1]-1):
                if cloth_mask[i, j] == 1:
                    if np.sum(cloth_mask[i-1:i+2, j-1:j+2]) == 9:
                        cloth_countor[i, j] = 0
                    else:
                        cloth_countor[i, j] = 1

                
        hidden_pos = projected_corner_positions[corner_id]
        hidden_pixel_pos = (hidden_pos + 1)/2*H

        target_pos = flatten_place_projected_corner_positions[pairs[corner_id][0]]
        target_pixel_pos = (target_pos + 1)/2*W
        
        
        xx = 0
        yy = 0
        min_dis = 1000
        for x in range(H):
            for y in range(W):
                if cloth_countor[x][y]:
                    dis_1 = np.linalg.norm(np.asarray([x,y]) - target_pixel_pos)
                    dis_2 = np.linalg.norm(np.asarray([x,y]) - hidden_pixel_pos)
                    
                    if dis_1 + dis_2 < min_dis:
                        min_dis = dis_1 + dis_2 
                        xx = x
                        yy = y
        
        action[0, 0] = xx/H * 2 - 1
        action[0, 1] = yy/W * 2 - 1

        displancement = action[0, :2] - hidden_pos
        length = np.linalg.norm(hidden_pos - action[0, :2]) * 1.3
        normaliesd_reverser_displacement = -displancement / np.sqrt(np.sum(displancement**2))
        action[0, 2:] = action[0, :2] + normaliesd_reverser_displacement * max(scale, length)

        self.internal_states[arena_id]['revealing_corner']= True
        if corner_id == self.internal_states[arena_id]['revealing_corner_id']:
            self.revealing_count += 1
        self.internal_states[arena_id]['revealing_corner_id'] = corner_id
        self.action_type = 'reveal-hidden-corner'

        return action
    
    def _revealAnEdgePointWhileHighCoverage(self,
            arena,
            action,
            hidden_edge_points, 
            pep,
            fp_pep):
        
        h, w = arena.get_cloth_dim()
        ## Scale ishalf of the cloth diagonal length
        scale = np.sqrt(h**2 + w**2)/2.0 * 0.8
        #random.shuffle(hidden_edge_points)

    
        cloth_countor = self._get_cloth_contour(arena)
        H, W = cloth_countor.shape
        
        max_min_dis = -1    
        for hep in hidden_edge_points:
            hidden_pos = pep[hep]
            hidden_pixel_pos = (hidden_pos + 1)/2*H
        
        
            xx = 0
            yy = 0
            min_dis = 1000
            for x in range(H):
                for y in range(W):
                    if cloth_countor[x][y]:
                        dis = np.linalg.norm(np.asarray([x,y]) - hidden_pixel_pos)
                        if dis < min_dis:
                            min_dis = dis
                            xx = x
                            yy = y
                            tmp_hidden_pos = hidden_pos

            if min_dis > max_min_dis:
                max_min_dis = min_dis
                target_hidden_pos = tmp_hidden_pos
                edge_point_id = hep
                action[0, 0] = xx/H * 2 - 1
                action[0, 1] = yy/W * 2 - 1

        displancement = action[0, :2] - target_hidden_pos
        normaliesd_reverser_displacement = -displancement / np.sqrt(np.sum(displancement**2))
        action[0, 2:] = action[0, :2] + normaliesd_reverser_displacement * scale

        self.action_type = 'reveal-edge-point'
        return action

    
    def _get_opposite_side_action(self,
            arena,
            action,
            far=True):
        
        h, w = arena.get_cloth_dim()
        ## Scale ishalf of the cloth diagonal length
        scale = np.sqrt(h**2 + w**2)/2.0 
        cloth_countor = self._get_cloth_contour(arena)
        
        ## Choose the point on the contour that is farthest from the center
        H, W = cloth_countor.shape
        center = np.asarray([H/2, W/2])
        
        max_d = -1
        min_d = 1000

        for x in range(H):
            for y in range(W):
                if cloth_countor[x][y]:
                    dis = np.linalg.norm(np.asarray([x,y]) - center)
                    if dis > max_d:
                        max_d = dis
                        xx = x
                        yy = y
                    elif dis < min_d:
                        min_d = dis
                        mxx = x
                        myy = y
        if far:
            action[0, 0] = xx/H * 2 - 1
            action[0, 1] = yy/W * 2 - 1
            action[0, 2:] = np.sign(action[0, :2])*-1 * scale * 1.4
        else:
            futhest_from_center = [xx/H * 2 - 1, yy/W * 2 - 1]
            action[0, 0] = mxx/H * 2 - 1
            action[0, 1] = myy/W * 2 - 1
            action[0, 2:] = np.sign(futhest_from_center)*-1 * scale
        
        # clip the place action between -0.9 and 0.9
        action[0, 2:] = np.clip(action[0, 2:], -0.9, 0.9)

        self.action_type = 'drag-to-opposite-side'

        logging.debug('[oracle,rect-fabric,flattening, expert] generate opposite side action {}'.format(action))
        return action
    
    def flatten(self, info):
        return  info['normalised_coverage'] >= 0.999    
    

    def _success(self, info):
        return np.all(info['corner_visibility']) and self.flatten(info)

    def success(self):
        return {k: self._success(v['last_info'])
                for k, v in self.internal_states.items()}
    
    def _reset_revealing_corner(self, arena_id):
        self.internal_states[arena_id]['revealing_corner']= False
        self.internal_states[arena_id]['revealing_corner_id'] = None
        self.revealing_count = 0

    def _get_flatten_action(self, arena_id, action, pairs, hidden_corner_points, 
            projected_corner_positions, 
            flatten_place_projected_corner_positions):

        #action = np.ones((1, 4))
        max_d = 0
        flatten_pairs = pairs
        for x, (y, d) in flatten_pairs.items():
            if x not in hidden_corner_points:
                if d > max_d - 0.01:
                    max_d = d
                    self.internal_states[arena_id]['last_corner_id ']= x
                    action[0, :2] = projected_corner_positions[x]
                    action[0, 2:] = flatten_place_projected_corner_positions[y]
        
        return action
                   
    def terminate(self):
        return {k: v['is_success'] for k, v in self.internal_states.items()}

    def act(self, info_list):
        ret_actions = []
        for info in info_list:
            ret_actions.append(self.single_act(info))
        return ret_actions
    
    def single_act(self, info):
        ## pcp: projected_corner_positions
        ## fp_pcp: flatten_place_projected_corner_positions
        ## fp_pep: flatten_place_projected_edge_positions
        ## cfp_pcp: canonical_flatten_place_projected_corner_positions
        ## cfp_pep: canonical_flatten_place_projected_edge_positions

        arena = info['arena']
        arena_id = info['arena_id']

        self.camera_to_world = arena.pixel_to_world_ratio
        self.camera_height = arena.camera_height
        
        # Case 1: When succesfully flattened, do nothing
        action = np.clip(
                    self.no_op.astype(float).reshape(*self.action_dim), 
                    self.action_space.low,
                    self.action_space.high)\
        .reshape(self.action_dim[0], 2, -1)[:, :, :2]\
        .reshape(self.action_dim[0], -1)
        
        
        if self._success(info):
            self.internal_states[arena_id]['is_success '] = True
            #logging.debug('[oracle,rect-fabric,flattening, expert] Case flattening: no-op')
            self.internal_states[arena_id]['action_type'] = 'flatten'
            
            action = np.clip(
                    self.no_op.astype(float).reshape(*self.action_dim), 
                    self.action_space.low,
                    self.action_space.high)
            return self._process_action(action, arena, noise=self.flatten_noise)
        
        # Process world, pixel position as well as visibility of corners and edges
        corner_world_positions = arena.get_corner_positions()
        wep = arena.get_edge_positions()        
        wcp_2d = corner_world_positions[:, [0, 2]]
        wep_2d = wep[:, [0, 2]]
        visible_corner_positions, pcp = arena._get_visibility(
                corner_world_positions,
                resolution=(128, 128))
        visible_edge_positions, pep = arena._get_visibility(
                wep,
                resolution=(128, 128))
        visible_corner_positions = visible_corner_positions[0]
        pcp = pcp[0]
        visible_edge_positions = visible_edge_positions[0]
        pep = pep[0]

        valid_corner_points = [] ## visible and inside the boundary
        hidden_corner_points = []
        out_corner_points = []
        hidden_edge_points = []
        for idx, (vis, target_pos) in enumerate(zip(visible_corner_positions, pcp)):    
            if vis and -self.boudary_2  < min(target_pos) and max(target_pos) < self.boudary_2 :
                valid_corner_points.append(idx)
            elif -self.boudary_2 > min(target_pos) or max(target_pos) > self.boudary_2 :
                out_corner_points.append(idx)        
            if not vis:
               hidden_corner_points.append(idx)
        for idx, (vis, target_pos) in enumerate(zip(visible_edge_positions, pep)):
            if not vis:
                hidden_edge_points.append(idx)

        random.shuffle(hidden_corner_points)

        # logging.debug('[oracle,rect-fabric,flattening, expert] hidden corner num {}: {}'\
        #               .format(len(hidden_corner_points), hidden_corner_points))
        # logging.debug('[oracle,rect-fabric,flattening, expert] hidden edge num {}'\
        #               .format(len(hidden_edge_points)))

        # Get Canonicalised corner and edge positions
        cf_wcp = arena.get_flatten_corner_positions()
        cf_wcp_2d = cf_wcp[:, [0, 2]] * self.over_ratio
        cfp_pcp = (cf_wcp_2d/(self.camera_height*self.camera_to_world))
        pixel_area = abs(cfp_pcp[0][0] - cfp_pcp[-1][0]) * abs(cfp_pcp[0][1] - cfp_pcp[-1][1])
        sqrt_pixel_area = np.sqrt(pixel_area)
        
        cf_wep = arena.get_flatten_edge_positions()
        cf_wep_2d = cf_wep[:, [0, 2]] * self.over_ratio
        cfp_pep = (cf_wep_2d/(self.camera_height*self.camera_to_world))
   

        ## Get flattening pairs for the corners
        pairs, fp_pcp, fp_pep = self._get_pairs(
            wcp_2d.copy(), 
            cf_wcp_2d.copy(), 
            cf_wep_2d.copy(),
            canonical=False
        )

        canon_pairs, cfp_pcp, cfp_pep = self._get_pairs(
            wcp_2d.copy(), 
            cf_wcp_2d.copy(), 
            cf_wep_2d.copy(),
            canonical=True
        )

        #action = np.ones((1, 4))
        
        ## Case 1: If revealing corners in the last step but not successfully to do so.
        if self.internal_states[arena_id]['revealing_corner']\
            and (self.internal_states[arena_id]['revealing_corner_id'] in hidden_corner_points) \
            and self.revealing_count < 2:
            #logging.debug('[oracle,rect-fabric,flattening, expert] Case revealing corner')
            self.internal_states[arena_id]['last_corner_id']= self.internal_states[arena_id]['revealing_corner_id']
            action = self._reveal_hidden_corner(
                arena, action, 
                pcp, 
                self.internal_states[arena_id]['last_corner_id'],
                pairs, 
                fp_pcp)
            
            return self._process_action(action, arena, noise=self.flatten_noise)
        
        ## Case 2: If succesffully revealed a corner in the last step, and we need to flatten the corner
        if self.internal_states[arena_id]['revealing_corner']\
            and (self.internal_states[arena_id]['revealing_corner_id'] not in hidden_corner_points):
            self._reset_revealing_corner(arena_id)
            logging.debug('[oracle,rect-fabric,flattening, expert] Case revealed corner and flatten')
            self.action_type = 'flatten'
            action[0, :2] = pcp[self.internal_states[arena_id]['last_corner_id']]
            action[0, 2:] = fp_pcp[pairs[self.internal_states[arena_id]['last_corner_id']][0]]
            return self._process_action(action, arena, noise=self.flatten_noise)

        
        
        self._reset_revealing_corner(arena_id)
        
    
        ## Case 3: No valid points or there are out of boundary points.
        if len(valid_corner_points) == 0 \
            or len(out_corner_points) >= 3 \
            or arena._get_normalised_coverage() < 0.4:
            
            logging.debug('[oracle,rect-fabric,flattening, expert] Case to drag to the opposite side')
            action = self._get_opposite_side_action(
                arena,
                action)
                
            return self._process_action(action, arena, noise=self.flatten_noise)
    
        ## Newly added 08/04/2024, drag to the center
        if len(out_corner_points) >= 1:
            logging.debug('[oracle,rect-fabric,flattening, expert] Case to drag to the center')
            action = self._get_opposite_side_action(
                arena,
                action,
                far=False)
                
            return self._process_action(action, arena, noise=self.flatten_noise)

        
        ## Case 4: If there is a level-1 twist, then untwist
        twist, twist_ids = self.is_twist(wcp_2d, arena)
        if twist:
            logging.debug('[oracle,rect-fabric,flattening, expert] Case untwist') 
            self.action_type = 'untwist'
            self.internal_states[arena_id]['last_corner_id']= twist_ids[0]
            action[0, :2] = pcp[twist_ids[3]]

            displacement = pcp[twist_ids[2]] - pcp[twist_ids[3]]
            length = np.linalg.norm(displacement)
            
            # rotation displacement 90 degree
            rotated_displacement = np.zeros_like(displacement)
            rotated_displacement[0] = displacement[1]
            rotated_displacement[1] = -displacement[0]
            normaliesd_displacement = rotated_displacement / length

            action[0, 2:] = pcp[twist_ids[3]] + normaliesd_displacement * max(0.15, length)
            return self._process_action(action, arena, noise=self.flatten_noise)

        #print('canonical', self.canonical)
        action = self._get_flatten_action(
            arena_id,
            action,
            (pairs if not self.canonical else canon_pairs),
            hidden_corner_points, 
            pcp,
            (fp_pcp if not self.canonical else cfp_pcp), 
        )
        
        flatten_distance = np.linalg.norm(action[0, 2:] - action[0, :2])
        self.action_type = 'flatten'

        if flatten_distance > 0.4*sqrt_pixel_area:
            #logging.debug('[oracle,rect-fabric,flattening, expert] Case to effective flatten the cloth')
            return self._process_action(action, arena, noise=self.flatten_noise)

            
        ## Case 5: If there are more than two hidden points
        ## and there is no out of boundary points, and the coverage is high enough
        if len(hidden_corner_points) >= 2 and flatten_distance < 0.3*sqrt_pixel_area:
            self.action_type = 'reveal-hidden-corners'
            #logging.debug('[oracle,rect-fabric,flattening, expert] Case more than two hidden points, reveal hidden point')
            self.internal_states[arena_id]['last_corner_id']= hidden_corner_points[0]
            action = self._reveal_hidden_corner(
                arena, 
                action, 
                pcp, 
                hidden_corner_points[0],
                pairs, 
                fp_pcp)

            return self._process_action(action, arena, noise=self.flatten_noise)
            
        ## Case 6: If there is only one hidden point and the cloth is almost flatten, then reveal it
        if arena._get_normalised_coverage() > 0.5 \
            and len(hidden_corner_points) == 1 \
            and flatten_distance < 0.2*sqrt_pixel_area:
            logging.debug('[oracle,rect-fabric,flattening, expert] Case definit reveal the only hidden point')
            self.internal_states[arena_id]['last_corner_id']= hidden_corner_points[0]
            action = self._reveal_hidden_corner(
                arena, 
                action, 
                pcp, 
                hidden_corner_points[0],
                pairs, 
                fp_pcp)
            
            return self._process_action(action, arena, noise=self.flatten_noise)
        
        
        
        
        ## Case: Revealing edge points when the coverage is high enough, and flattening corner is trival
        if arena._get_normalised_coverage() > 0.85 \
            and len(hidden_edge_points) > 5 \
            and flatten_distance < 0.125*sqrt_pixel_area: #0.15 before
            
            #logging.debug('[oracle,rect-fabric,flattening, expert] Case to reveal the edge points')
            self.internal_states[arena_id]['last_corner_id']= None
            action = self._revealAnEdgePointWhileHighCoverage(
                arena,
                action,
                hidden_edge_points.copy(),
                pep.copy(),
                (fp_pep if not self.canonical else cfp_pep), 
            )
            return self._process_action(action, arena, noise=self.flatten_noise)
        

        ## Case 7: flatten the cloth
        #logging.debug('[oracle,rect-fabric,flattening, expert] Case to flatten the cloth')
        return self._process_action(action, arena, noise=self.flatten_noise)
    
    def _process_action(self, action, arena, noise=False):
        action = self.action_noise(action, noise=noise)
        length, width = arena.get_cloth_dim()
        a = length * width
        
        if noise:
            self.action_type = 'noise_' + self.action_type
        
        ### Newly added 08/04/2024
        if a >= 0.25:
            print('large area')
            action[0, 2:] = action[0, :2] + 1.2 * (action[0, 2:] - action[0, :2])
            action[0, 2:] = np.clip(action[0, 2:], -0.95, 0.95)
        action = self.hueristic_z(arena, action.copy()).reshape(*self.action_dim)[:1]
        
        return {
            'pick_0': action[0, :2][::-1],
            'place_0': action[0, 2:][::-1],
        }


    def _all_one_side(self, projected_corner_positions):
        ## The signs of x and y are consistent for all corners
        signs = np.sign(projected_corner_positions)
        return np.all(signs[0] == signs)
    
    def _get_cloth_contour(self, arena):
        cloth_mask = arena._get_cloth_mask(resolution=(128, 128)).T
        H, W = cloth_mask.shape

        cloth_countor = np.zeros_like(cloth_mask)
        for i in range(1, cloth_mask.shape[0]-1):
            for j in range(1, cloth_mask.shape[1]-1):
                if cloth_mask[i, j] == 1:
                    if np.sum(cloth_mask[i-1:i+2, j-1:j+2]) == 9:
                        cloth_countor[i, j] = 0
                    else:
                        cloth_countor[i, j] = 1
        return cloth_countor
    
    

    def get_action_type(self):
        return self.action_type