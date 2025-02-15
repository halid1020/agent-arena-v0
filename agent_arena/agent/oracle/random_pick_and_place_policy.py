import numpy as np

from agent_arena.agent.random_policy import RandomPolicy


class RandomPickAndPlacePolicy(RandomPolicy):

    def __init__(self,  config):
        super().__init__(config)
        # convert dotmap config to dict
        config = config.toDict()
        #self.hueristic_z = kwargs['hueristic_z'] if 'hueristic_z' in kwargs else False
        self.heuristic_pick_z= config['heuristic_pick_z'] if 'heuristic_pick_z' in config else False
        self.heuristic_place_z= config['heuristic_place_z'] if 'heuristic_place_z' in config else False
        self.pick_offset = config['pick_offset'] if 'pick_offset' in config else 0.0
        self.camera_height = 1.5 # TODO: this is a hack, should be read from environment
        self.pick_action_noise = config['pick_action_noise'] if 'pick_action_noise' in config else 0.0
        self.place_action_noise = config['place_action_noise'] if 'place_action_noise' in config else 0.0
        self.drag_action_noise = config['drag_action_noise'] if 'drag_action_noise' in config else 0.0
        self.action_types = ['random_pick_and_place']
        self.action_dim = config['action_dim']
        # self.last_info = None
    


    def act(self, info_list, update=False):
        
        if update:
            actions = [self.internal_states[info['arena_id']]['last_action'] for info in info_list]
            self.update(info_list, actions)
        
        ret_action = []
        for info in info_list:
            arena_id = info['arena_id']
            action = super().act(info)
            arena = info['arena']
            self.action_space = info['action_space']
            action = self.action_noise(action)
            action = self.hueristic_z(arena, action.copy()).reshape(*self.action_dim)
            ret_action.append(action)
            self.internal_states[arena_id]['last_action'] = action
        return ret_action
    
    def hueristic_z(self, environment, action):
        N = self.action_dim[0]
        #print('self.action_dim', self.action_dim)
        action = action.copy().reshape(N, 2, -1)
        res_action = np.zeros(tuple(self.action_dim)).astype(np.float32)
        res_action = res_action.reshape(N, 2, -1)
        res_action[:, :, :2] = action[:, :, :2]
    
        if self.heuristic_pick_z:
            res_action[:, 0, -1] = self.camera_height + self.pick_offset - self.get_depth(environment, action[:, 0, :2].copy())
            
        if self.heuristic_pick_z:
            res_action[:, 1, -1] = self.camera_height + self.pick_offset - self.get_depth(environment, action[:, 1, :2].copy())
        
        res_action = res_action.reshape(N, -1)

        res_action = np.clip(res_action, 
                             self.action_space.low, 
                             self.action_space.high)
        return res_action.copy()

    ## Shape has to be (N, 4/5/6)
    def action_noise(self, action, noise=True):
        if not noise:
            return action.copy()
        #print('here noisy', self.pick_action_noise, self.place_action_noise, self.drag_action_noise)

        N = self.action_dim[0]
        action = action.copy().reshape(N, 2, -1)
        noisy_action = action[:, :, :2].copy().reshape(N, -1)

        drag_vector = noisy_action[:, 2:] - noisy_action[:, :2]
        noise_drag_vector = drag_vector * (1 + np.random.normal(0, self.drag_action_noise, size=drag_vector.shape))
        noisy_action[:, 2:] = noisy_action[:, :2] + noise_drag_vector

        noisy_action[:, :2] += np.random.normal(0, self.pick_action_noise, size=noisy_action[:, :2].shape)
        noisy_action[:, 2:] += np.random.normal(0, self.place_action_noise, size=noisy_action[:, 2:].shape)

        action[:, 0, :2] = noisy_action[:, :2]
        action[:, 1, :2] = noisy_action[:, 2:]
        action = action.reshape(N, -1)

        return action.copy()

    def get_depth(self, environment, projected_positions):
        H, W = environment.observation_shape()['depth'][:2]
        depth = environment.render(mode='d', resolution=(H, W))

        projected_positions[:, 0] = (projected_positions[:, 0] + 1)/2*H
        projected_positions[:, 1] = (projected_positions[:, 1] + 1)/2*W
        projected_positions = projected_positions.astype(np.int32)
        projected_positions[:, 0] = np.clip(projected_positions[:, 0], 0, H-1)
        projected_positions[:, 1] = np.clip(projected_positions[:, 1], 0, W-1)
        
        res = np.asarray([depth[p[1], p[0]] for p in projected_positions])

        return res

    def get_action_type(self):
        return 'random_pick_and_place'
    

    def get_name(self):
        return 'Random Pick and Place'
    
    def init(self, info_list):
        for info in info_list:
            arena_id = info['arena_id']
            self.internal_states[arena_id]['last_info'] = info

    def update(self, info_list, actions):
        for i, info in enumerate(info_list):
            arena_id = info['arena_id']
            self.internal_states[arena_id]['last_info'] = info

    def success(self):
        return False

    def reset(self, arena_ids):
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {'last_info': None}