import os
import numpy as np
import cv2

from agent_arena import TrainableAgent
from agent_arena.utilities.utils import TrainWriter
import torch
from itertools import product

from .nets import MaximumValuePolicy
from .utils import prepare_image, generate_primitive_cloth_mask,\
    generate_workspace_mask, transform, get_transform_matrix
from agent_arena.utilities.torch_utils import np_to_ts
from agent_arena.utilities.logger.logger_interface import Logger

LOSS_NORMALIZATIONS = {
    'rigid':{
        'fling':{'manipulation':5, 'nocs':3}, 
        'place':{'manipulation':1.2, 'nocs':3}},
    'deformable':{
        'fling':{'manipulation':0.8, 'nocs':3}, 
        'place':{'manipulation':0.1, 'nocs':3}}
}

class ClothFunnel(TrainableAgent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'cloth-funnel'
        self.writer = TrainWriter()
        self.internal_states = {}

        self.device = config.device if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.logger = Logger()

        self._init_network()
        self._init_action_primitives()
    
    def _init_network(self):

        self.network = MaximumValuePolicy(
            action_expl_prob=self.config.action_expl_prob,
            action_expl_decay=self.config.action_expl_decay,
            value_expl_decay=self.config.value_expl_decay,
            value_expl_prob=self.config.value_expl_prob,
            action_primitives=list(self.config.action_primitives),
            num_rotations=self.config.num_rotations,
            scale_factors=list(self.config.scale_factors),
            obs_dim=self.config.obs_dim,
            pix_grasp_dist=self.config.pix_grasp_dist,
            pix_place_dist=self.config.pix_place_dist,
            pix_drag_dist=self.config.pix_drag_dist,
            deformable_weight=self.config.deformable_weight,
            network_gpu=self.config.network_gpu,
            input_channel_types=self.config.input_channel_types,
            gpu=0)

        self.optimiser = torch.optim.Adam(
            self.network.value_net.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        
    def _init_action_primitives(self):

        self.action_primitives = list(self.config.action_primitives)
        self.adaptive_scale_factors = list(self.config.scale_factors)
        self.rotations = np.linspace(-180, 180, self.config.num_rotations + 1)
        # print("ALL ROTATIONS", self.rotations)
        self.rotation_indices = {
            'fling': np.where(np.logical_and(self.rotations >= -90, self.rotations <= 90)),
            'place': np.where(np.logical_and(self.rotations >= -180, self.rotations <= 167.5)),
        }
        self.scale_factors = list(self.config.scale_factors)
        self.primitive_vmap_indices = {}
        for primitive, indices in self.rotation_indices.items():
            self.primitive_vmap_indices[primitive] = [None, None]
            self.primitive_vmap_indices[primitive][0] = indices[0] * len(self.scale_factors)
            self.primitive_vmap_indices[primitive][1] = (indices[-1]+1) * len(self.scale_factors)

        # physical limit of dual arm system, TODO: we need to decouple the following
        self.TABLE_WIDTH = 0.765 * 2
        self.left_arm_base = np.array([0.765, 0, 0])
        self.right_arm_base = np.array([-0.765, 0, 0])
        self.reach_distance_limit = self.config.reach_distance_limit

        render_dim = self.config.render_dim
        pix_radius = int(render_dim * (self.reach_distance_limit/self.TABLE_WIDTH))

        left_arm_reach = np.zeros((render_dim, render_dim))
        left_arm_reach = cv2.circle(left_arm_reach, (render_dim//2, 0), pix_radius, (255, 255, 255), -1)

        right_arm_reach = np.zeros((render_dim, render_dim))
        right_arm_reach = cv2.circle(right_arm_reach, (render_dim//2, render_dim), pix_radius, (255, 255, 255), -1)

        cv2.imwrite('tmp/left_arm_reach.png', left_arm_reach)
        cv2.imwrite('tmp/right_arm_reach.png', right_arm_reach)

        self.left_arm_mask = torch.tensor(left_arm_reach).bool().to(self.device)
        self.right_arm_mask = torch.tensor(right_arm_reach).bool().to(self.device)

    
    def load(self, path = None) -> int:
        load_path = path if path is not None else self.config.save_dir
        load_path = os.path.join(load_path, 'last_train.pth')
        checkpoint = torch.load(load_path)
        #print(checkpoint.keys())
        self.network.load_state_dict(checkpoint['net'])
        self.optimiser.load_state_dict(checkpoint['optimizer'])
        if 'replay_buffer' in checkpoint:
            self.replay_buffer = checkpoint['replay_buffer']
            self.update_step = checkpoint['step']
        self.update_step = -1
        if 'step' in checkpoint:
            self.update_step = checkpoint['step']
        print(f"Loaded checkpoint {self.update_step}")
        return self.update_step
    
    def load_checkpoint(self, checkpoint: int) -> bool:
        if checkpoint == -1:
            self.load()
        load_path = os.path.join(self.config.save_dir, f'checkpoint_{checkpoint}.pth')
        self.network.load_state_dict(torch.load(load_path)['net'])
        self.optimiser.load_state_dict(torch.load(load_path)['optimizer'])

        return True
    
    def reset(self, arena_ids):
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {}

    def get_state(self):
        return self.internal_states
    
    def init(self, information):
        pass

    def set_eval(self):
        return self.network.eval()
    
    def set_train(self):
        return self.network.train()
    
    def update(self, info, action):
        pass

    def _preporcess_info(self, info):
        retval = {}

        ##GENERATE OBSERVATION
        #print('rgb shape', info['observation']['rgb'].shape)
        H, W = info['observation']['rgb'].shape[:2]
        retval['prerot_rgb'] = info['observation']['rgb'].copy()
        # from matplotlib import pyplot as plt
        # cv2.imwrite('tmp/prerot_rgb.png', info['observation']['rgb'])
        
        # depth2plot = info['observation']['depth'].copy()
        # depth2plot = (depth2plot - depth2plot.min()) / (depth2plot.max() - depth2plot.min())
        # cv2.imwrite('tmp/depth.png', (depth2plot*255).astype(np.uint8))

        # mask2plot = info['observation']['mask'].copy()
        # cv2.imwrite('tmp/mask.png', mask2plot.astype(np.uint8)*255)


        for key in ['rgb', 'depth', 'mask']:
            assert key in info['observation'], f"Key {key} not found in observation"
            ## rotate the image clothwise 90 degrees
            # originally the flip base is on the right side
            # but my current environment is on the bottom side
            info['observation'][key] = np.rot90(info['observation'][key], 1)



        #print('depth shape', info['observation']['depth'].shape)

        
       
        #plt.imsave('tmp/rgb.png', info['observation']['rgb'])
        #print('mask shape', info['observation']['mask'].shape)

        retval['pretransform_depth'] = info['observation']['depth'].copy()
        retval['pretransform_rgb'] = info['observation']['rgb'].copy()
        
        masked_rgb = info['observation']['rgb'].copy()
        masked_rgb[~info['observation']['mask']] = 0
        rgbd = torch.cat([
            np_to_ts(masked_rgb, self.device).float()/255,
            np_to_ts(info['observation']['depth'].copy(), self.device).reshape(H, W, -1).float(),
        ], dim=2).permute(2, 0, 1)

        #print('RGBD shape', rgbd.shape)
        
        retval['transformed_obs'] = prepare_image(
            rgbd, 
            self._get_transformations(self.rotations), 
            self.config.obs_dim,
            orientation_net = None,
            parallelize=self.config.parallelize_prepare_image,
            nocs_mode=self.config.nocs_mode,
            inter_dim=256,
            constant_positional_enc=self.config.constant_positional_enc,)
    
        #print('Transformed obs shape', retval['transformed_obs'].shape)
        
        mask = np_to_ts(info['observation']['mask'].copy(), self.device)
        left_arm_mask = self.left_arm_mask.clone()
        right_arm_mask = self.right_arm_mask.clone()

        pretransform_mask = torch.stack([mask, left_arm_mask, right_arm_mask], dim=0)
        #print('Pretransform mask shape', pretransform_mask.shape)
        transformed_mask = prepare_image(
            pretransform_mask, 
            self._get_transformations(self.rotations), 
            self.config.obs_dim,
            parallelize=self.config.parallelize_prepare_image,
            nocs_mode=self.config.nocs_mode,
            inter_dim=128,
            constant_positional_enc=self.config.constant_positional_enc,)
        retval['pretransform_mask'] = pretransform_mask
        
        #print('Transformed mask shape', transformed_mask.shape)
        cloth_mask = transformed_mask[:, 0]
        #print('Cloth mask shape', cloth_mask.shape)
        left_arm_mask = transformed_mask[:, 1]
        right_arm_mask = transformed_mask[:, 2]

        workspace_mask = generate_workspace_mask(
            left_arm_mask,
            right_arm_mask,
            self.action_primitives,
            self.config.pix_place_dist,
            self.config.pix_grasp_dist,
        )

        cloth_mask = generate_primitive_cloth_mask(
            cloth_mask,
            self.action_primitives,
            self.config.pix_place_dist,
            self.config.pix_grasp_dist,
        )

        
        for primitive in self.action_primitives:
            GUARANTEE_OFFSET=6
            offset = self.config.pix_grasp_dist if primitive == 'fling' else self.config.pix_place_dist + GUARANTEE_OFFSET
            offset = int(offset)
            primitive_vmap_indices = self.primitive_vmap_indices[primitive] ## TODO: define this
            #print('Primitive vmap indices', primitive_vmap_indices)

            valid_transforms_mask = torch.zeros_like(cloth_mask[primitive]).bool()
            #print('valid_transforms_mask shape', valid_transforms_mask.shape)
            #print('vmp_idx dtype', type(primitive_vmap_indices[0]), type(primitive_vmap_indices[1]))
            id0s = torch.tensor(primitive_vmap_indices[0], device=self.device, dtype=torch.long)
            id1s = torch.tensor(primitive_vmap_indices[1], device=self.device, dtype=torch.long)

            first_dim_mask = torch.zeros(valid_transforms_mask.shape[0], dtype=torch.bool, device=self.device)
            for start, end in zip(id0s, id1s):
                first_dim_mask[start:end] = True

            #print('id0s', id0s, 'id1s', id1s)
            valid_transforms_mask[first_dim_mask, 
                        offset:-offset,
                        offset:-offset] = True
        
            table_mask = retval['transformed_obs'][:, 3] > 0
            offset_table_mask_up = torch.zeros_like(table_mask).bool()
            offset_table_mask_down = torch.zeros_like(table_mask).bool()
            offset_table_mask_up[:, :-offset, :] = table_mask[:, offset:]
            offset_table_mask_down[:, offset:, :] = table_mask[:, :-offset]
            table_mask = offset_table_mask_up & offset_table_mask_down & table_mask

            primitive_workspace_mask = torch.logical_and(workspace_mask[primitive], table_mask)
            primitive_workspace_mask = torch.logical_and(primitive_workspace_mask, valid_transforms_mask)

            retval[f"{primitive}_cloth_mask"] = cloth_mask[primitive]
            retval[f"{primitive}_workspace_mask"] = primitive_workspace_mask
            retval[f"{primitive}_mask"] = torch.logical_and(cloth_mask[primitive], primitive_workspace_mask)
        
        for key in ['rgb', 'depth', 'mask']:
            assert key in info['observation'], f"Key {key} not found in observation"
            ## rotate the image clothwise 90 degrees
            # originally the flip base is on the right side
            # but my current environment is on the bottom side
            info['observation'][key] = np.rot90(info['observation'][key], -1)

        #print('preproces keys', retval.keys())
        return retval



    def _get_transformations(self, rotations):
        #print('Rotations', rotations)
        #print('Adaptive scale factors', self.adaptive_scale_factors)
        return list(product(
            rotations, self.adaptive_scale_factors))

    
    def act(self, infos):
        ret_actions = []
        for info in infos:
            

            ## preprocess the info
            transformed_info = self._preporcess_info(info)
            

            ## Get the action tuple from the network
            action_tuple = self.network.act([transformed_info])[0]

            if action_tuple is None:
                return {
                    'no_op': True
                }

            ## Postprocess the action tuple
            action = self._postprocess_action_tuple(action_tuple, transformed_info)
            ret_actions.append(action)

        return ret_actions

    def _postprocess_action_tuple(self, action_tuple, info):

        primitive = action_tuple['chosen_primitive']
        if self.config.fling_only:
            primitive = 'fling'
        elif self.config.place_only:
            primitive = 'place'
   
        chosen_indices = action_tuple[primitive]['chosen_index']
        # print(f"Max indices: {max_indices}")

        chosen_deformable_value = action_tuple[primitive]['chosen_deformable_value']
        chosen_rigid_value = action_tuple[primitive]['chosen_rigid_value']
        chosen_value = action_tuple[primitive]['chosen_value']

        x, y, z = chosen_indices
        all_value_maps = action_tuple[primitive]['all_value_maps']
        value_map = all_value_maps[x]
        action_mask = torch.zeros(value_map.size())

        try:
            action_mask[y, z] = 1 
        except:
            print("Indices", chosen_indices)
            exit(1)

        num_scales = len(self.adaptive_scale_factors)
        rotation_idx = torch.div(x, num_scales, rounding_mode='floor')
        scale_idx = x - rotation_idx * num_scales
        scale = self.adaptive_scale_factors[scale_idx]

        rotation = self.rotations[rotation_idx]

        reach_points = np.array(self._get_action_params(
            action_primitive=primitive,
            max_indices=(x, y, z),
            # cloth_mask = transform_cloth_mask
            ))

        p1, p2 = reach_points[:2]

        if (p1 is None) or (p2 is None):
            print("\n [SimEnv] Invalid pickpoints \n", primitive, p1, p2)
            raise ValueError("Invalid pickpoints")

        action_kwargs = {
            'observation': info['transformed_obs'][x],
            'mask': info[f'{primitive}_mask'][x],
            'workspace_mask': info[f'{primitive}_workspace_mask'][x],
            'action_primitive': str(primitive),
            'primitive': primitive,
            'p1': p1,
            'p2': p2,
            'scale': scale,
            'nonadaptive_scale': self.scale_factors[scale_idx],
            'rotation': rotation,
            'predicted_deformable_value': float(chosen_deformable_value),
            'predicted_rigid_value': float(chosen_rigid_value),
            'predicted_weighted_value': float(chosen_value),
            'chosen_indices': np.array(chosen_indices),
            'action_mask': action_mask,
            'value_map': value_map,
            'all_value_maps': all_value_maps,
        }

        ## plot p1 and p2 on observation

        chosen_image = action_kwargs['observation'].detach().cpu().numpy()[:3]
        chosen_image = (chosen_image.transpose(1, 2, 0)*255).astype(np.uint8)
        # print('Chosen image shape', chosen_image.shape)
        # print('p1', p1, 'p2', p2)
        image_with_circles = chosen_image.copy()

        # Draw the circles
        cv2.circle(image_with_circles, (int(p1[1]), int(p1[0])), 5, (255, 0, 0), -1)
        cv2.circle(image_with_circles, (int(p2[1]), int(p2[0])), 5, (0, 255, 0), -1)
        cv2.imwrite('tmp/chosen_image.png', image_with_circles)

        # reverse the chosen_image to original image, the rotation is given `rotation`, the scale is given `scale`

        inverse_chosen_image = action_kwargs['observation']

        # rotate by `rotation` degrees
        inverse_chosen_image = transform(inverse_chosen_image, -rotation, 1.0/scale, inverse_chosen_image.shape[1])[:3].detach().cpu().numpy()
        inverse_chosen_image = (inverse_chosen_image.transpose(1, 2, 0)*255).astype(np.uint8).copy()
        # also do the inverse rotate and scale for p1 and p2
        #print('rotation', rotation, 'scale', scale)
        T2d = get_transform_matrix(inverse_chosen_image.shape[0], inverse_chosen_image.shape[0], -rotation, scale)

        transform_pixels = np.concatenate((np.array([p1, p2]), np.ones((2, 1))), axis=1)
        #print('transform_pixels', transform_pixels)
        pixels = np.matmul(transform_pixels, T2d)[:, :2].astype(int)
        #print('inverse pixels', pixels)
        p1, p2 = pixels[0], pixels[1]

        # Draw the circles
        # print('inverse chosen image shape', inverse_chosen_image.shape)
        # cv2.circle(inverse_chosen_image, (int(p1[1]), int(p1[0])), 5, (255, 0, 0), -1)
        # cv2.circle(inverse_chosen_image, (int(p2[1]), int(p2[0])), 5, (0, 255, 0), -1)
        # cv2.imwrite('tmp/inverse_chosen_image.png', inverse_chosen_image)

        if action_tuple[primitive].get('raw_value_map') is not None:
            action_kwargs['raw_value_maps'] = action_tuple[primitive]['raw_value_maps']


        assert ((action_kwargs['p1'] is not None) and (action_kwargs['p2'] is not None))

        action_kwargs.update({
            'transformed_depth':
            action_kwargs['observation'][3, :, :],
            'transformed_rgb':
            action_kwargs['observation'][:3, :, :],
        })

        #print('pix1', p1, 'pix2', p2)

        # action_params = self.check_action(
        #     info,
        #     pixels=np.array([p1, p2]),
        #     **action_kwargs)


        # try:
        #     reachable, left_or_right = self._check_action_reachability(
        #         action=primitive,
        #         p1=action_params['p1'],
        #         p2=action_params['p2'])
        # except ValueError as e:
        #     raise ValueError("Reach pos none")
        

        # if primitive == 'place':
        #     action_kwargs['left_or_right'] = left_or_right


        # for k in ['valid_action',
        #             'pretransform_pixels']:
        #     del action_params[k]

        # print('primitive', action_kwargs['action_primitive'])
        # print('p1', p1, 'p2', p2)
        ### rotation the pixel points 90 degrees anti-clockwise
        T2d = get_transform_matrix(info['prerot_rgb'].shape[1], inverse_chosen_image.shape[1], 90, 1)

        transform_pixels = np.concatenate((np.array([p1, p2]), np.ones((2, 1))), axis=1)
        #print('transform_pixels', transform_pixels)
        pixels = np.matmul(transform_pixels, T2d)[:, :2].astype(int)
        #print('output pixels', pixels)
        p1, p2 = pixels[0], pixels[1]

        #print('p1', p1, 'p2', p2)

        # draw_image = info['prerot_rgb'].copy()
        # cv2.circle(draw_image, (int(p1[1]), int(p1[0])), 5, (255, 0, 0), -1)
        # cv2.circle(draw_image, (int(p2[1]), int(p2[0])), 5, (0, 255, 0), -1)
        # cv2.imwrite('tmp/output_action.png', draw_image)

        # swap x and y
        p1, p2 = p1[::-1], p2[::-1]

        ### pack and send the action
        H, W = info['prerot_rgb'].shape[:2]
        chosen_primitive = 'norm-pixel-pick-and-place' if action_kwargs['action_primitive'] == 'place' else 'norm-pixel-pick-and-fling'
        chosen_primitive_params = {
            'pick_0': p1/np.array([H, W]) * 2 - 1,
            'pick_1': p2/np.array([H, W]) * 2 - 1,
        }

        if action_kwargs['action_primitive'] == 'place':
            chosen_primitive_params['place_0'] = chosen_primitive_params['pick_1']
            del chosen_primitive_params['pick_1']

        action =  {
            chosen_primitive: chosen_primitive_params
        }

        #print('Action', action)
        return action

    
    ### TODO: this should be in the environment
    def _check_action_reachability(
            self, action: str, p1: np.array, p2: np.array):
        if (p1 is None) or (p2 is None):
           raise ValueError(f'[Invalid action] {action} reach points are None')
        if action in ['fling','drag','stretchdrag']:
            # right and left must reach each point respectively
            return self._check_arm_reachability(self.left_arm_base, p1) \
                and self._check_arm_reachability(self.right_arm_base, p2), None
        elif action == 'drag' or action == 'place':
            # either right can reach both or left can reach both
            if self._check_arm_reachability(self.left_arm_base, p1) and\
                    self._check_arm_reachability(self.left_arm_base, p2):
                return True, 'left'
            elif self._check_arm_reachability(self.right_arm_base, p1) and \
                    self._check_arm_reachability(self.right_arm_base, p2):
                return True, 'right'
            else:
                return False, None
        raise NotImplementedError()
    
    def _check_arm_reachability(self, arm_base, reach_pos):
        try:
            return np.linalg.norm(arm_base - reach_pos) < self.reach_distance_limit
        except Exception as e:
            print(e)
            print("[Check arm] Reachability error")
            print("arm_base:", arm_base)
            print("reach_pos:", reach_pos)
            return False, None
    
    def _get_action_params(self, action_primitive, max_indices, cloth_mask=None):
        x, y, z = max_indices
        if action_primitive == 'fling' or\
                action_primitive == 'stretchdrag':
            center = np.array([x, y, z])
            p1 = center[1:].copy()
            p2 = center[1:].copy()

            p1[0] = p1[0] + self.config.pix_grasp_dist
            p2[0] = p2[0] - self.config.pix_grasp_dist

        elif action_primitive == 'drag':
            p1 = np.array([y, z])
            p2 = p1.copy()
            p2[0] += self.pix_drag_dist
        elif action_primitive == 'place':
            p1 = np.array([y, z])
            p2 = p1.copy()
            p2[0] += self.config.pix_place_dist
        else:
            raise Exception(
                f'Action Primitive not supported: {action_primitive}')
        if (p1 is None) or (p2 is None):
            raise Exception(
                f'None reach points: {action_primitive}')
        return p1, p2


    def save(self, path=None) -> bool:
        save_path = path if path is not None else self.config.save_path
        checkpoint_path = os.path.join(save_path, f'checkpoint_{self.update_step}.pth')
        
        # save network to the checkpoint path,
        # and save replay buffer, optimiser and network with train_{} prefix
        torch.save({
            'network': self.network.state_dict(),
            'optimiser': self.optimiser.state_dict(),
        }, checkpoint_path)

        torch.save({
            'network': self.network.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'replay_buffer': self.replay_buffer,
            'step': self.update_step
            }, os.path.join(save_path, f'last_train.pth'))
            
    
        return True

    def train(self, update_steps, arena=None) -> bool:
        
        if self.config.train_mode == 'offline':
            return self._train_offline(update_steps)
        elif self.config.train_mode == 'online':
            return self._train_online(update_steps, arena)
    
    def _train_online(self, update_steps, arena):
        target_update_step = self.update_step + update_steps
        
        if self.replay_buffer.size() < self.config.pretrain_collect_steps:
            ## collect data using network output
            ## in the original code, it runs paralelle environments to collect data
            ## but here we provide sequential version.
            collect_steps = self.config.pretrain_collect_steps - self.replay_buffer.size()
            self._collect_data(collect_steps, arena)

        for u in range(self.update_step, target_update_step):
            
            self._update_network()

            if u%self.config.updates_per_collect == 0:
                # updates_per_collect is 8 and steps_per_collect is 128 
                collect_steps = (u // self.config.updates_per_collect) \
                    * self.config.steps_per_collect - self.replay_buffer.size()
                self._collect_data(collect_steps, arena)
    
    def _train_offline(self, update_steps):
        target_update_step = self.update_step + update_steps
        for u in range(self.update_step, target_update_step):
            self._update_network()

    def _update_network(self):
        self.network.train()
        value_net = self.network.value_net
        
        sample = self.replay_buffer.sample(self.config.batch_size)

        #COMPUTE LOSSES
        # 2-picker pick-and-place and pick-and-fling in the original paper.
        stats = dict()
        unfactorized_losses = dict()
        distance_types = ['rigid', 'deformable'] # rigid and deformable distances in the original paper

        losses = {
            dt: {primitive: {'manipulation': 0} for primitive in self.config.action_primitives}
            for dt in distance_types
        }
        l2_error = {
            dt: {primitive: {'manipulation': 0} for primitive in self.config.action_primitives}
            for dt in distance_types
        }
        unfactorized_losses = {primitive: 0 for primitive in self.config.action_primitives}

        for primitive_id in range(len(self.config.action_primitives)): 

            action_primitive = self.config.action_primitives[primitive_id]

            in_dict = sample[primitive_id]
            obs = in_dict['obs']
            action_mask = in_dict['action']
            weighted_reward = in_dict['weighted_reward']
            deformable_reward = in_dict['deformable_reward']
            rigid_reward = in_dict['rigid_reward']
            l2_reward = in_dict['l2_reward']
            cov_reward = in_dict['coverage_reward']
            is_terminal = in_dict['is_terminal'].to(self.device)

            action_mask = action_mask.unsqueeze(1)
            
            rewards = {'rigid': rigid_reward, 'deformable': deformable_reward}

            #preprocess here so that we can log the obs
            obs = value_net.preprocess_obs(obs.to(self.device, non_blocking=True))
            out = value_net.forward_for_optimize(obs, action_primitive, preprocess=False)

            action_mask = torch.cat([action_mask], dim=1)

            # deformable_weight is 0.65 in the original code
            unfactorized_value_pred_dense = (1-self.config.deformable_weight) \
                  * out['rigid'][action_primitive] + \
                 self.config.deformable_weight * out['deformable'][action_primitive]
            unfactorized_value_pred = torch.masked_select(
                unfactorized_value_pred_dense, action_mask.\
                    to(self.device, non_blocking=True))

            if self.config.unfactorized_rewards: # False in the original code
                if self.config.coverage_reward: # False in the original code
                    print("[Network] Using coverage reward")
                    unfactorized_reward = cov_reward.to(self.device)
                else:
                    print("[Network] Using unfactorized reward")
                    unfactorized_reward = l2_reward.to(self.device)
            else:
                print("[Network] Using factorized reward")
                unfactorized_reward = weighted_reward.to(self.device)

          
            unfactorized_losses[action_primitive] = \
                torch.nn.functional.smooth_l1_loss(
                    unfactorized_value_pred, unfactorized_reward.to(self.device, non_blocking=True))

            for distance in distance_types:

                value_pred_dense = out[distance][action_primitive]
                value_pred = torch.masked_select(
                    value_pred_dense,
                    action_mask.to(self.device, non_blocking=True))
    
                reward = rewards[distance].to(self.device)

                manipulation_loss = torch.nn.functional.smooth_l1_loss(value_pred, reward)

                losses[distance][action_primitive]['manipulation'] = \
                    manipulation_loss / LOSS_NORMALIZATIONS[distance][action_primitive]['manipulation']
             
                l2_error[distance][action_primitive]['manipulation'] = manipulation_loss

                log_idx = 0
                # visualizations[distance][action_primitive]['manipulation'] = value_pred_dense[log_idx].detach().cpu().numpy()
                # visualizations[distance][action_primitive]['obs'] = obs[log_idx].detach().cpu().numpy()

        #OPTIMIZE
        loss = 0

        for distance in distance_types:
            for primitive in self.config.action_primitives:
                stats[f'loss/{primitive}/unfactorized']= unfactorized_losses[primitive] / len(self.config.action_primitives)
                stats[f'loss/{primitive}/{distance}/factorized'] = losses[distance][primitive]['manipulation'] / len(self.config.action_primitives)
                stats[f'l2_error/{primitive}/{distance}/factorized'] = l2_error[distance][primitive]['manipulation'] / len(self.config.action_primitives)

        self.optimiser.zero_grad()
        if self.config.unfactorized_networks: # False in the original code
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/unfactorized' in k)
        else:
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/factorized' in k)
        loss.backward()
        self.optimiser.step()
