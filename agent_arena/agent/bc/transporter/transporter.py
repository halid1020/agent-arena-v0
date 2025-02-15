# Code from https://github.com/thomaschabal/transporter-nets-torch/blob/main/ravens_torch/agents/transporter.py

# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transporter Agent."""

import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


from agent_arena.arena.raven.tasks import cameras

from .models.attention import Attention
from .models.transport import Transport
from .models.transport_ablation import TransportPerPixelLoss
from .models.transport_goal import TransportGoal
from .utils import utils


class TransporterAgent:
    """Agent that uses Transporter Networks."""

    def __init__(self, input_obs='rgb',
                 n_rotations=36, in_shape=(320, 160, 6),
                 crop_size=64, pixel2world=True, transformer=(lambda x: x), **kwargs):
        self.current_update_steps = 0
        self.crop_size = crop_size
        self.n_rotations = n_rotations
        self.depth_only = kwargs['depth_only'] if 'depth_only' in kwargs else False
        self.save_dir = '.'
        self.models_dir = os.path.join(self.save_dir, 'checkpoints')
        self.augmentation = kwargs['augmentation'] if 'augmentation' in kwargs else True
        
        self.in_shape = tuple(in_shape)
        self.pixel2world = pixel2world
        self.transform = transformer
        self.input_obs = input_obs
        
        if pixel2world:
            self.pix_size = 0.003125
            self.cam_config = cameras.RealSenseD415.CONFIG
            self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        #self.models_dir = os.path.join(root_dir, 'checkpoints')

        self.use_goal_image = \
            kwargs['goal_condition'] if 'goal_condition' in kwargs else None
        
        if self.use_goal_image:
            self.goal_mode = kwargs['goal_mode']

        #self.rotate_samples = kwargs['rotate_samples'] if 'rotate_samples' in kwargs else False
        
        self.two_picker = kwargs['two_picker'] if 'two_picker' in kwargs else False
        self.pick_action_filter = kwargs['pick_action_filter'] if 'pick_action_filter' in kwargs else 'mask'
        self.place_action_filter = kwargs['place_action_filter'] if 'place_action_filter' in kwargs else 'identity'
        self.pick_and_place_policy = kwargs['pick_and_place_policy'] if 'pick_and_place_policy' in kwargs else 'default'
        if self.pick_and_place_policy in ['probability_product', 'sample']:
            self.num_pick_prob_prod = kwargs['num_pick_prob_prod'] if 'num_pick_prob_prod' in kwargs else 10
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1

    def get_image(self, obs, goal=None, train=False, sim2real=False):
        """Stack color and height images image. Rotate Samples if needed."""

        #print('get image sim2real', sim2real)
        # Get color and height maps from RGB-D images.
        if self.pixel2world:
            cmap, hmap = utils.get_fused_heightmap(
                obs, self.cam_config, self.bounds, self.pix_size)
            
            obs = {
                'rgb': cmap,
                'depth': hmap,
            }
            
            obs = self.transform(obs, train=train, sim2real=sim2real)   
            cmap, hmap = obs['rgb'], obs['depth']
            act = obs['action'] if 'action' in obs else None

            
            if self.input_obs == 'rgbd3':
                img = np.concatenate((cmap, hmap.copy(), hmap.copy(), hmap.copy()), axis=2)\
                        .astype(np.float32)
            else:
                raise NotImplementedError('input_obs: {} not implemented'.format(self.input_obs))
            

            if goal is not None and self.use_goal_image:
                colormap_g, heightmap_g = \
                    utils.get_fused_heightmap(
                        goal, self.cam_config, self.bounds, self.pix_size)

                goal_image = np.concatenate((colormap_g,
                              heightmap_g[Ellipsis, None],
                              heightmap_g[Ellipsis, None],
                              heightmap_g[Ellipsis, None]), axis=2)


                img = np.concatenate((img, goal_image), axis=2)
        
        else:
            #   from matplotlib import pyplot as plt
            # plt.imshow(obs['color'])
            # plt.show()
            # print('obs keys', obs.keys())
            # print('goals keys', goal.keys())
            if goal is not None:
                for k, v in goal.items():
                    obs['goal_{}'.format(k)] = v
            # plt.imshow(obs['goal_color'])
            # plt.savefig('goal_color_pre.png')
            #print('obs keys', obs.keys())
            obs = self.transform(obs, train=train, sim2real=sim2real)   
            if goal is not None:
                for k, v in obs.items():
                    if 'goal' in k:
                        ## remove the goal_ prefix
                        goal[k[5:]] = v
            # plt.imshow(obs['goal_rgb'])
            # plt.savefig('goal_color_post.png')

            act = obs['action'] if 'action' in obs else None
            cmap = obs['rgb']
            if 'depth' in obs:
                hmap = obs['depth']
                hmap = hmap.reshape(hmap.shape[0], hmap.shape[1], 1)
            # statistics of cmap
            # print('cmap max', np.max(cmap))
            # print('cmap min', np.min(cmap))
        
            # from matplotlib import pyplot as plt
            # plt.imshow(cmap)
            # plt.show()
            # plt.imshow(hmap)
            # plt.show()

            # print('cmap shape', cmap.shape)
            # print('hmap shape', hmap.shape)
            if self.input_obs == 'depth' :
                img =  hmap.copy().astype(np.float32)
            elif self.input_obs == 'depth3':
                img = np.concatenate((hmap.copy(), hmap.copy(), hmap.copy()), axis=2).astype(np.float32)
                    
            elif self.input_obs == 'rgbd3':
                img = np.concatenate((cmap, hmap.copy(), hmap.copy(), hmap.copy()), axis=2)\
                        .astype(np.float32)
            elif self.input_obs == 'rgbd':
                img = np.concatenate((cmap, hmap.copy()), axis=2).astype(np.float32)
            elif self.input_obs == 'rgb':
                img = cmap.astype(np.float32)
            elif self.input_obs == 'gray+depth':
                gmap = obs['gray']
                # plt.imshow(gmap)
                # plt.show()
                # plt.imshow(hmap)
                # plt.show()
                img = np.concatenate((gmap, hmap.copy()), axis=2).astype(np.float32)
            elif self.input_obs == 'gray':
                gmap = obs['gray']
                img = gmap.astype(np.float32)
                
                # plt.imshow(gmap)
                # plt.show()


            if goal is not None and self.use_goal_image:
                colormap_g, heightmap_g = goal['rgb'], goal['depth']
                if self.input_obs == 'rgbd3':
                    goal_image = np.concatenate((colormap_g, heightmap_g.copy(), heightmap_g.copy(), heightmap_g.copy()), axis=2)\
                            .astype(np.float32)
                else:
                    raise NotImplementedError('goal image not implemented for input_obs: {}'.format(self.input_obs))
                img = np.concatenate((img, goal_image), axis=2)
                
                
                # plt.imshow(img[:, :, :3] + 0.5)
                # plt.savefig('cur_rgb.png')
                # plt.imshow(img[:, :, 3] + 0.5)
                # plt.savefig('cur_depth.png')
                # plt.imshow(img[:, :, 6:9] + 0.5)
                # plt.savefig('goal_rgb.png')
                # plt.imshow(img[:, :, 9] + 0.5)
                # plt.savefig('goal_depth.png')
                # plt.imshow(img[:, :, 6:9].astype(np.uint8))
                # plt.show()

        
        return img, act
        
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        self.models_dir = os.path.join(save_dir, 'checkpoints')

    # TODO: check the different here in the two cases
    def get_sample(self, dataset, augment=True):
        """Get a dataset sample.

        Args:
          dataset: a agents.transporter.Dataset (train or validation)
          augment: if True, perform data augmentation.

        Returns:
          tuple of data for training:
            (input_image, p0pick, p0pick_theta, p0place, p0place_theta)
          tuple additionally includes (z, roll, pitch) if self.six_dof
          if self.use_goal_image, then the goal image is stacked with the
          current image in `input_image`. If splitting up current and goal
          images is desired, it should be done outside this method.
        """

        (obs, act_, _, _), (goal_obs, _, _, _) = dataset.sample()
        obs['action'] = act_
        #print('before transform', act_)
        img, act = self.get_image(obs, goal_obs, train=True)
        if act is None:
            act = act_
        #act = obs['action']
        #print('after transform', act)
        #print('img get image shape', img.shape)

        # Get training labels from data sample.
        if self.pixel2world:
            p0pick_xyz, p0pick_xyzw = act['pose0']
            p0place_xyz, p0place_xyzw = act['pose1']
            p0pick = utils.xyz_to_pix(p0pick_xyz, self.bounds, self.pix_size)
            p0pick_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0pick_xyzw)[2])
            p0place = utils.xyz_to_pix(p0place_xyz, self.bounds, self.pix_size)
            p0place_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0place_xyzw)[2])
            p0place_theta = p0place_theta - p0pick_theta
            p0pick_theta = 0
        else:
            # print('act shape', act.shape)
            # print('in shape', self.in_shape)
            act = act.reshape(-1, 4)
            p0pick = ((act[0, :2] + 1)/2 * self.in_shape[0]).astype(np.int32).clip(0, self.in_shape[0]-1)
            p0pick_theta = 0 ## TODO: refactor this
            p0place = ((act[0, 2:4] + 1)/2 * self.in_shape[1]).astype(np.int32).clip(0, self.in_shape[1]-1)
            p0place_theta = 0 ## TODO: refactor this

        if self.two_picker:
            p1pick = ((act[1, :2] + 1)/2 * self.in_shape[0]).astype(np.int32).clip(0, self.in_shape[0]-1)
            p1pick_theta = 0 ## TODO: refactor this
            p1place = ((act[1, 2:4] + 1)/2 * self.in_shape[1]).astype(np.int32).clip(0, self.in_shape[1]-1)
            p1place_theta = 0 ## TODO: refactor this


            
        # Data augmentation.
        pick_and_place = {'p0pick': p0pick, 'p0theta': p0pick_theta, 
                          'p0place': p0place, 'p0place_theta': p0place_theta}
        if self.two_picker:
            pick_and_place.update({'p1pick': p1pick, 'p1theta': p1pick_theta, 
                          'p1place': p1place, 'p1place_theta': p1place_theta})

        #print('pick_and_place', pick_and_place)
        
        if augment:
            print('hereree')
            pick_and_place_pos = [p0pick, p0place]
            if self.two_picker:
                pick_and_place_pos.extend([p1pick, p1place])

            #img, _, (p0pick, p0place), _ = utils.perturb(img, )
            # plt.imshow(img)
            # plt.show()

            
            
            img, _, pick_and_place_pos, _ = utils.perturb(img, pick_and_place_pos)
            #print('img shape', img.shape)

            # plt.imshow(img[:, :, :3].astype(np.uint8))
            # plt.show()
            # plt.imshow(img[:, :, 3])
            # plt.show()
            # plt.imshow(img[:, :, 6:9].astype(np.uint8))
            # plt.show()
            # plt.imshow(img[:, :, 9])
            # plt.show()
            
            

            pick_and_place['p0pick'] = pick_and_place_pos[0]
            pick_and_place['p0place'] = pick_and_place_pos[1]
            if self.two_picker:
                pick_and_place['p1pick'] = pick_and_place_pos[2]
                pick_and_place['p1place'] = pick_and_place_pos[3]



        return img, pick_and_place

    def train(self, dataset, writer=None):
        """Train on a dataset sample for 1 iteration.

        Args:
          dataset: a agents.transporter.Dataset.
          writer: a TensorboardX SummaryWriter.
        """
        self.attention.train_mode()
        self.transport.train_mode()

        #img, p0pick, p0pick_theta, p0place, p0place_theta = self.get_sample(dataset)
        images = []
        p0picks = []
        p0pick_thetas = []
        p0places = []
        p0place_thetas = []
        for b in range(self.batch_size):
            img, pick_and_place = self.get_sample(dataset, augment=self.augmentation)
            p0pick = pick_and_place['p0pick']
            p0pick_theta = pick_and_place['p0theta']
            p0place = pick_and_place['p0place']
            p0place_theta = pick_and_place['p0place_theta']

            images.append(img)
            p0picks.append(p0pick)
            p0pick_thetas.append(p0pick_theta)
            p0places.append(p0place)
            p0place_thetas.append(p0place_theta)




        # Get training losses.
        step = self.current_update_steps + 1
        loss0 = self.attention.train(images, p0picks, p0pick_thetas)
        
        
        if isinstance(self.transport, Attention):
            loss1 = self.transport.train(images, p0places, p0place_thetas)

        elif self.use_goal_image and self.goal_mode == 'goal-split':

            loss1 = self.transport.train(
                img[:, :, :6], img[:, :, 6:], 
                p0pick, p0place, p0place_theta)
        else:
            loss1 = self.transport.train(images, p0picks, p0places, p0place_thetas)
        
       

        writer.add_scalars([
            ('train_loss/attention', loss0, step),
            ('train_loss/transport', loss1, step),
        ])

        print(
            f'Train Iter: {step} \t Attention Loss: {loss0:.4f} \t Transport Loss: {loss1:.4f}')
        self.current_update_steps = step

    def validate(self, dataset, writer=None):  # pylint: disable=unused-argument
        """Test on a validation dataset for 10 iterations."""

        n_iter = 10
        loss0, loss1 = 0, 0
        for i in range(n_iter):
            #img, p0pick, p0pick_theta, p0place, p0place_theta = self.get_sample(dataset, False)
            img, pick_and_place = self.get_sample(dataset, augment=False)
            if writer is not None:
                if self.input_obs in ['rgb', 'rgbd3', 'rgbd']:
                    # normalise rgb to 0-1
                    #print('img shape', img.shape)
                    rgb = {'rgb': img[:, :, :3]}
                    rgb = self.transform.postprocess(rgb)['rgb'].transpose(2, 0, 1)
                    #print('rgb shape', rgb.shape)
                    #rgb = (rgb - np.min(rgb))/(np.max(rgb) - np.min(rgb))
                    img_to_vis = (rgb*255).astype(np.uint8)
                    writer.add_image('val image/{}/rgb'.format(i), img_to_vis, self.current_update_steps)
                if self.input_obs in ['depth', 'rgbd3']:
                    if len(img.shape) == 2:
                        img_to_vis = img.copy()
                    else:
                        img_to_vis = img[:, :, -1].copy()
                    #print('max depth', np.max(img_to_vis))
                    #print('min depth', np.min(img_to_vis))
                    img_to_vis = img_to_vis.clip(0, 1)
                    img_to_vis = (img_to_vis*255).astype(np.uint8)
                    img_to_vis = cv2.applyColorMap(img_to_vis, cv2.COLORMAP_AUTUMN).transpose(2, 0, 1)
                    ## if there nan entry exit
                    if np.isnan(img_to_vis).any():
                        print('nan entry in depth image')
                        exit()
                    #print('img_to_vis shape', img_to_vis.shape)
                    writer.add_image('val image/{}/depth'.format(i), img_to_vis, self.current_update_steps)
                
                # if self.input_obs in ['depth', 'rgbd3']:
                #     img_to_vis = img[:, :, -2:-1].copy().astype(np.float32).transpose(2, 0, 1)
                #     writer.add_image('val image/{}/depth'.format(i), img_to_vis, self.current_update_steps)

            p0pick = pick_and_place['p0pick']
            p0pick_theta = pick_and_place['p0theta']
            p0place = pick_and_place['p0place']
            p0place_theta = pick_and_place['p0place_theta']
            #print('action', pick_and_place)
            
            ## TODO: add support for two picker

            # Get validation losses. Do not backpropagate.
            loss0 += self.attention.test([img], [p0pick], [p0pick_theta])
            
            if isinstance(self.transport, Attention):
                loss1 += self.transport.test([img], [p0place], [p0place_theta])
            elif self.use_goal_image and self.goal_mode == 'goal-split':
                loss1 += self.transport.test(
                    img[:, :, :6], img[:, :, 6:], 
                    p0pick, p0place, p0place_theta)
            else:
                loss1 += self.transport.test([img], [p0pick], [p0place], [p0place_theta])

            
            if self.two_picker:

                p1pick = pick_and_place['p1pick']
                p1pick_theta = pick_and_place['p1theta']
                p1place = pick_and_place['p1place']
                p1place_theta = pick_and_place['p1place_theta']

                ## Get label using p0pick_pix and p0pick_theta
                theta_i =  p0pick_theta / (2 * np.pi / 1)
                theta_i = np.int32(np.round(theta_i)) % 1
                label_size = self.in_shape[:2] + (1,)
                labelpick = np.zeros(label_size)
                labelpick[p0pick[0], p0pick[1], theta_i] = 1

                ## Get label using p0place_pix and p0place_theta
                theta_i =  p0place_theta / (2 * np.pi / self.n_rotations)
                theta_i = np.int32(np.round(theta_i)) % self.n_rotations
                label_size = self.in_shape[:2] + (self.n_rotations,)
                labelplace = np.zeros(label_size)
                labelplace[p0place[0], p0place[1], theta_i] = 1

                ## concanate the labels to the img as new channels
                img4second = np.concatenate((img, labelpick, labelplace), axis=2)
                loss0_1 = self.attention_1.test(img4second, p1pick, p1pick_theta)

                if isinstance(self.transport, Attention):
                    loss1 += self.transport.test(img, p1place, p1place_theta)
   
                else:
                    loss1 += self.transport.test(img, p1pick, p1place, p1place_theta)

            

        loss0 /= n_iter
        loss1 /= n_iter

        writer.add_scalars([
            ('test_loss/attention', loss0, self.current_update_steps),
            ('test_loss/transport', loss1, self.current_update_steps),
        ])

        if self.two_picker:
            writer.add_scalars([
                ('test_loss/attention_1', loss0_1, self.current_update_steps)
            ])

        print(
            f'Validation: \t Attention Loss: {loss0:.4f} \t Transport Loss: {loss1:.4f}')
    def set_eval(self):
        self.attention.eval_mode()
        self.transport.eval_mode()
    def set_train(self):
        self.attention.train_mode()
        self.transport.train_mode()
    
    def init(self):
        self.last_pick = None
        self.state = {}
    
    def act(self, obs, info=None, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        self.attention.eval_mode()
        self.transport.eval_mode()
        
        #cv2.imwrite('TN-color.png', obs['current']['color'])
        
        # if 'depth' in obs['current']:
        #     depth_img = obs['current']['depth']
        #     depth_img = (depth_img - np.min(depth_img))/(np.max(depth_img) - np.min(depth_img))
        #     depth_img = cv2.applyColorMap((depth_img*255).astype(np.uint8), cv2.COLORMAP_SUMMER)

        #print('!!!!obs sim2real', obs['sim2real'])
        img, _ = self.get_image(obs['current'], 
                                obs['goal'], train=False, 
                                sim2real=(obs['sim2real'] if 'sim2real' in obs.keys() else False))
        if 'rgb' in self.input_obs:
            self.state['input_obs'] = self.transform.postprocess({'rgb': img[:, :, :3]})['rgb']
        self.state['input_type'] = self.input_obs
        #print('act img shape', img.shape)

        # Attention model forward pass.
        pick0_conf = self.attention.forward(np.stack([img]))
        pick_heat = (pick0_conf - np.min(pick0_conf))/(np.max(pick0_conf) - np.min(pick0_conf))
        pick_heat = cv2.applyColorMap((pick_heat * 255).astype(np.uint8), cv2.COLORMAP_AUTUMN)
        self.state['pick-heat'] = pick_heat
        #cv2.imwrite('TN-pick.png', pick_heat)
        
        if self.pick_action_filter == 'mask':
            ## resize the mask to the same size as the pick0_conf
            mask = obs['current']['mask'].astype(np.float32)
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            #mask = cv2.resize(mask, (pick0_conf.shape[0], pick0_conf.shape[1]), interpolation=cv2.INTER_LINEAR)
            ## get make the resized mask to be 0 or 1
            mask = np.where(mask > 0.5, 1, 0).reshape(mask.shape[0], mask.shape[1], 1)
            ## make the boundary of the mask all 0
            mask[0, :, :] = 0
            mask[-1, :, :] = 0
            mask[:, 0, :] = 0
            mask[:, -1, :] = 0
            pick0_conf_ = pick0_conf.copy()
            pick0_conf = pick0_conf * mask
            pick_mask = (pick0_conf - np.min(pick0_conf))/(np.max(pick0_conf) - np.min(pick0_conf))
            pick_mask = cv2.applyColorMap((pick_mask*255).astype(np.uint8), cv2.COLORMAP_AUTUMN)
            self.state['masked-pick-heat'] = pick_mask
            #cv2.imwrite('TN-pick-mask.png', pick_mask)


            ### Mask over rgb image
            to_mask = (mask  + 0.5).clip(0, 1).repeat(3, axis=2)
            mask_over_rgb = (obs['current']['color'].astype(np.float32)/255) * to_mask
            
           
        elif self.pick_action_filter == 'identity':
            pass
        else:
            raise ValueError('Invalid pick_action_filter: {}'.format(self.pick_action_filter))
    
        pick0pick_pixs = []
        pick0pick_thetas = []
        
        if self.pick_and_place_policy == "probability_product":
            pick0_conf_tmp = pick0_conf.copy()
            # colourise the heatmap with JET
            
            
            #print('num picks', self.num_pick_prob_prod)
            for i in range(self.num_pick_prob_prod):
                argmax = np.argmax(pick0_conf_tmp)
                argmax = np.unravel_index(argmax, shape=pick0_conf_tmp.shape)
                pick0_conf_tmp[argmax] = 0
                pick0pick_pix = argmax[:2]
                pick0pick_theta = argmax[2] 
                pick0pick_pixs.append(pick0pick_pix)
                pick0pick_thetas.append(pick0pick_theta)

        elif self.pick_and_place_policy == "default":
            argmax = np.argmax(pick0_conf)
            argmax = np.unravel_index(argmax, shape=pick0_conf.shape)
            p0pick_pix = argmax[:2]
            p0pick_theta = argmax[2]
            pick0pick_pixs.append(p0pick_pix)
            pick0pick_thetas.append(p0pick_theta)
        elif self.pick_and_place_policy == "sample":
            pick0_conf_tmp = pick0_conf.copy()
            for i in range(self.num_pick_prob_prod):
                argmax = np.argmax(pick0_conf_tmp)
                argmax = np.unravel_index(argmax, shape=pick0_conf_tmp.shape)
                pick0_conf_tmp[argmax] = 0
                pick0pick_pix = argmax[:2]
                pick0pick_theta = argmax[2] 
                pick0pick_pixs.append(pick0pick_pix)
                pick0pick_thetas.append(pick0pick_theta)

        else:
            raise ValueError('Invalid pick_and_place_policy: {}'.format(self.pick_and_place_policy))

        #print('p0pick_pix', p0pick_pix)

        
        # Transport model forward pass.
        p0place_pixs = []
        p0place_thetas = []
        place0_confs = []
        for i in range(len(pick0pick_pixs)):
            if self.use_goal_image and self.goal_mode == 'goal-split':
                place0_conf = self.transport.forward(
                    np.stack([img[:, :, :6]]), np.stack([img[:, :, 6:]]), np.stack([pick0pick_pixs[i]])
                )
            else:
                place0_conf = self.transport.forward(
                    np.stack([img]), 
                    np.stack([pick0pick_pixs[i]])).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                if i == 0 and place0_conf.shape[-1] == 1:
                    # normalise the place0_conf
                    place_heat = (place0_conf - np.min(place0_conf))/(np.max(place0_conf) - np.min(place0_conf))
                    print('place-heat', place_heat.shape)
                    place_heat = cv2.applyColorMap((place_heat * 255).astype(np.uint8), cv2.COLORMAP_SPRING)
                    self.state['place-heat-best-pick'] = place_heat
                    #cv2.imwrite('TN-place.png', place_heat)

                #print('place0_conf shape', place0_conf.shape)

            if self.place_action_filter == 'square-center':
                mask = np.zeros(place0_conf.shape)
                boundary = int(place0_conf.shape[0] * 0.05)
                mask[boundary:-boundary, boundary:-boundary, :] = 1
                place0_conf = place0_conf * mask
            if self.place_action_filter == 'around-pick':
                mask = np.ones(place0_conf.shape)
                boundary = 3
                mask[pick0pick_pixs[i][0]-boundary:pick0pick_pixs[i][0]+boundary, \
                     pick0pick_pixs[i][1]-boundary:pick0pick_pixs[i][1]+boundary, :] = 0
                place0_conf = place0_conf * mask
                #print('here')
            elif self.place_action_filter == 'identity':
                pass
            else:
                raise ValueError('Invalid place_action_filter: {}'.format(self.place_action_filter))
            place0_confs.append(place0_conf.copy())
            argmax = np.argmax(place0_conf)

            argmax = np.unravel_index(argmax, shape=place0_conf.shape)
            
            p0place_pix = argmax[:2]
            p0place_theta = argmax[2] 
            p0place_pixs.append(p0place_pix)
            p0place_thetas.append(p0place_theta)

        if self.pick_and_place_policy == "default":
            p0pick_pix = pick0pick_pixs[0]
            p0pick_theta = pick0pick_thetas[0] * (2 * np.pi / place0_conf.shape[2])
            p0place_pix = p0place_pixs[0]
            p0place_theta = p0place_thetas[0] * (2 * np.pi / place0_conf.shape[2])
        elif self.pick_and_place_policy == "probability_product":
            probabilities = []
            for i in range(len(pick0pick_pixs)):
                #print('pick0_conf shape', pick0_conf.shape)
                #print('place0_conf shape', place0_confs[i].shape)
                probabilities.append(pick0_conf[pick0pick_pixs[i][0], pick0pick_pixs[i][1], pick0pick_thetas[i]] * \
                                    place0_confs[i][p0place_pixs[i][0], p0place_pixs[i][1], p0place_thetas[i]])
            
            
            
            # choose the idx among the top 3 probabilities
            idxes = np.argsort(probabilities)
            
            ii = -1
            while (self.last_pick is not None) and (np.linalg.norm(np.asarray(self.last_pick) - np.asarray(pick0pick_pixs[idxes[ii]])) < 3) and ii >= -5:
                #print('next pick')
                ii -= 1
            
            idx = idxes[ii]
            p0pick_pix = pick0pick_pixs[idx]
            p0pick_theta = pick0pick_thetas[idx] * (2 * np.pi / place0_conf.shape[2])
            p0place_pix = p0place_pixs[idx]
            p0place_theta = p0place_thetas[idx] * (2 * np.pi / place0_conf.shape[2])
            self.last_pick = p0pick_pix

            place0_conf = self.transport.forward(
                    np.stack([img]), 
                    np.stack([pick0pick_pixs[idx]])).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            
            # normalise the place0_conf
            place_heat = (place0_conf - np.min(place0_conf))/(np.max(place0_conf) - np.min(place0_conf))
            place_heat = cv2.applyColorMap((place_heat * 255).astype(np.uint8), cv2.COLORMAP_SPRING)
            self.state['place-heat-chosen-pick'] = place_heat
        
        elif self.pick_and_place_policy == "sample":
            i = np.random.randint(0, len(pick0pick_pixs))
            p0pick_pix = pick0pick_pixs[i]
            p0pick_theta = pick0pick_thetas[i] * (2 * np.pi / place0_conf.shape[2])
            p0place_pix = p0place_pixs[i]
            p0place_theta = p0place_thetas[i] * (2 * np.pi / place0_conf.shape[2])

            place0_conf = self.transport.forward(
                    np.stack([img]), 
                    np.stack([pick0pick_pixs[i]])).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            
            # normalise the place0_conf
            place_heat = (place0_conf - np.min(place0_conf))/(np.max(place0_conf) - np.min(place0_conf))
            place_heat = cv2.applyColorMap((place_heat * 255).astype(np.uint8), cv2.COLORMAP_SPRING)
            self.state['place-heat-chosen-pick'] = place_heat


        if self.pixel2world:
            # Pixels to end effector poses.
            hmap = img[:, :, 3]
            p0pick_xyz = utils.pix_to_xyz(p0pick_pix, hmap, self.bounds, self.pix_size)
            p0place_xyz = utils.pix_to_xyz(p0place_pix, hmap, self.bounds, self.pix_size)
            p0pick_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0pick_theta))
            p0place_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0place_theta))

            return {
                'pose0': (np.asarray(p0pick_xyz), np.asarray(p0pick_xyzw)),
                'pose1': (np.asarray(p0place_xyz), np.asarray(p0place_xyzw))
            }
        else:
            # Pixels to end effector poses.
            p0pick = (np.asarray(p0pick_pix) / self.in_shape[0] * 2 - 1).astype(np.float32)
            p0place = (np.asarray(p0place_pix) / self.in_shape[1] * 2 - 1).astype(np.float32)
            
        

            action_ = np.stack((p0pick, p0place), axis=0)
            action = action_.copy()
            action[:, 0] = action_[:, 1]
            action[:, 1] = action_[:, 0]

            return action

    def get_checkpoint_names(self, n_iter):
        attention_fname = 'attention-ckpt-%d.pth' % n_iter
        transport_fname = 'transport-ckpt-%d.pth' % n_iter

        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname = os.path.join(self.models_dir, transport_fname)

        ret = {
            'attention': attention_fname,
            'transport': transport_fname
        }

        if self.two_picker:
            attention1_fname = 'attention1-ckpt-%d.pth' % n_iter
            attention1_fname = os.path.join(self.models_dir, attention1_fname)
            ret['attention1'] = attention1_fname

        return ret

    def load(self, n_iter=None, verbose=False):
        """Load pre-trained models."""

        if n_iter is None:
            ### Get the last checkpoint
            if not os.path.exists(self.models_dir):
                return 0
            
            checkpoints = os.listdir(self.models_dir)

            if len(checkpoints) == 0:
                return 0
            
            checkpoints = [int(c.split('.')[0].split('-')[-1]) \
                           for c in checkpoints if 'attention' in c]

            ## Sort checkpoints and get the highest one
            checkpoints = sorted(checkpoints)
            n_iter = checkpoints[-1]
            
            

        ret = self.get_checkpoint_names(n_iter)
        attention_fname = ret['attention']
        print('attention_fname', attention_fname)
        transport_fname = ret['transport']

        self.attention.load(attention_fname, verbose)
        self.transport.load(transport_fname, verbose)
        if self.two_picker:
            attention1_fname = ret['attention1']
            self.attention_1.load(attention1_fname, verbose)
        self.current_update_steps = n_iter
        return n_iter

    def save(self, verbose=False):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        ret = self.get_checkpoint_names(
            self.current_update_steps)
        attention_fname = ret['attention']
        transport_fname = ret['transport']


        self.attention.save(attention_fname, verbose)
        self.transport.save(transport_fname, verbose)

        if self.two_picker:
            attention1_fname = ret['attention1']
            self.attention_1.save(attention1_fname, verbose)


# -----------------------------------------------------------------------------
# Other Transporter Variants
# -----------------------------------------------------------------------------


class OriginalTransporterAgent(TransporterAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            # preprocess=kwargs['preprocess'],
            encoder_version=kwargs['encoder_version'],
            verbose=kwargs['verbose'],
            optimiser=kwargs['attention_optimiser']
        )

        self.transport = Transport(
            #in_shape=self.in_shape,
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            # preprocess=kwargs['preprocess'],
            verbose=kwargs['verbose'],
            key_optimiser=kwargs['key_optimiser'],
            query_optimiser=kwargs['query_optimiser'],
            neg_samples = (kwargs['neg_samples'] if 'neg_samples' in kwargs else 0),
        )



class TwoPickerTransporterAgent(TransporterAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            # preprocess=kwargs['preprocess'],
            encoder_version=kwargs['encoder_version'],
            verbose=kwargs['verbose'])
        
        self.attention_1 = Attention(
            in_shape=(*self.in_shape[:2], 2 + self.in_shape[2]),
            n_rotations=1,
            # preprocess=kwargs['preprocess'],
            encoder_version=kwargs['encoder_version'],
            verbose=kwargs['verbose'])
        
        self.transport = Transport(
            #in_shape=self.in_shape,
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=kwargs['preprocess'],
            verbose=kwargs['verbose'])

class GoalNaiveTransporterAgent(TransporterAgent):
    """Naive version which stacks current and goal images through normal Transport."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Stack the goal image for the vanilla Transport module.
        # t_shape = (self.in_shape[0], self.in_shape[1],
        #            int(self.in_shape[2] * 2))

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            
            # preprocess=kwargs['preprocess'],
            verbose=kwargs['verbose'],
            encoder_version=kwargs['encoder_version'])

        self.transport = Transport(
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            # preprocess=kwargs['preprocess'],
            verbose=kwargs['verbose'])

class GoalTransporterAgent(TransporterAgent):
    """Goal-Conditioned Transporters supporting a separate goal FCN."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            # preprocess=kwargs['preprocess'],
            encoder_version=kwargs['encoder_version'],
            verbose=kwargs['verbose'])
        
        self.transport = TransportGoal(
            in_channels=self.in_shape[2]//2,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            # preprocess=kwargs['preprocess'],
            verbose=kwargs['verbose'])

class NoTransportTransporterAgent(TransporterAgent):

    def __init__(self, name, task, n_rotations=36, verbose=False):
        super().__init__(name, task, n_rotations)

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport = Attention(
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            preprocess=utils.preprocess,
            verbose=verbose)


class PerPixelLossTransporterAgent(TransporterAgent):

    def __init__(self, name, task, n_rotations=36, verbose=False):
        super().__init__(name, task, n_rotations)

        self.attention = Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            verbose=verbose)
        self.transport = TransportPerPixelLoss(
            in_channels=self.in_shape[2],
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            verbose=verbose)