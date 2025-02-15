import os

from tqdm import tqdm
import numpy as np
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from pathlib import Path
from dotmap import DotMap

from agent.algorithm.dreamer_rssm import *
from agent.utilities.torch_utils import *
# from logger.visualisation_utils import *
from agent.algorithm.rssm_bc import RSSM_BC
# from agent.transformations.pick_and_place_heatmap_transformer import get_action_from_heatmap, create_heatmap

class RSSM_BC_Pick_And_Place_Heatmap(RSSM_BC):
    
    def __init__(self, config):
        super().__init__(config)

        # self.config = config
        # self.config.model_dir = os.path.join(self.config.save_dir, 'model')
        
        # self.actor_config = config.actor_params

        # configs = yaml.safe_load(Path('yamls/{}.yaml'.format(config.dynamic_model_config)).read_text())
        # dynamic_model_config = DotMap(configs)
        #self.dynamic_model = Dreamer(self.config)
        
    
    def init_actor(self):
        actor_config = self.config.actor_params
        H, W = tuple(actor_config.output_shape)

        if not self.config.pick_condition_place:
            self.actor = ConvDecoder(
                self.feat_size,  # pytorch version
                actor_config.cnn_depth, 
                ACTIVATIONS[actor_config.actor_act], 
                (2, H, W), 
                actor_config.decoder_kernels,
                actor_config.decoder_thin, 
                output_mode=actor_config.output_mode).to(actor_config.device)
        else:
            self.pick_actor = ConvDecoder(
                self.feat_size,  # pytorch version
                actor_config.cnn_depth, 
                ACTIVATIONS[actor_config.actor_act], 
                (1, H, W), 
                actor_config.decoder_kernels,
                actor_config.decoder_thin, 
                output_mode=actor_config.output_mode).to(actor_config.device)
            
            self.place_actor = ConvDecoder(
                self.feat_size + 2,  # pytorch version
                actor_config.cnn_depth, 
                ACTIVATIONS[actor_config.actor_act],
                (1, H, W), 
                actor_config.decoder_kernels,
                actor_config.decoder_thin, 
                output_mode=actor_config.output_mode).to(actor_config.device)
            

        
        
        kw = dict(wd=actor_config.weight_decay, opt=actor_config.actor_opt, use_amp=False)
        self.params = list(self.pick_actor.parameters())
        if self.config.pick_condition_place:
            self.params.extend(list(self.place_actor.parameters()))

        self._actor_opt = Optimizer(
            'actor', self.params, 
            actor_config.actor_lr, actor_config.opt_eps, actor_config.actor_grad_clip,**kw)

    
    def compute_loss(self, data):

        embed = self.dynamic_model.model.encoder(data)
        heatmap_action = torch.concat(
            [data['pick_heatmap'].unsqueeze(2), data['place_heatmap'].unsqueeze(2)], 
            dim=2).permute(0, 1, 3, 4, 2)
        
        ### print the statistics of heatmap_action
        post, prior = self.dynamic_model.model.dynamics.observe(
            embed[:, 1:], 
            data['action'][:, :-1].float(),
            state=None,
            terminal=data['terminal'][:, :-1].float())

        if not self.config.pick_condition_place:
            pred_heatmap = self.actor(get_feat(post, self.dynamic_model.config))
        else:
            pick_pred_heatmap = self.pick_actor(get_feat(post, self.dynamic_model.config))

            place_pred_heatmap = self.place_actor(
                torch.cat([get_feat(post, self.dynamic_model.config),  data['action'][:, :-1, :2].float()], dim=2))    
            pred_heatmap = torch.cat([pick_pred_heatmap, place_pred_heatmap], dim=4)


        
        if self.actor_config.actor_loss == 'mse':
            if self.actor_config.output_mode == 'stochastic':
                pred_heatmap = pred_heatmap.mode()
            loss = F.mse_loss(pred_heatmap, heatmap_action[:, 1:])
        elif self.actor_config.actor_loss == 'nll':
            loss = -pred_heatmap.log_prob(heatmap_action[:, 1:]).mean()
        else:
            raise NotImplementedError
        
        res = {}
        if self.actor_config.pick_action_feature_loss_scale > 0:
            pick_action_feature_loss = F.mse_loss(data['action'][:, :-1, :2].float(), torch.zeros_like(data['action'][:, :-1, :2].float())) 
            loss += self.config.pick_action_feature_loss_scale * pick_action_feature_loss
            res['pick_action_feature_loss'] = pick_action_feature_loss
        
        res['total_loss'] = loss
        return res

    def preprocess(self, data):
        res = self.dynamic_model.model.preprocess(data)
        #print(res.keys())
        res['pick_heatmap'] = data['pick_heatmap']
        res['place_heatmap'] = data['place_heatmap']

        ##################################################
        # Following code is needed for making the algorithm work.
        ##################################################

        ## Add no_op action at the beginning
        ## config.no_op has the no_op action as list
        no_op = np.asarray(self.config.no_op, dtype=np.float32)
        ts_no_op = np_to_ts(no_op, self.config.device)
        res['action'] = torch.cat([torch.zeros_like(res['action'][:, :1]), res['action']], dim=1)
        res['action'][:, 0] = ts_no_op

        ## Add non terminal at the beginning
        if 'terminal' in res.keys():
            res['terminal'] = torch.cat([torch.zeros_like(res['terminal'][:, :1]), res['terminal']], dim=1)

        # Repeat the first image
        res['input_image'] = torch.cat([res['input_image'][:, :1], res['input_image']], dim=1)

        # Convert the no_op to heatmap
        H, W =  res['pick_heatmap'].shape[-2:]
        pick_heatmap = create_heatmap(ts_no_op[:2], size=(H, W), sigma=self.config.heatmap_sigma, kernel_size=self.config.heatmap_kernel_size) * self.config.heatmap_scale
        place_heatmap = create_heatmap(ts_no_op[2:], size=(H, W), sigma=self.config.heatmap_sigma, kernel_size=self.config.heatmap_kernel_size) * self.config.heatmap_scale
        res['pick_heatmap'] = torch.cat([torch.zeros_like(res['pick_heatmap'][:, :1]), res['pick_heatmap']], dim=1)
        res['pick_heatmap'][:, 0] = np_to_ts(pick_heatmap, self.config.device)
        res['place_heatmap'] = torch.cat([torch.zeros_like(res['place_heatmap'][:, :1]), res['place_heatmap']], dim=1)
        res['place_heatmap'][:, 0] = np_to_ts(place_heatmap, self.config.device)

        ##################################################
        # Above code is needed for making the algorithm work.
        ##################################################

        return res
    
    # state: B*T*state_size
    # action_hetmap: B*T*H*W*2

    def get_action_heatmap(self, state):

    
        if not self.config.pick_condition_place:
            heatmap = self.actor(get_feat(state, self.dynamic_model.config))
            if self.actor_config.output_mode == 'stochastic':
                heatmap = heatmap.mode()
        else:
            
            pick_heatmap = self.pick_actor(get_feat(state, self.dynamic_model.config))
            if self.actor_config.output_mode == 'stochastic':
                pick_heatmap = pick_heatmap.mode()

            pick_action = get_action_from_heatmap(
                ts_to_np(torch.cat([pick_heatmap, pick_heatmap], dim=4)).transpose(0, 1, 4, 2, 3)
            )[:, :, :2].clip(self.action_lower_bound[:2], self.action_upper_bound[:2])

            place_heatmap = self.place_actor(
                torch.cat([get_feat(state, self.dynamic_model.config), np_to_ts(pick_action, pick_heatmap.device)], dim=2))    
            if self.actor_config.output_mode == 'stochastic':
                place_heatmap = place_heatmap.mode()

            heatmap = torch.cat([pick_heatmap, place_heatmap], dim=4)
        
        return heatmap

    def sample_action(self, state, env=None):
        
        heatmap = self.get_action_heatmap(state)
        
        heatmap = ts_to_np(heatmap.squeeze(0).squeeze(0)).transpose(2, 0, 1)
        plot_trajectory(
                np.expand_dims(heatmap, axis=-1),
                title='current_heatmap',
                save_png = True, 
                save_path='.')
        
        action = get_action_from_heatmap(heatmap)
        #print('action', action)
        action = action.clip(self.action_lower_bound, self.action_upper_bound)
        #print('clip action', action)
        return action

    def evaluate(self, dataset):
        self.visualise(dataset) ## TODO: refine
        heatmap_mse = []
        action_mse = []

        for i in tqdm(range(100), desc='evaluatuion is going on'):
            data = dataset[i]
            for k, v in data.items():
                data[k] = v.unsqueeze(0)

            data = self.preprocess(data)

            embed = self.dynamic_model.model.encoder(data)
            post, prior = self.dynamic_model.model.dynamics.observe(
                embed[:, 1:], 
                data['action'][:, :-1].float(),
                state=None,
                terminal=data['terminal'])

            gt_heatmap = torch.concat(
                [data['pick_heatmap'].unsqueeze(2), data['place_heatmap'].unsqueeze(2)], 
                dim=2)

            pred_heatmap = self.get_action_heatmap(post)

            pred_heatmap = pred_heatmap.permute(0, 1, 4, 2, 3)
            pred_action = np_to_ts(
                get_action_from_heatmap(ts_to_np(pred_heatmap)).clip(self.action_lower_bound, self.action_upper_bound), 
                self.actor_config.device)

            heatmap_diff = ts_to_np(F.mse_loss(pred_heatmap, gt_heatmap[:, 1:]))
            action_diff = ts_to_np(F.mse_loss(pred_action, data['action'][:, :-1]))
        
            heatmap_mse.append(heatmap_diff)
            action_mse.append(action_diff)

        results = {
            'heatmap_mse': np.mean(heatmap_mse),
            'action_mse': np.mean(action_mse)
        }

        return results
    
    def visualise(self, datasets):
        self._visualise(datasets['train_bc'], train=True)
        self._visualise(datasets['test_bc'], train=False)

    def _visualise(self, dataset, train=False):
        train_str = 'Train' if train else 'Eval'

        for e in range(3):
            data = dataset.get_episode(e, transform=True, train=False)

            plot_trajectory(
                ts_to_np(data['pick_heatmap'].unsqueeze(-1)),
                title='{} Episode {} GT Pick Heatmap'.format(train_str, e),
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))
            
            plot_trajectory(
                ts_to_np(data['place_heatmap'].unsqueeze(-1)),
                title='{} Episode {} GT Place Heatmap'.format(train_str, e),
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))

            org_gt = dataset.transform.post_transform(data)
            true_actions = ts_to_np(data['action'])


            for k, v in data.items():
                data[k] = v.unsqueeze(0)
            data = self.preprocess(data)

            embed = self.dynamic_model.model.encoder(data)

            ####################
            post, prior = self.dynamic_model.model.dynamics.observe(
                embed[:, 1:], 
                data['action'][:, :-1].float(),
                state=None,
                terminal=None)
            pred_heatmap = ts_to_np(self.get_action_heatmap(post).squeeze(0)).transpose(0, 3, 1, 2)
            pred_action = get_action_from_heatmap(pred_heatmap).clip(self.action_lower_bound, self.action_upper_bound)
            ##########################
            
      
        
            plot_trajectory(
                org_gt['observation'].transpose(0, 2 ,3, 1)[:-1],
                acts1=true_actions,
                acts2=pred_action,
                title='{} Episode {} Predicted Actions'.format(train_str, e),
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))
            
            plot_trajectory(
                np.expand_dims(pred_heatmap[:, 0], axis=-1),
                title='{} Episode {} Estimated Pick Heatmap'.format(train_str, e),
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))
            
            plot_trajectory(
                np.expand_dims(pred_heatmap[:, 1], axis=-1),
                title='{} Episode {} Estimated Place Heatmap'.format(train_str, e),
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))

    
    def save_checkpoint(self, update_step, path, title):
        
        if not os.path.exists(path):
            os.makedirs(path)

        if self.config.pick_condition_place:
            model_dict = {
                'pick_actor': self.pick_actor.state_dict(),
                'place_actor': self.place_actor.state_dict(),
                'optimser': self._actor_opt._opt.state_dict(),
                'update_step': update_step
            }
        else:
            model_dict = {
                'actor': self.actor.state_dict(),
                'optimser': self._actor_opt._opt.state_dict(),
                'update_step': update_step
            }

        torch.save(
            model_dict, 
            os.path.join(path, '{}.pth'.format(title))
        )

    def load_models(self, model_dir):
        
        if not os.path.exists(model_dir):
            print('No model found at {}'.format(model_dir)) 
            return 0
        
        checkpoint = torch.load(model_dir)

        if self.config.pick_condition_place:
            self.pick_actor.load_state_dict(checkpoint['pick_actor'])
            self.place_actor.load_state_dict(checkpoint['place_actor'])
        else:    
            self.actor.load_state_dict(checkpoint['actor'])
        
        self._actor_opt._opt.load_state_dict(checkpoint['optimser'])
        return checkpoint['update_step'] + 1
    
    def set_eval(self):
        if self.config.pick_condition_place:
            self.pick_actor.eval()
            self.place_actor.eval()
            self.dynamic_model.set_eval()
        else:
            super().set_eval()

    def set_train(self):
        if self.config.pick_condition_place:
            self.pick_actor.train()
            self.place_actor.train()
            self.dynamic_model.set_train()
        else:
            super().set_train()