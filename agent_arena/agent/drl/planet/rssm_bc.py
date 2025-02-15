import os
import torch
from torch.nn import functional as F

from agent.algorithm.dreamer_rssm import *
from agent.utilities.torch_utils import *
from agent.planet.rssm import RSSM
from agent.behaviour_cloning.algo import BehaviourCloning
import api as ag_ar

class RSSM_BC(BehaviourCloning):
    
    def __init__(self, config):
        
        super().__init__(config)

        self.config = config
        #self.config.model_dir = os.path.join(self.config.save_dir, 'model')
        
        self.actor_config = config.actor_params

        rssm_configs = ag_ar.retrieve_config(config.rssm_config)
        rssm_configs.save_dir = config.rssm_save_dir
        self.rssm = RSSM(rssm_configs)
        self.feat_size = rssm_configs.deterministic_latent_dim + \
            rssm_configs.stochastic_latent_dim
        
        
        
        self._init_actor()

    def init(self, obs):
        self.rssm.init_state(obs)
    
    def update(self, state, action):
        self.rssm.update(state, action)

    def get_name(self):
        return "Behaviour Clonining on RSSM's representation"
    
    def reset(self):
        pass

    def act(self, state, arena=None):
        latent_state = torch.cat(
            [self.rssm.cur_state['deter'], 
             self.rssm.cur_state['stoch']['sample']], dim=-1)
        
        pred_act = self.actor(latent_state)
        if 'deter' not in self.actor_config.actor_dist:
            pred_act = pred_act.mode()
        pred_act = pred_act.squeeze(0).squeeze(0).detach().cpu().numpy()
        return pred_act

    def set_train(self):
        self.rssm.set_eval()
        self.actor.train()

    def set_eval(self):
        self.rssm.set_eval()
        self.actor.eval()
        
    
    def process_state(self, state):
        return state['observation'][self.rssm.input_obs]
    
    def load(self, path=None):
        
        self.rssm.load()

        if path is None:
            path = os.path.join(self.config.save_dir, 'model')

        actor_dir = os.path.join(path, 'actor.pth')
        if not os.path.exists(actor_dir):
            print('No actor found at {}'.format(actor_dir)) 
            return {}
        
        checkpoint = torch.load(actor_dir)
        self.actor.load_state_dict(checkpoint['model'])
        self._actor_opt._opt.load_state_dict(checkpoint['optimser'])
        return {
            'update_step': checkpoint['update_step']
        }
    
    def save(self, path=None):

        if path is None:
            path = os.path.join(self.config.save_dir, 'model')
        
        os.makedirs(path, exist_ok=True)

        model_dict = {
            'model': self.actor.state_dict(),
            'optimser': self._actor_opt._opt.state_dict(),
            'update_step': self.update_step
        }

        torch.save(
            model_dict, 
            os.path.join(path, 'actor.pth')
        )
    

    def update_actor(self):
        
        states, actions = next(iter(self.train_dataloader))
        loss = self._compute_loss(states, actions)
        metrics = self._actor_opt(loss['total_loss'], self.actor.parameters())
    

    def _preprocess(self, data):
        data = self.rssm.model.preprocess(data)
        return data
    
    def _init_actor(self):
        actor_config = self.config.actor_params
        self.actor = ActionHead(self.feat_size,  # pytorch version
            actor_config.num_actions, actor_config.actor_layers, actor_config.actor_units, 
            ACTIVATIONS[actor_config.actor_act],
            actor_config.actor_dist, actor_config.actor_init_std, actor_config.actor_min_std,
            actor_config.actor_dist, actor_config.actor_temp, actor_config.actor_outscale).to(actor_config.device)

        
        
        kw = dict(wd=actor_config.weight_decay, opt=actor_config.actor_opt, use_amp=False)
        self._actor_opt = Optimizer(
            'actor', self.actor.parameters(), 
            actor_config.actor_lr, actor_config.opt_eps, actor_config.actor_grad_clip,**kw)

    

    def _compute_loss(self, obs, actions):
        
        # embed = self.rssm.model.encoder(data)
        # post, prior = self.rssm.model.dynamics.observe(
        #     embed[:, 1:], 
        #     data['action'][:, :-1].float().float(),
        #     state=None,
        #     terminal=data['terminal'][:, :-1].float())

        init_belief = torch.zeros(
            self.config.batch_size, 
            self.rssm.config.deterministic_latent_dim).to(self.config.device)

        init_state = torch.zeros(
            self.config.batch_size, 
            self.rssm.config.stochastic_latent_dim).to(self.config.device)
        
        unroll_no_ops = np_to_ts(np.asarray(self.no_op), self.config.device)\
            .repeat(self.config.batch_size, 1)
        
        blfs, post, prior, embd = self.rssm._unroll_state_action(
            obs,
            unroll_no_ops,
            init_belief, 
            init_state,
        )

        latent_state = torch.cat(
            [blfs, 
             post['sample']], dim=-1)
        
        pred_act = self.actor(latent_state)
        
        if self.actor_config.actor_loss == 'mse':
            if 'deter' not in self.actor_config.actor_dist:
                pred_act = pred_act.mode()
            loss = F.mse_loss(pred_act, actions)
        elif self.actor_config.actor_loss == 'nll':
            loss = -pred_act.log_prob(actions).mean()
        else:
            raise NotImplementedError
        return {'total_loss': loss}