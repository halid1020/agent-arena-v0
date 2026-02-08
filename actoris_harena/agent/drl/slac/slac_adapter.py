
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch

import cv2
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from agent.algorithm.slac.algo import SlacAlgorithm
from agent.algorithm.slac.trainer import Trainer, SlacObservation
from agent.algorithm.slac.buffer import ReplayBuffer
# from logger.visualisation_utils import *
# from registration.data_transformer import *

class SLAC_Adapter():

    def __init__(self, config):

        ### Update Config with default setting
        ## config Dotmap to dict
        
        
        self.config = config
        self.log_dir = self.config.save_dir
        self.model_dir = os.path.join(self.log_dir, "model")
        self.algo = SlacAlgorithm(
            state_shape=self.config.observation_shape,
            action_shape=self.config.action_shape,
            action_repeat=self.config.action_repeat,
            device=torch.device(self.config.device),
            seed=self.config.seed,
            buffer_size=self.config.buffer_size,
            reward_scale=self.config.reward_scale,
            batch_size_latent=self.config.latent_batch_size

        )
        self.cur_state = SlacObservation(
            config.observation_shape, 
            config.action_shape, 
            self.config.num_sequences)
        
        transform_config = self.config.transform
        self.transform = TRANSORMER[transform_config.name](transform_config.params)

        

    def train_online(self, env, loss_logger, eval_logger):
        self.trainer = Trainer(
            env=env,
            env_test=env,
            algo=self.algo,
            log_dir=self.log_dir,
            seed=self.config.seed,
            num_steps=self.config.num_steps,
            initial_collection_steps=self.config.initial_collection_steps,
            eval_interval=self.config.eval_interval,
            initial_learning_steps=self.config.initial_learning_steps,
            config=self.config
        )

        self.trainer.train()

    def process_episode_to_buffer(self, episode, train=False):
        

        ### RGB needs to 0-255 and 64, 64.

        episode = self.transform(
            episode, train=train, to_tensor=False, single=True) # Make transform also works for single-batch episode.

        observations =  episode['rgb']
        actions = episode['action']
        rewards = episode['reward']
        # obs = cv2.resize(observations[0].transpose(1, 2, 0), (64, 64)).transpose(2, 0, 1).astype(np.float32)
        self.algo.buffer.reset_episode(observations[0])
        for j in range(len(actions)):
            obs = observations[j+1]
            self.algo.buffer.append(
                actions[j],
                rewards[j],
                False, #mask
                obs,
                False if j < len(actions) else True)
    

    def train_offline(self, env, datasets, loss_logger, eval_logger):
        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(self.log_dir, "log.csv")
        self.ob_test = SlacObservation(
            self.config.observation_shape, 
            self.config.action_shape, 
            self.config.num_sequences)
        
        start_step = self.load_model()

        if start_step + 1 < self.config.initial_learning_steps + self.config.num_steps:
            dataset = datasets['train']
        
            for _ in tqdm(range(self.config.add_initial_episodes)):
                eid = random.randint(0, dataset._N-1)
                episode = dataset.get_episode(eid)
                self.process_episode_to_buffer(episode, train=True)

       

        print('start_step', start_step)
        print('initial_learning_steps', self.config.initial_learning_steps)

        if start_step + 1 < self.config.initial_learning_steps:
      
            for step in tqdm(range(start_step+1, self.config.initial_learning_steps), 
                        desc="Updating latent variable model."):
                self.algo.update_latent(self.writer)
                if step % self.config.eval_interval == 0:
                    self.evaluate(step, env)
                    self.algo.save_model(os.path.join(self.model_dir, f"step{step}"))
                    env.set_train() ## halid: added
        
            self.algo.save_model(os.path.join(self.model_dir, f"step{self.config.initial_learning_steps-1}"))
            start_step = self.config.initial_learning_steps-1

        for step in tqdm(range(start_step + 1, self.config.num_steps + self.config.initial_learning_steps)):
            # Update the algorithm.
            self.algo.update_latent(self.writer)
            if self.config.update_sac:
                self.algo.update_sac(self.writer)

            # Evaluate regularly.
            if step % self.config.eval_interval == 0:
                self.evaluate(step, env)
                self.algo.save_model(os.path.join(self.model_dir, f"step{step}"))
                env.set_train() ## halid: added
            
            if step % self.config.add_episode_interval == 0:

                for _ in range(self.config.add_episode_num):
                    eid = random.randint(0, dataset._N-1)

                    
                    self.process_episode_to_buffer(dataset.get_episode(eid))

                  
                   



        self.algo.save_model(os.path.join(self.model_dir, f"step{self.config.num_steps-1}"))

    def load_model(self):
        
        if not os.path.exists(self.model_dir):
            return -1
        
        ## Find the latest model in the model directory
        model_files = os.listdir(self.model_dir)
        ## it looks like step0, step1000, step2000, ...
        steps = [int(model_file.split('step')[1]) for model_file in model_files]
        latest_step = max(steps)
        print(f"Loading model from step {latest_step}")
        self.algo.load_model(os.path.join(self.model_dir, f"step{latest_step}") )
        return latest_step


    
    def train(self, datasets, env, loss_logger, eval_logger):
        if self.config.train_mode == 'online':
            self.train_online(env, loss_logger, eval_logger)
        elif self.config.train_mode == 'offline':
            self.train_offline(env, datasets, loss_logger, eval_logger)
        else:
            raise NotImplementedError
        
    def visualise(self, datasets):
        self._visualise(datasets['train'], train=True)
        self._visualise(datasets['test'], train=False)
        

    def _visualise(self, dataset, train=False):

        #print('trans model', self.model.dynamics)
        
        train_str = 'Train' if train else 'Eval'
        eval_action_horizon = dataset.eval_action_horizon
        old_cur_state = self.cur_state
        self.cur_state = SlacObservation(self.config.observation_shape, self.config.action_shape, 21)
    
        
        for e in range(5):
            data = dataset.get_episode(e)
            #org_gt = dataset.transform.post_transform(data)
            
            
            recon_image = []
            
            plot_trajectory(
                data['rgb'][6:16].transpose(0, 2 ,3, 1),
                data['action'][6:16],
                title='{} Ground Truth Episode {}'.format(train_str, e),
                # rewards=data['reward'][5:15], 
                save_png = True, 
                save_path=os.path.join(self.config.save_dir, 'visualisations'))
            
            self.init_state({'observation': {'image': data['rgb'][0].transpose(1, 2, 0)}})
            for step in range(len(data['action'])):
                self.update_state({'observation': {'image': data['rgb'][step+1].transpose(1, 2, 0)}}, 
                                  data['action'][step])
            

            observations, actions = self.cur_state.state, self.cur_state.action
            # print('observations shape', observations.shape)
            # print('max observations', observations.max())
            observations = torch.tensor(observations, dtype=torch.uint8, device=self.config.device).float().div_(255.0) - 0.5
            actions = torch.tensor(actions, dtype=torch.float, device=self.config.device).reshape(1, -1, self.config.action_shape[0])
            

            feature_ = self.algo.latent.encoder(observations)
            z1_mean_post_, z1_std_post_, z1, z2 = self.algo.latent.sample_posterior(feature_, actions)
           
            z= torch.cat([z1, z2], dim=-1)
            state_mean_, state_std_ = self.algo.latent.decoder(z)
            state_mean_ = ((state_mean_.detach().cpu().numpy() + 0.5) * 255.0).astype(np.uint8).clip(0, 255)


            plot_trajectory(
                state_mean_[0, 6:16].transpose(0, 2, 3, 1),
                title='{} Posterior Trajectory Episode {}'.format(train_str, e), 
                save_png = True,
                save_path=os.path.join(self.config.save_dir, 'visualisations'))

            recon_image.append( state_mean_[0, 6:11].transpose(0, 2 ,3, 1))

            
            for horizon in [1, 2, 3, 4, 5]:
                horizon_actions = [actions[:, j+1: j+horizon+1] for j in range(eval_action_horizon-horizon)]
                horizon_actions = torch.stack(horizon_actions).squeeze(1)

                ### repeat  z2_[:, -1, :] to have shape candidates  * 1 * z2_.shape[-1]
                prz1 = [z1[:, j+1] for j in range(eval_action_horizon-horizon)]
                prz2 = [z2[:, j+1] for j in range(eval_action_horizon-horizon)]
                prz1 = torch.stack(prz1).squeeze(1)
                prz2 = torch.stack(prz2).squeeze(1)

                # print('horizon actions shape', horizon_actions.shape)
                # print('z1 shape', prz1.shape)
                # print('z2 shape', prz2.shape)


                # z2 = z2[:, -1, :].repeat(candidates, 1, 1).reshape(candidates, -1)

                # future_actions = torch.tensor(actions, dtype=torch.float, device=self.config.device).reshape(candidates, -1, self.config.action_shape[0])

                # z1_mean_init, z1_std_init = self.algo.latent.z1_prior_init(future_actions[:, 0])
                # z1 = z1_mean_init + torch.randn_like(z1_std_init) * z1_std_init
                # # z1_mean_ = [z1_mean_init]
                # # z1_std_ = [z1_std_init]
                prz1_ = []
                prz2_ = []

                for t in range(1, horizon_actions.size(1)+ 1):
                    prz1_mean, prz1_std = self.algo.latent.z1_prior(torch.cat([prz2, horizon_actions[:, t-1]] , dim=-1))
                    prz1 = prz1_mean + torch.randn_like(prz1_std) * prz1_std


                    prz2_mean, prz2_std = self.algo.latent.z2_posterior(torch.cat([prz1, prz2, horizon_actions[:, t - 1]], dim=1))
                    prz2 = prz2_mean + torch.randn_like(prz2_std) * prz2_std

                    # z1_mean_.append(z1_mean)
                    # z1_std_.append(z1_std)
                    prz1_.append(prz1)
                    prz2_.append(prz2)
                

                # # z1_mean_ = torch.stack(z1_mean_, dim=1)
                # # z1_std_ = torch.stack(z1_std_, dim=1)
                prz1_ = torch.stack(prz1_, dim=1)
                prz2_ = torch.stack(prz2_, dim=1)

                # print('z1_ shape', prz1_.shape)
                # print('z2_ shape', prz2_.shape)

                prz= torch.cat([prz1_, prz2_], dim=-1)
                state_mean_, state_std_ = self.algo.latent.decoder(prz)
                state_mean_ = ((state_mean_.detach().cpu().numpy() + 0.5) * 255.0).astype(np.uint8).clip(0, 255)
                #print('state mean shape', state_mean_.shape)


                
                plot_trajectory(
                    state_mean_[6-horizon:16-horizon, -1].transpose(0, 2, 3, 1),
                    # rewards=imagin_reward[5-horizon:15-horizon], 
                    title='{}-Step {} Prior Trajectory Episode {}'.format(horizon, train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.config.save_dir, 'visualisations'))
                recon_image.append(state_mean_[5-horizon+5+horizon:6-horizon+5+horizon, -1].transpose(0, 2 ,3, 1))
            
            recon_image = np.concatenate(recon_image, axis=0)
            plot_trajectory(
                    recon_image,
                    # rewards=posterior_rewards[6:16], 
                    title='{} Recon Trajectory Episode {}'.format(train_str, e), 
                    save_png = True,
                    save_path=os.path.join(self.config.save_dir, 'visualisations'))

        self.cur_state = old_cur_state
    
    def evaluate(self, step_env, env):
        mean_return = 0.0

        for i in range(self.config.num_eval_episodes):
            env.set_eval() ## halid: added
            state = env.reset(episode_id=i) ## halid: modified
            obs = cv2.resize(state['observation']['image'], (64, 64)).transpose(2, 0, 1).astype(np.float32)  ## halid: modified
            self.ob_test.reset_episode(obs) ## halid: modified
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(self.ob_test)
                # state, reward, done, _ = self.env_test.step(action)
                external_state = env.step(action) ## halid: modified
                obs, reward, done, = external_state['observation']['image'], external_state['reward'], external_state['done'] ## halid: modified
                obs = cv2.resize(obs, (64, 64)).transpose(2, 0, 1).astype(np.float32) ## halid: modified
                self.ob_test.append(obs, action)
                episode_return += reward

            mean_return += episode_return / self.config.num_eval_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}")


    def reset(self):
        pass

    def init_state(self, state):
        obs = cv2.resize(state['observation']['image'], (64, 64)).transpose(2, 0, 1)
        self.cur_state.reset_episode(obs)


    def update_state(self, state, action):
        obs = cv2.resize(state['observation']['image'], (64, 64)).transpose(2, 0, 1)
        self.cur_state.append(obs, action)

    def sample_action(self, state, env):
        
        action = self.algo.exploit(self.cur_state)
        return action
    
    def unroll_action_from_cur_state(self, actions):
        candidates = actions.shape[0]

        pre_obs, pre_actions = self.cur_state.state, self.cur_state.action
        pre_obs = torch.tensor(pre_obs, dtype=torch.uint8, device=self.config.device).float().div_(255.0) - 0.5
        pre_actions = torch.tensor(pre_actions, dtype=torch.float, device=self.config.device).reshape(1, -1, self.config.action_shape[0])
        

        feature_ = self.algo.latent.encoder(pre_obs)
        z1_mean_post_, z1_std_post_, z1, z2 = self.algo.latent.sample_posterior(feature_, pre_actions)

        ### repeat  z2_[:, -1, :] to have shape candidates  * 1 * z2_.shape[-1]
        z2 = z2[:, -1, :].repeat(candidates, 1, 1).reshape(candidates, -1)
        z1 = z1[:, -1, :].repeat(candidates, 1, 1).reshape(candidates, -1)

        future_actions = torch.tensor(actions, dtype=torch.float, device=self.config.device).reshape(candidates, -1, self.config.action_shape[0])

        # z1_mean_init, z1_std_init = self.algo.latent.z1_prior_init(future_actions[:, 0])
        # z1 = z1_mean_init + torch.randn_like(z1_std_init) * z1_std_init
        # z1_mean_ = [z1_mean_init]
        # z1_std_ = [z1_std_init]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, future_actions.size(1)+ 1):
            z1_mean, z1_std = self.algo.latent.z1_prior(torch.cat([z2, future_actions[:, t-1]] , dim=-1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std


            z2_mean, z2_std = self.algo.latent.z2_posterior(torch.cat([z1, z2, future_actions[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            # z1_mean_.append(z1_mean)
            # z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)
        

        # z1_mean_ = torch.stack(z1_mean_, dim=1)
        # z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)


        return {
            'z1': z1_,
            'z2': z2_,
            'action': future_actions
        }
    
    def cost_fn(self, trajectory):
        z1, z2, action = trajectory['z1'], trajectory['z2'], trajectory['action']
        
        z_ = torch.cat([z1, z2], dim=-1)

        x = torch.cat([z_[:, :-1], action, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.algo.latent.reward(x.view(B * S, X))

        returns = reward_mean_.sum(dim=1).detach().cpu().numpy()

        return -returns