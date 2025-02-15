import copy
import time
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from dotmap import DotMap
from tqdm import tqdm

from agent.algorithm.curl_sac.curl_sac import CurlSacAgent
from agent.algorithm.curl_sac.default_config import DEFAULT_CONFIG
from agent.algorithm.curl_sac import utils
from agent.algorithm.curl_sac.logger import Logger
# from logger.visualisation_utils import *


def make_agent(obs_shape, action_shape, config, device):
    
    return CurlSacAgent(
        args=config,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=config.hidden_dim,
        discount=config.discount,
        init_temperature=config.init_temperature,
        alpha_lr=config.alpha_lr,
        alpha_beta=config.alpha_beta,
        alpha_fixed=config.alpha_fixed,
        actor_lr=config.actor_lr,
        actor_beta=config.actor_beta,
        actor_log_std_min=config.actor_log_std_min,
        actor_log_std_max=config.actor_log_std_max,
        actor_update_freq=config.actor_update_freq,
        critic_lr=config.critic_lr,
        critic_beta=config.critic_beta,
        critic_tau=config.critic_tau,
        critic_target_update_freq=config.critic_target_update_freq,
        encoder_type=config.encoder_type,
        encoder_feature_dim=config.encoder_feature_dim,
        encoder_lr=config.encoder_lr,
        encoder_tau=config.encoder_tau,
        num_layers=config.num_layers,
        num_filters=config.num_filters,
        log_interval=config.log_interval,
        detach_encoder=config.detach_encoder,
        curl_latent_dim=config.curl_latent_dim
    )


class CurlSAC_Adapter():

    def __init__(self, config):

        ### Update Config with default setting
        ## config Dotmap to dict
        vv = config.toDict()
        updated_vv = copy.copy(DEFAULT_CONFIG)
        updated_vv.update(**vv)
        config = DotMap(updated_vv)


        ### Args will be a dot map

        utils.set_seed_everywhere(config.seed)

        self.config = config

        if config.encoder_type == 'pixel':
            obs_shape = (3, config.image_size, config.image_size)
            pre_aug_obs_shape = (3, config.pre_transform_image_size, config.pre_transform_image_size)


        self.replay_buffer = utils.ReplayBuffer(
            obs_shape=pre_aug_obs_shape,
            action_shape=self.config.action_shape,
            capacity=self.config.replay_buffer_capacity,
            batch_size=config.batch_size,
            device=config.device,
            image_size=config.image_size)
        
        self.agent = make_agent(
            obs_shape=obs_shape,
            action_shape=config.action_shape,
            config=config,
            device=config.device
        )

        self.video_dir = utils.make_dir(os.path.join(config.save_dir, 'video'))
        self.model_dir = utils.make_dir(os.path.join(config.save_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(config.save_dir, 'buffer'))
        

    def train_online(self, env, loss_logger, eval_logger):
        
        L = Logger(
            self.config.save_dir, 
            use_tb=self.config.save_tb, chester_logger=None)
        
        episode, episode_reward, done, ep_info = 0, 0, True, []
        start_time = time.time()
        for step in range(self.config.num_train_steps):
            # evaluate agent periodically

            if step % self.config.eval_freq == 0:
                print('replay buffer size {}/{}'.format(self.replay_buffer.idx, self.replay_buffer.capacity))
                L.log('eval/episode', episode, step)
                self.evaluate(env, self.agent, self.video_dir, self.config.num_eval_episodes, L, step, self.config)
                if self.config.save_model and (step % (self.config.eval_freq * 5) == 0):
                    self.agent.save(self.model_dir, step)
                if self.config.save_buffer:
                    self.replay_buffer.save(self.buffer_dir)
            if done:
                if step > 0:
                    if step % self.config.log_interval == 0:
                        L.log('train/duration', time.time() - start_time, step)
                        # for key, val in get_info_stats([ep_info]).items():
                        #     L.log('train/info_' + key, val, step)
                        L.dump(step)
                    start_time = time.time()
                if step % self.config.log_interval == 0:
                    L.log('train/episode_reward', episode_reward, step)

                env.set_train()
                external_states = env.reset()
                obs = external_states['observation']['image'].transpose(2, 0, 1)
                done = False
                #ep_info = []
                episode_reward = 0
                episode_step = 0
                episode += 1
                if step % self.config.log_interval == 0:
                    L.log('train/episode', episode, step)

            # sample action for data collection
            if step < self.config.init_steps:
                action = env.sample_random_action()
            else:
                with utils.eval_mode(self.agent):
                    #TODO
                    
                    action = self.agent.sample_action(obs)

            # run training update
            if step >= self.config.init_steps:
                num_updates = 1
                for _ in range(num_updates):
                    self.agent.update(self.replay_buffer, L, step)

            external_states = env.step(action)

            next_obs, reward, done = \
                external_states['observation']['image'].transpose(2, 0, 1), external_states['reward'], external_states['done']

            # allow infinit bootstrap
            done_bool = 0 if episode_step + 1 == env.horizon else float(done)
            episode_reward += reward
            self.replay_buffer.add(obs, action, reward, next_obs, done_bool)

            obs = next_obs
            episode_step += 1

    
    def train_offline(self, env, datasets, loss_logger, eval_logger):
        
        L = Logger(
            self.config.save_dir, 
            use_tb=self.config.save_tb, chester_logger=None)
        
        dataset = datasets['train']
        ## load dataset to replay buffer
        print('Loading dataset to replay buffer...')
        for i in tqdm(range(self.config.add_initial_episodes)):
            epsiode = dataset.get_episode(i)
            observations = epsiode['observation']
            actions = epsiode['action']
            rewards = epsiode['reward']

            for j in range(len(actions)):
                self.replay_buffer.add(
                    observations[j],
                    actions[j], 
                    rewards[j], 
                    observations[j+1], 
                    False) # inifit bootstrap
        
        last_episode_idx = self.config.add_initial_episodes - 1


        print('Start offline training...')
        #episode, episode_reward, done, ep_info = 0, 0, True, []
        for step in tqdm(range(0, self.config.offline_update_steps)):
            if step % self.config.eval_freq == 0:
                #L.log('eval/episode', episode, step)
                self.evaluate(env, self.agent, self.video_dir, self.config.num_eval_episodes, L, step, self.config)
                if self.config.save_model and (step % (self.config.eval_freq * 5) == 0):
                    self.agent.save(self.model_dir, step)

            if step % self.config.add_episode_interval == 0:
                print('add episodes')
                for i in range(self.config.add_episode_num):
                    last_episode_idx += 1
                    if last_episode_idx >= dataset._N:
                        last_episode_idx = 0

                    epsiode = dataset.get_episode(last_episode_idx)
                    observations = epsiode['observation']
                    actions = epsiode['action']
                    rewards = epsiode['reward']

                    for j in range(len(actions)):
                        self.replay_buffer.add(
                            observations[j],
                            actions[j], 
                            rewards[j], 
                            observations[j+1], 
                            False)
               

            self.agent.update(self.replay_buffer, L, step)


    def load_model(self):
        ## Find the latest model in the model directory
        model_files = os.listdir(self.model_dir)
        steps = [int(f.split('.')[0].split('_')[-1]) for f in model_files]
        latest_step = max(steps)
        print('Loading model from step {}...'.format(latest_step))
        self.agent.load(
            self.model_dir, 
            latest_step)

    def train(self, datasets, env, loss_logger, eval_logger):
        if self.config.train_mode == 'online':
            self.train_online(env, loss_logger, eval_logger)
        elif self.config.train_mode == 'offline':
            self.train_offline(env, datasets, loss_logger, eval_logger)
        else:
            raise NotImplementedError

    def visualise(self, datasets):
        pass

    def evaluate(self, env, agent, video_dir, num_episodes, L, step, args):
        print('Evaluating curl_sac...')
        all_ep_rewards = []
        env.set_eval()

        def run_eval_loop(sample_stochastically=True):
            start_time = time.time()
            prefix = 'stochastic_' if sample_stochastically else ''
            infos = []
            all_frames = []
            plt.figure()
            for i in range(num_episodes):
                print('eval episode {}/{}'.format(i+1, num_episodes))
                external_state = env.reset(episode_id=i)
                done = False
                episode_reward = 0
                ep_info = []
                frames = [env.render(mode='rgb', resolution=(128, 128))]
                rewards = []
                while not done:

                    obs_image = external_state['observation']['image'].transpose(2, 0, 1)
                    # center crop image
                    if args.encoder_type == 'pixel':
                        obs_image = utils.center_crop_image(obs_image, args.image_size)
                    with utils.eval_mode(agent):
                        if sample_stochastically:
                            action = agent.sample_action()
                        else:
                            action = agent.select_action(obs_image)
                        
                    #print('action shape', action.shape)
                    external_state = env.step(action)
                    done = external_state['done']
                    episode_reward += external_state['reward']
                    # ep_info.append(info)
                    frames.append(env.render(mode='rgb', resolution=(128, 128)))
                    rewards.append(external_state['reward'])
                plt.plot(range(len(rewards)), rewards)
                if len(all_frames) < 8:
                    all_frames.append(frames)
                infos.append(ep_info)

                L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
                all_ep_rewards.append(episode_reward)
            plt.savefig(os.path.join(video_dir, '%d.png' % step))
            all_frames = np.array(all_frames).swapaxes(0, 1)
            all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
            save_numpy_as_gif(all_frames, os.path.join(video_dir, '%d.gif' % step))

            # for key, val in get_info_stats(infos).items():
            #     L.log('eval/info_' + prefix + key, val, step)
            L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
            L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        run_eval_loop(sample_stochastically=False)
        L.dump(step)


    def reset(self):
        pass

    def init(self, state):
        pass

    def update(self, state, action):
        pass

    def act(self, state, env):
        obs_image = state['observation']['image'].transpose(2, 0, 1)
        if self.config.encoder_type == 'pixel':
            obs_image = utils.center_crop_image(obs_image, self.config.image_size)
        with utils.eval_mode(self.agent):
            action = self.agent.select_action(obs_image)
        return action
