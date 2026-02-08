import copy
import time
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from dotmap import DotMap
from tqdm import tqdm
import torch
import ruamel.yaml as yaml
from pathlib import Path
from dotmap import DotMap


from agent.algorithm.drq_sac.logger import Logger
from agent.algorithm.drq_sac import utils
from agent.algorithm.drq_sac.Drq import DRQAgent
from agent.algorithm.drq_sac.replay_buffer import ReplayBuffer
# from logger.visualisation_utils import *



class DrqSAC_Adapter():

    def __init__(self, config):

        ### Update Config with default setting
        ## config Dotmap to dict
        os.makedirs(config.save_dir, exist_ok=True)

        vv = config.toDict()

        cwd = os.getcwd() # get current working directory
        src_index = cwd.find("src") # find the index of "src" in the path
        if src_index != -1: # if "src" is found in the path
            truncated_path = cwd[:src_index+3] # truncate the path at "src"
        else:
            raise Exception("Could not find src directory. Make sure you are running the script from the root of the repository.")
    

        updated_vv =  yaml.safe_load(Path('{}/agents/drq_sac/config.yml'.format(truncated_path)).read_text()) 
        
        updated_vv.update(**vv)
        self.config = config =  DotMap(updated_vv)

        self.work_dir = self.config.save_dir

        self.logger = Logger(self.work_dir,
                             save_tb=config.log_save_tb,
                             log_frequency=config.log_frequency_step,
                             agent='drq',
                             action_repeat=1,
                             chester_log=None)
        utils.set_seed_everywhere(self.config.seed)
        self.device = torch.device(self.config.device)


        obs_shape = self.config.obs_shape
        new_obs_shape = np.zeros_like(obs_shape)
        new_obs_shape[0] = obs_shape[0]
        new_obs_shape[1] = obs_shape[1]
        new_obs_shape[2] = obs_shape[2]
        config.agent['obs_shape'] = config.encoder['obs_shape'] = new_obs_shape
        config.agent['action_shape'] = self.config.action_shape
        config.actor['action_shape'] = self.config.action_shape
        config.critic['action_shape'] = self.config.action_shape
        config.actor['encoder_cfg'] = config.encoder
        config.critic['encoder_cfg'] = config.encoder
        config.agent['action_range'] = [
            torch.tensor(self.config.action_lower_bound).float().to(self.device),
            torch.tensor(self.config.action_upper_bound).float().to(self.device)
        ]
        config.agent['encoder_cfg'] = config.encoder
        config.agent['critic_cfg'] = config.critic
        config.agent['actor_cfg'] = config.actor
        config.agent['reward_scale'] = config.reward_scale

        self.agent = DRQAgent(**config.agent)

        self.replay_buffer = ReplayBuffer(new_obs_shape,
                                          self.config.action_shape,
                                          config.replay_buffer_capacity,
                                          self.config.image_pad, self.device)

        # self.video_recorder = VideoRecorder(
        #     self.work_dir if config.save_video else None)
        self.step = 0
        self.video_dir = os.path.join(self.work_dir, 'video')
        self.model_dir = os.path.join(self.work_dir, 'model')
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir, exist_ok=True)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        

    def train_online(self, env, loss_logger, eval_logger):
        
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        #ep_info = []
        start_time = time.time()
        while self.step < self.config.num_train_steps:
            #print("step: ", self.step)


            # evaluate agent periodically
            if self.step % self.config.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate(env)
                if self.config.save_model and self.step % (self.config.eval_frequency *5):
                    self.agent.save(self.model_dir, self.step)

            if done:
                if self.step > 0:
                    if self.step % self.config.log_interval == 0:
                        self.logger.log('train/duration',
                                        time.time() - start_time, self.step)
                        # for key, val in get_info_stats([ep_info]).items():
                        #     self.logger.log('train/info_' + key, val, self.step)
                        self.logger.dump(
                            self.step, save=(self.step > self.config.num_seed_steps))

                    start_time = time.time()

                if self.step % self.config.log_interval == 0:
                    self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                
                env.set_train()
                external_state = env.reset()
                obs = external_state['observation']['image'].transpose(2,0,1)
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                #ep_info = []

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.config.num_seed_steps:
                action = env.sample_random_action()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.config.num_seed_steps:
                for _ in range(self.config.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            external_state = env.step(action)
            next_obs = external_state['observation']['image'].transpose(2,0,1)
            reward = external_state['reward']
            done = external_state['done']

            # next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            # done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            done_no_max = 0
            episode_reward += reward
            # ep_info.append(info)

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

    
    def train_offline(self, env, datasets, loss_logger, eval_logger):
        
      
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
                    True if j == len(actions)-1 else False,
                    0) # inifit bootstrap
        last_episode_idx = self.config.add_initial_episodes - 1


        print('Start offline training...')
        #episode, episode_reward, done, ep_info = 0, 0, True, []
        for step in tqdm(range(0, self.config.offline_update_steps)):
                    
            if step % self.config.eval_frequency == 0:
                #L.log('eval/episode', episode, step)
                self.evaluate(env)
                if self.config.save_model and (step % (self.config.eval_frequency * 5) == 0):
                    self.agent.save(self.model_dir, step)

            self.agent.update(self.replay_buffer, self.logger, step)

            if step % self.config.add_episode_interval == 0:
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
                            True if j == len(actions)-1 else False,
                            0)




    def train(self, datasets, env, loss_logger, eval_logger):
        if self.config.train_mode == 'online':
            self.train_online(env, loss_logger, eval_logger)
        elif self.config.train_mode == 'offline':
            self.train_offline(env, datasets, loss_logger, eval_logger)
        else:
            raise NotImplementedError

    def visualise(self, datasets):
        pass

    def evaluate(self, env):
        print('Evaluating drq sac...')
        average_episode_reward = 0
        infos = []
        all_frames = []
        plt.figure()
        env.set_eval()

        for episode in range(self.config.num_eval_episodes):
            print('Episode {}/{}: '.format(episode+1, self.config.num_eval_episodes))
            external_state = env.reset(episode_id=episode)
            obs = external_state['observation']['image'].transpose(2,0,1)

            # obs = self.env.reset()
            # print(type(obs))
            # print(obs.shape)
            # print(obs)
            # exit()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            ep_info = []
            frames = [env.render(mode='rgb', resolution=(128, 128))]
            rewards = []

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                external_state = env.step(action)
                obs = external_state['observation']['image'].transpose(2,0,1)
                reward = external_state['reward']
                done = external_state['done']

                # obs, reward, done, info = self.env.step(action)
                # self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1
                # ep_info.append(info)
                frames.append(env.render(mode='rgb', resolution=(128, 128))) 
                rewards.append(reward)

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
            # infos.append(ep_info)
            plt.plot(range(len(rewards)), rewards)
            if len(all_frames) < 8:
                all_frames.append(frames)

        average_episode_reward /= self.config.num_eval_episodes
        # for key, val in get_info_stats(infos).items():
        #     self.logger.log('eval/info_' + key, val, self.step)

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        all_frames = np.array(all_frames).swapaxes(0, 1)
        all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
        save_numpy_as_gif(all_frames, os.path.join(self.video_dir, '%d.gif' % self.step))
        plt.savefig(os.path.join(self.video_dir, '%d.png' % self.step))

    def load_model(self):
        ## Find the latest model in the model directory
        model_files = os.listdir(self.model_dir)
        steps = [int(f.split('.')[0].split('_')[-1]) for f in model_files]
        latest_step = max(steps)
        self.agent.load(
            self.model_dir, 
            latest_step)


    def reset(self):
      pass

    def init(self, state):
        pass

    def update(self, state, action):
        pass

    def visualise(self, datasets):
        pass

    def act(self, state):
        obs = state['observation']['image'].transpose(2,0,1)
        with utils.eval_mode(self.agent):
            action = self.agent.act(obs, sample=False)
        return action
    