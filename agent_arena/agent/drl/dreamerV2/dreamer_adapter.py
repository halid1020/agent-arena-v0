
import functools
import os
import pathlib
import sys
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml
from dotmap import DotMap
import torch
from torch import nn
from torch import distributions as torchd
from torch.utils.data import DataLoader
from tqdm import tqdm
to_np = lambda x: x.detach().cpu().numpy()

# from registration.data_transformer import *
import agent.algorithm.dreamerV2.exploration as expl
import agent.algorithm.dreamerV2.models
from agent.algorithm.dreamerV2 import tools
import agent.algorithm.dreamerV2.wrappers as wrappers
from agent.algorithm.dreamerV2.dreamer import *
from registration.dataset import *


class DreamerAdapter():

  def __init__(self, config):


    default_configs = yaml.safe_load(
        (pathlib.Path('{}/agent/algorithm/dreamerV2/configs.yaml'\
                      .format(os.environ['AGENT_ARENA_PATH'])).read_text()))
    
    vv = config.toDict()
    updated_vv = default_configs['defaults']
    updated_vv.update(**vv)
    self.config = config =  DotMap(updated_vv)

    #print("config: ", config)

    
    self.logdir = logdir = pathlib.Path(config.save_dir).expanduser()
    config.traindir = config.traindir or logdir / 'train_eps'
    config.evaldir = config.evaldir or logdir / 'eval_eps'
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)

    print('Logdir', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    self.logger = tools.Logger(logdir, config.action_repeat * step)
  

  def get_name(self):
    return "Dreamer"

    

  def train(self, env):
      if self.config.train_mode == 'online':
          self.train_online(env)
      elif self.config.train_mode == 'offline':
        datasets = {}
        if 'datasets' in self.config:
          for dataset_dict in self.config['datasets']:
              key = dataset_dict['key']
              print()
              print('Initialising dataset {} from name {}'.format(key, dataset_dict['name']))

              dataset_params = yaml.safe_load(
                  Path('{}/yamls/datasets/{}.yaml'.\
                      format(os.environ['AGENT_ARENA_PATH'], 
                              dataset_dict['name'])).read_text())
              dataset_params.update(dataset_dict['params'])
              
              
              dataset = name_to_dataset[dataset_dict['name']](
                  **dataset_params)

              datasets[key] = dataset

          self.train_offline(env, datasets)
      else:
          raise NotImplementedError
      
  def train_offline(self, env, datasets):

    transform_config = self.config.transform
    self.transform = TRANSORMER[transform_config.name](transform_config.params)

    if self.config.offline_traindir:
      directory = self.config.offline_traindir.format(**vars(self.config))
    else:
      directory = self.config.traindir
    
    if self.config.offline_evaldir:
      directory = self.config.offline_evaldir.format(**vars(self.config))
    else:
      directory = self.config.evaldir

    eval_eps = tools.load_episodes(directory, limit=1)
    train_dataset = wrap_dataset(datasets['train'], self.config, self.transform)
    callbacks = [functools.partial(
        process_episode, self.config, self.logger, 'eval', eval_eps, eval_eps)]
    eval_env = [wrappers.SelectAction(wrappers.CollectDataset(wrappers.RewardObs(env), callbacks), 'action')]
    eval_dataset = make_dataset(eval_eps, self.config)

    acts = eval_env[0].get_action_space()
    self.config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

    self.agent = agent = Dreamer(self.config, self.logger, train_dataset).to(self.config.device)
    agent.requires_grad_(requires_grad=False)
    if (self.logdir / 'latest_model.pt').exists():
      agent.load_state_dict(torch.load(self.logdir / 'latest_model.pt'))
      agent._should_pretrain._once = False


    state = None
    ### start a tqdm progress bar
    # pbar = tqdm(total=self.config.steps)

    self.logger.write()
    for step in tqdm(range(self.config.steps)):

    # step = 0
    # while step < self.config.steps:
    #   ### update the pbar with the current agent._step
    #   pbar.update(agent._step - pbar.n)
      if step % self.config.eval_every == 0:
         
        
        print('Start evaluation.')
        # video_pred = agent._wm.video_pred(next(eval_dataset))
        # self.logger.video('eval_openl', to_np(video_pred))
        eval_policy = functools.partial(agent, training=False)
        eval_env[0].set_eval()
        tools.simulate(eval_policy, eval_env, episodes=self.config.eval_episodes)
        torch.save(agent.state_dict(), self.logdir / 'latest_model.pt')

      self.agent._train(next(train_dataset))

    


  def train_online(self,  env):
    if self.config.offline_traindir:
      directory = self.config.offline_traindir.format(**vars(self.config))
    else:
      directory = self.config.traindir
    
    # train_envs = [wrappers.CollectDataset(env) for _ in range(self.config.envs)]
    # eval_envs = [wrappers.CollectDataset(env) for _ in range(self.config.envs)]
    acts = env.get_action_space()
    self.config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

    self.agent = agent = Dreamer(self.config, self.logger).to(self.config.device)
    agent.requires_grad_(requires_grad=False)
    if (self.logdir / 'latest_model.pt').exists():
      print('loading agent')
      agent.load_state_dict(torch.load(self.logdir / 'latest_model.pt'))
      agent._should_pretrain._once = False
      print('agent step', agent._step, 'train steps', self.config.steps)


    ## Load data
    if agent._step >=  self.config.steps:
      return
    
    train_eps = tools.load_episodes(directory, limit=self.config.dataset_size)
    if self.config.offline_evaldir:
      directory = self.config.offline_evaldir.format(**vars(self.config))
    else:
      directory = self.config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    # make = lambda mode: make_env(self.config, self.logger, mode, train_eps, eval_eps)
    callbacks = [functools.partial(
        process_episode, self.config, self.logger, 'train', train_eps, eval_eps)]
    train_env = [wrappers.SelectAction(wrappers.CollectDataset(wrappers.RewardObs(env), callbacks), 'action')]
    callbacks = [functools.partial(
        process_episode, self.config, self.logger, 'eval', eval_eps, eval_eps)]
    eval_env = [wrappers.SelectAction(wrappers.CollectDataset(wrappers.RewardObs(env), callbacks), 'action')]

    train_dataset = make_dataset(train_eps, self.config)
    eval_dataset = make_dataset(eval_eps, self.config)
    self.agent.set_dataset(train_dataset)


    if not self.config.offline_traindir:
      prefill = max(0, self.config.prefill - count_steps(self.config.traindir))
      print(f'Prefill dataset ({prefill} steps).')
      if hasattr(acts, 'discrete'):
        random_actor = tools.OneHotDist(torch.zeros_like(torch.Tensor(acts.low))[None])
      else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(torch.Tensor(acts.low)[None],
                                  torch.Tensor(acts.high)[None]), 1)
      def random_agent(o, d, s, r):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {'action': action, 'logprob': logprob}, None
      
      train_env[0].set_train()
      tools.simulate(random_agent, train_env, prefill)
      
      eval_env[0].set_eval()
      tools.simulate(random_agent, eval_env, episodes=self.config.eval_episodes)
      
      self.logger.step = self.config.action_repeat * count_steps(self.config.traindir)

    

    state = None

    while agent._step < self.config.steps:
      self.logger.write()
      print('Start evaluation.')
      video_pred = agent._wm.video_pred(next(eval_dataset))
      self.logger.video('eval_openl', to_np(video_pred))
      eval_policy = functools.partial(agent, training=False)
      eval_env[0].set_eval()
      tools.simulate(eval_policy, eval_env, episodes=self.config.eval_episodes)
      print('Start training.')
      train_env[0].set_train()
      state = tools.simulate(agent, train_env, self.config.eval_every, state=state)
      torch.save(agent.state_dict(), self.logdir / 'latest_model.pt')

  def act(self, state, env):
      obs = state['observation']
      new_obs = {}
      new_obs['rgb'] = np.expand_dims(obs['rgb'], axis=0)
      reward = np.asarray([state['reward']]).astype(np.float32)
      new_obs['reward'] = reward

      action, self.agent_state = self.agent._policy(new_obs, self.agent_state, False)

      return to_np(action['action'])
    
  def visualise(self, datasets):
      pass

  def reset(self):
      pass

  def init(self, state):
      self.agent_state = None

  def update(self, state, action):
      pass


def wrap_dataset(dataset, config, transform):
  
  def wrap_generate(dataloader):
    while True:
       res = {}
       batch = next(iter(dataloader))
       batch = transform(batch, train=True)
       res['image'] = batch['rgb'][:, 1:].permute(0, 1,  3, 4, 2)
       res['reward'] = batch['reward']
       res['action'] = batch['action']

       yield res   
    
  dataloader = DataLoader(
      dataset,
      batch_size=config.batch_size,
      prefetch_factor=2,
      shuffle=True)
  
  return wrap_generate(dataloader)


def make_dataset(episodes, config):
  generator = tools.sample_episodes(
      episodes, config.batch_length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['rgb']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


