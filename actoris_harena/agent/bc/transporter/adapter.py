# It has only offline training

# Dataset:
# (1) get data from a dataset, or
# (2) generate data by an policy


# Agent
# It task RGB-D image
# It outputs a action, 2 poses, cartesian position and quterion for orientation.
# TODO: for our use case, we just get the pick and place action in [-1, 1] 
#        pixel space, we do not have to transform. 


# Steps:
# 1. Make transporter work within our framework on a raven environment.
#    (a) investigate what the return shape and content fo get_image function
#    (b) why input shape is 320, 160, 6: RGBD and two heatmaps??
# 2. raname act function to sample_action, or change other agent's name to act.
# 3. Make transperter work on SoftGym.
# 4. Make it work both with collecting data with a policy and with a dataset.


### Checks
# TODO: need to check preocessing of images
# TODO: need to check the difference of pixel2world and not.

import os
from tqdm import tqdm
import time
import torch
import logging 


from .utils.initializers import set_seed
from .transporter import *
from .dataset import Dataset
from actoris_harena import TrainableAgent

class TransporterAdapter(TrainableAgent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'Transporter'
        self.internal_states = {}
        self.config = config
        
        self.success_noop = config['success_noop'] if 'success_noop' in config else False
        self.bc_update_steps = config['bc_update_steps'] if 'bc_update_steps' in config else 0
        self.bc_add_all_trials = config['bc_add_all_trials'] if 'bc_add_all_trials' in config else False
        self.add_all_trials = config['add_all_trials'] if 'add_all_trials' in config else False
        self.collect_interval = config['collect_interval'] if 'collect_interval' in config else 0
        self.bc_demo_episodes_per_iteration = config['bc_demo_episodes_per_iteration'] \
            if 'bc_demo_episodes_per_iteration' in config else 0
        #self.depth_only = config['depth_only'] if 'depth_ony' in configs else False
        
        transform_config = self.config.transform
        self.two_picker = config['two_picker'] if 'two_picker' in config else False
        from actoris_harena.api import build_transform
        transformer = build_transform(transform_config.name, transform_config.params)
        
        # self.config['preprocess'] = \
        #     DATA_TRANSFORMER[transform_config.name](transform_config.params)
        
        if 'goal_condition' in config.keys() and self.config.goal_condition:

            if config.goal_mode == 'naive-stack':
                self.agent = GoalNaiveTransporterAgent(
                    #root_dir=self.config.save_dir,
                    transformer=transformer,
                    **self.config)
            elif config.goal_mode == 'goal-split':
                self.agent = GoalTransporterAgent(
                    #root_dir=self.config.save_dir,
                    transformer=transformer,
                    **self.config)
    
        else:
            if self.two_picker:
                self.agent = TwoPickerTransporterAgent(
                    #root_dir=self.config.save_dir,
                    
                    **self.config)
            
            else:
                #print('HERERE!!!')
                self.agent = OriginalTransporterAgent(
                    #root_dir=self.config.save_dir,
                    transformer=transformer,
                    **self.config)
        
        self.agent.load()
        
        if self.config.train_mode == 'from_dataset':
            self.datasets = {}
            from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
            
            # Iterate through the dictionary keys ('train', 'eval') and values
            for key, dataset_dict in self.config.datasets.items():
                
                # Access the nested 'dataset_config' block
                # Use dictionary access since YAML parsers return dicts
                ds_config = dataset_dict['dataset_config']
                
                # # If your config manager (like OmegaConf/Addict) requires conversion to a standard dict:
                # if hasattr(ds_config, 'toDict'):
                #     ds_config = ds_config.toDict()
                # elif hasattr(ds_config, 'to_dict'):
                #     ds_config = ds_config.to_dict()
                # elif hasattr(ds_config, '__dict__') and not isinstance(ds_config, dict):
                #      ds_config = vars(ds_config)
                
                # Unpack the parameters into the dataset class
                #ds_config.pop('_metadata', None)
                dataset = TrajectoryDataset(**ds_config)
                self.datasets[key] = dataset


        # if self.config.train_mode == 'from_dataset':
        #     self.datasets = {}

            
        #     for dataset_dict in self.config.datasets:
                
        #         # print()
        #         # print('Initialising dataset {} from name {}'.format(key, dataset_dict.name))

        #         # dataset_params = yaml.safe_load(
        #         #     Path('{}/configuration/datasets/{}.yaml'.\
        #         #         format(os.environ['actoris_harena_PATH'], dataset_dict.name)).read_text())
        #         # dataset_params.update(dataset_dict['params'])
                
                
                
        #         # dataset = name_to_dataset[dataset_dict.name](
        #         #     **dataset_params)

                

    def get_name(self):
        return self.name
    
    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir
        self.agent.set_save_dir(logdir)

    def train(self, update_steps, arena):
        torch.backends.cudnn.benchmark = True
        set_seed(0)


        if self.config.train_mode == 'from_dataset':
            self.train_from_dataset(self.datasets, update_steps)
        elif self.config.train_mode == 'from_policy':
            self.train_from_policy(arena, update_steps)
        else:
            raise NotImplementedError
    
    def load_checkpoint(self, checkpoint):
        self.agent.load(n_iter=checkpoint)
    
    def load(self):
        return self.agent.load()
    
    
    
    def get_writer(self):
        return self.logger

    def success(self):
        return {arena_id: False for arena_id in self.internal_states.keys()}

    def terminate(self):
        return {arena_id: False for arena_id in self.internal_states.keys()}

    def get_phase(self):
        return {arena_id: 'N/A' for arena_id in self.internal_states.keys()}


    def _init_dataset_from_policy(self, policy, arena, mode='train'):

        n_sample = self.config.n_sample \
            if (('n_sample' in self.config) and (mode=='train')) else -1

        dataset = Dataset(os.path.join(self.save_dir,
            '{}_dataset'.format(mode)), 
            self.config.swap_action, 
            tuple(self.config.in_shape[:2]),
            save_mask=self.config.save_mask if 'save_mask' in self.config else False,
            save_contour=self.config.save_contour if 'save_contour' in self.config else False,
            n_sample=n_sample)

        if mode == 'train':
            arena.set_train()
            print('num episodes for train arena', arena.get_num_episodes())
        else:
            arena.set_eval()

        max_episode = self.config.num_train_demo_episodes \
            if mode == 'train' else self.config.num_test_demo_episodes

        qbar = tqdm(total=max_episode, desc='Collecting {} data from policy ...'.format(mode))
        ## initialise qbar with dataset.n_episodes
        logging.debug('dataset.n_episodes {}'.format(dataset.n_episodes))
        #print('n_episodes', dataset.n_episodes)
        qbar.update(dataset.n_episodes)
        qbar.refresh()
        
        num_episodes = arena.get_num_episodes()
        #print('num episodes', num_episodes)
        episode_id = dataset.n_episodes % num_episodes
        while dataset.n_episodes < max_episode:
            episodes = []
            policy.reset([0])
            #print('episode_id', episode_id)
            info = arena.reset({'eid': episode_id})
            policy.init([info])
            info['reward'] = 0
            done = info['done']
            while not done:
                action = policy.act([info])[0]

            
                if action is None:
                    break
                episodes.append(
                    (info['observation'], 
                     action, 
                     info['reward'], 
                     None))
                
                info = arena.step(action)
                policy.update([info], [action])
                info['reward'] = 0
                done = info['done']
                if info['evaluation']['success']:
                    break
                if policy.terminate()[0]:
                    break
                
            #print('info', info['success'], info['largest_particle_distance'])
            episodes.append(
                    (info['observation'], 
                     None, 
                     info['reward'], 
                     None))

            # print('info success', info['success'])
            # print('policy terminate', policy.terminate())
            # print('env done', info['done'])

           
            if info['evaluation']['success'] or self.add_all_trials:
                dataset.add(episode_id, episodes)
                qbar.update(1)
            
            episode_id += 1
            episode_id %= arena.get_num_episodes()
        
        return dataset
    
    def _extend_dataset_from_policy(self, dataset, policy, arena, iteration):

        arena.set_train()
        
        max_episode = self.bc_demo_episodes_per_iteration * (iteration+1) \
            + self.config.num_train_demo_episodes

        qbar = tqdm(total=max_episode, desc='Collecting BC data from demo policy and the agent policy ...')
        qbar.update(dataset.n_episodes)
        qbar.refresh()
        
        episode_id = 0
        arena_id = arena.id
        while dataset.n_episodes < max_episode:
            episodes = []
            policy.reset([arena_id])
            self.reset([arena_id])
            info = arena.reset()
            policy.init([info])
            self.init([info])
            info['reward'] = info['reward'] if 'reward' in info else 0

            while not info['done'] and not info['evaluation']['success']:
                demo_action = policy.act([info])[0]
                agent_aciton = self.act([info])[0]
                #print('agent aciont shape', agent_aciton.shape)

                episodes.append(
                    (info['observation'], 
                     demo_action, 
                     info['reward'], 
                     None))

                info = arena.step([agent_aciton])[0]
                info['reward'] = info['reward'] if 'reward' in info else 0

                policy.update([info], [agent_aciton])
                self.update([info], [agent_aciton])
                
               
                
            ## Add the goal in the end.
            episodes.append(
                    (info['goal'], 
                     None, 
                     info['reward'], 
                     None))
            
            if self.bc_add_all_trials or (not info['evaluation']['success']):
                dataset.add(episode_id, episodes)
                qbar.update(1)
            
            episode_id += 1
        
        return dataset


    def train_from_dataset(self, datasets, update_steps=-1):
        train_dataset = datasets['train']
        test_dataset = datasets['test'] # Or 'eval', depending on your YAML!
        
        load_start_step = self.agent.load()
        
        # Supervised Learning boundary logic
        sl_update_steps = self.config.sl_update_steps
        if update_steps != -1 and load_start_step < self.config.sl_update_steps:
            sl_update_steps = min(self.config.sl_update_steps, load_start_step + update_steps + 1)
        
        # Initial validation before training starts
        self.agent.validate(test_dataset, self.logger)
        
        # Train loop restricted to SL update steps
        for u in tqdm(range(load_start_step, sl_update_steps + 1), desc='Training agent (SL only)...'):
            self.agent.train(train_dataset, self.logger)
            
            if u % self.config.test_interval == 0:
                self.agent.validate(test_dataset, self.logger)
                self.agent.save()
        
        # Final validation and save after the SL phase is complete
        self.agent.validate(test_dataset, self.logger)
        self.agent.save()

    def train_from_policy(self, arena, update_steps=-1):
        
        # collect data
        import actoris_harena.api as ag_ar
        policy = ag_ar.build_agent(
            self.config.demo_policy.name,
            self.config.demo_policy.param)

        train_dataset = self._init_dataset_from_policy(policy, arena, mode='train')
        test_dataset = self._init_dataset_from_policy(policy, arena, mode='test')

        
        load_start_step = self.agent.load()
        #self.test_agent(arena, start_step, self.logger)
        sl_update_steps = self.config.sl_update_steps
        if load_start_step < self.config.sl_update_steps:
            sl_update_steps = min(self.config.sl_update_steps, load_start_step + update_steps + 1)
        
        #print('Herr')
        self.agent.validate(test_dataset, self.logger)
        
        for u in tqdm(range(load_start_step, sl_update_steps + 1), desc='Training agent ...'):
            self.agent.train(train_dataset, self.logger)
            if u % self.config.test_interval == 0:
                self.agent.validate(test_dataset, self.logger)
                #self.test_agent(arena, u, self.logger)
                self.agent.save()
        

        if self.bc_update_steps == 0:
            self.agent.validate(test_dataset, self.logger)
            self.agent.save()
            return

        ## data collection and bc traning.
        bc_start_step = self.agent.load()

        total_iteration = self.bc_update_steps // self.collect_interval
        iteration = (bc_start_step - self.config.sl_update_steps) \
            // self.collect_interval


        self._extend_dataset_from_policy(
            train_dataset,
            policy,
            arena,
            iteration=iteration
        )
            
            # self.bc_demo_episodes_per_iteration * (iteration+1) \
            # + self.config.num_train_demo_episodes
        # step {}, num episodes {}'.format(bc_start_step, train_dataset.n_episodes))
        if update_steps == -1:
            bc_update_steps = self.bc_update_steps + self.config.sl_update_steps
        else:
            bc_update_steps = min(self.bc_update_steps + self.config.sl_update_steps, load_start_step + update_steps + 1)

        for u in tqdm(
            range(bc_start_step, bc_update_steps), 
            desc='Training agent with BC ...'):
            
            if u % self.collect_interval == 0:
                #self.test_agent(arena, u, self.logger)
                self._extend_dataset_from_policy(
                    train_dataset,
                    policy,
                    arena,
                    iteration=iteration
                )
                iteration += 1
                
                
                # self.agent.save()

            self.agent.train(train_dataset, self.logger)

        self.agent.validate(test_dataset, self.logger)
        self.agent.save()

    def single_act(self, info, update=False):
        start_time = time.time()
        if self.success_noop:
            if info['evaluation']['success']:
                return info['no_op']

        ret_state = {
            'current': {},
            'goal': None,
            # 'eid': env.get_episode_id(),
            # 'step': env.get_step()
        }
        if 'color' not in info['observation']:
            ret_state['current']['color'] = info['observation']['rgb']
        else:
            ret_state['current']['color'] = info['observation']['color'] #be careful here if change

        if 'depth' in info['observation']:
            ret_state['current']['depth'] = info['observation']['depth'] # be careful here if change
        #print('state obsrvation keys', state['observation'].keys())
        if 'mask' in info['observation']:
            ret_state['current']['mask'] = info['observation']['mask'] # be careful here if change
            
            #cv2.imwrite('tmp/mask.png', state['observation']['mask']*255)
        

        ## rehape the color and depth with self.config.inshape[:2]
        shape = tuple(self.config.in_shape[:2])

        # TODO: need to make the following works
        if (not isinstance(ret_state['current']['color'], tuple)) \
            and (not isinstance(ret_state['current']['color'], list)):


            ret_state['current']['color'] = \
                cv2.resize(ret_state['current']['color'], shape)

            if 'depth' in ret_state['current']:
                ret_state['current']['depth'] = \
                    cv2.resize(ret_state['current']['depth'], shape).reshape(shape[0], shape[1], 1)

            if 'mask' in ret_state['current']:
                #print('mask shape', ret_state['current']['mask'].shape)
                ret_state['current']['mask'] = \
                    cv2.resize(ret_state['current']['mask'].astype(np.float64), shape).reshape(shape[0], shape[1], 1)
                ret_state['current']['mask'] = ret_state['current']['mask'] > 0.9

                # from matplotlib import pyplot as plt
                # plt.imshow(ret_state['current']['mask'])
                # plt.show()
        
        
        
        if self.config.get('goal_condition', False):
            goal = info['goal']

            if 'color' not in goal:
                goal['color'] = goal['rgb']

            ret_state['goal'] = goal

            # resize goal
            if (not isinstance(ret_state['goal']['color'], tuple)) \
                and (not isinstance(ret_state['goal']['color'], list)):
                ret_state['goal']['color'] = \
                    cv2.resize(ret_state['goal']['color'], shape)
                ret_state['goal']['depth'] = \
                    cv2.resize(ret_state['goal']['depth'], shape).reshape(shape[0], shape[1], 1)
        action = self.agent.act(ret_state)
        if self.config.get('action_mode', 'norm-pixel-pick-and-place'):
            action = np.asarray(action).reshape(4)
        
            #print('action', action)
            # res_action = {
            #     'pick_0': action[0, :2][::-1],
            #     'place_0': action[0, 2:4][::-1],
            # }
            # res_action['norm-pixel-pick-and-place'] = res_action.copy()#
            res_action = action
        elif self.config.action_mode == 'original':
            res_action = action
        else:
            raise NotImplementedError
        duration = time.time() - start_time
        print(f"[TransporterNet] {info.get('arena_id', 'Unknown')}: Action planned in {duration:.4f} seconds.")
        return res_action

    def act(self, infos, updates):
        res_actions = []

        for info, up in zip(infos, updates):
            
            
            res_action = self.single_act(info, up)

            self.internal_state = self.agent.state

            res_actions.append(res_action)
        
        return res_actions
    
    def reset(self, arena_ids):

        for arena_id in arena_ids:
            self.internal_states[arena_id] = {}

    def get_state(self):
        return self.internal_states

    def init(self, info_list):
        self.agent.init()

    def update(self, info_list, actions):
        pass

    def save(self):
        self.agent.save()

    def set_eval(self):
        self.agent.set_eval()
    
    def set_train(self):
        self.agent.set_train()