import os

import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import cv2
import api as ag_ar

from agent_arena.agent.agent import TrainableAgent

from .dataset import Dataset
from utilities.data_transformer.register import DATA_TRANSFORMER
from .register import PHASE_PREDICTOR



class PhasePredictionAgent(TrainableAgent):
    
    def __init__(self, config):
        self.config = config
        self.phase_predictor = PHASE_PREDICTOR[config.predictor.name](config.predictor.params)
        self._load_phase_agents(config)  # Method to load different phase agents
        self.current_phase = None
        self.writer = SummaryWriter(self.config.save_dir)
        self.num_phases = len(self.config.phases)
        self.num2phase = {v: k for k, v in self.config.phases.items()}
        transform_config = self.config.transform
        self.transform = DATA_TRANSFORMER[transform_config.name](transform_config.params)
        self.balance_phase_sampling = self.config.balance_phase_sampling if 'balance_phase_sampling' in self.config else False

        ## num of samples for each phase
        self.num_samples_each_phase = {phase: 0 for phase in self.config.phases.keys()}
        self.fixed_window_size = self.config.fixed_window_size if 'fixed_window_size' in self.config else 0
        self.window_step = 0
        self.phase_discount = self.config.phase_discount if 'phase_discount' in self.config else False
        self.decay_rate = self.config.decay_rate if 'decay_rate' in self.config else 1.0
        

    def act(self, state):
        self.current_phase  = self._decide_phase(state)
        return self.phase_agents[self.current_phase].act(state)
    
    def get_state(self):
        return {}

    def get_phase(self):
        return self.current_phase

    def reset(self):
        self.current_phase = None
        for agent in self.phase_agents.values():
            agent.reset()
        self.window_step = 0
        self.phase_weights = np.ones(self.num_phases)/self.num_phases

    def success(self):
        return False
    
    def terminate(self):
        return False
    
    def train(self, update_steps, arena):

        torch.backends.cudnn.benchmark = True
        
        # Collect Data from policy
        policy = ag_ar.build_agent(
            self.config.collection_policy.name,
            arena,
            self.config.collection_policy.param)
        print('type policy', type(policy))
        
        train_dataset = self._init_dataset_from_policy(policy, arena, mode='train')
        #print('num_samples_each_phase', self.num_samples_each_phase)
        val_dataset = self._init_dataset_from_policy(policy, arena, mode='validation')


        ## load latest checkpoint
        start_update_steps = self.load() + 1


        ## Update prediction network
        # criterion = torch.nn.CrossEntropyLoss()
        # self.phase_predictor.train()
        # self.validate(val_dataset, start_update_steps)
        if update_steps == -1:
            update_steps = self.config.num_update_steps
        else:
            update_steps = update_steps + start_update_steps
        
        for update_step in tqdm(range(start_update_steps, update_steps + 1)):
            
            ## Update one step
            data = self._get_samples(train_dataset, 
                                     batch_size=self.config.train_batch_size,
                                     balance_phase_sampling=self.balance_phase_sampling) ### when loaded phase is a one-hot encoding vector.
            obs, phase = data
            # print('phase', phase[0])
            # import matplotlib.pyplot as plt
            # img = obs[0].permute(1, 2, 0).detach().cpu().numpy()
            # img = (img + 0.5)*255.0
            # img = img.astype(np.uint8)
            # print('img shape', img.shape)
            # plt.imshow(img)
            # plt.show()
            # print('obs input shape', obs['input'].shape)
            # print('obs output shape', obs['output'].shape)
            train_loss = self.phase_predictor.train(obs, phase, 
                update_step=update_step, total_steps=self.config.num_update_steps)

            # obs = obs.to(self.config.device)
            # phase = phase.to(self.config.device)

            # self.optimiser.zero_grad()
            # #print('obs shape', obs.shape)
            # pred_phase = self.phase_predictor(obs)
            # loss = criterion(pred_phase, phase)
            # loss.backward()
            # self.optimiser.step()

            for name, loss in train_loss.items():
                self.writer.add_scalar('train/{}'.format(name), loss, update_step)

            # self.writer.add_scalar('train/loss', train_loss, update_step)

            if update_step % self.config.test_interval == 0:
                # validate test dataset, and generate confusion matrix, and log results
                #self.phase_predictor.eval()
                self.validate(val_dataset, update_step)
                
                
                self.save_checkpoint(checkpoint=update_step)

    def validate(self, dataset, update_step):
        confusion_matrix = torch.zeros(self.num_phases, self.num_phases)
        # with torch.no_grad():
        #     # val_loss = 0
        #     # correct = 0
        #     # total = 0
            
        obs, phase = self._get_samples(dataset, 
                augment=True, 
                batch_size=self.config.val_batch_size, 
                balance_phase_sampling=True)
        #print('obs shape', obs.shape)
        #print('pred_phase', pred_phase)
        #print('obs keys', obs.keys())
        val_loss  = self.phase_predictor.test(obs.copy(), phase)
        recon = self.phase_predictor.reconstruct(obs['input'])
        
        # _, predicted = pred_phase.max(1)
        _, ground_truth = phase.max(1)
        ground_truth = ground_truth.detach().cpu().numpy()
        pred_probs = self.phase_predictor.predict(obs['input'])

        predicted = np.argmax(pred_probs, axis=1)

        #print('predicted', predicted)
        for t, p in zip(ground_truth, predicted):
            confusion_matrix[t, p] += 1
        
        total = phase.size(0)
        correct = (predicted == ground_truth).sum().item()
        mis_predicted_items = (predicted != ground_truth).nonzero()[0]
        print('mis_predicted_items', mis_predicted_items)
        
        acc = 100. * correct / total
        for name, loss in val_loss.items():
            self.writer.add_scalar('val/{}'.format(name), loss, update_step)
        self.writer.add_scalar('val/acc', acc, update_step)
        ## write first image and its reconstruction
        # img = self.transform.postprocess({self.config.input_obs: obs['input']})\
        #     [self.config.input_obs].detach().cpu().numpy()
        # img = img.astype(np.uint8)
        # self.writer.add_image('val/sample_input', img[0], update_step)
        # print('img shape', img.shape)
        if recon is not None:
            
            
            if self.config.output_obs in ['binary_contour', 'binary_mask']:
                recon = (recon[:, 0:1] > 0.5).astype(np.float32)
                ground_output = obs['output'][:, 0:1].detach().cpu().numpy().astype(np.float32)
            else:
                recon = (recon + 0.5)*255.0
                recon = recon.astype(np.uint8)
                ground_output = ((obs['output'] + 0.5)*255.0).detach().cpu().numpy().astype(np.float32)
            
            self.writer.add_image('val/sample_recon', recon[0], update_step)

            for i, id in enumerate(mis_predicted_items[:3]):
                # self.writer.add_image('val/mis_predicted_{}_input'.format(i), img[id], update_step)
                ## write the ground truth phase and predicted phase
                self.writer.add_text('val/mis_predicted_{}_phase'.format(i), 
                    'ground truth: {}\npredicted: {}'.format(self.num2phase[ground_truth[id]], self.num2phase[predicted[id]]), update_step)
                self.writer.add_image('val/mis_predicted_{}_recon'.format(i), recon[id], update_step)
                self.writer.add_image('val/mis_predicted_{}_ground_recon'.format(i), ground_output[id], update_step)

            print('update step {},  val acc {}'\
                    .format(update_step, acc))

        ## Print confusion matrix
        print('confusion matrix\n', confusion_matrix)

        # save the confusion matrix as numpy array
        np_confusion_matrix = confusion_matrix.numpy()
        os.makedirs(os.path.join(self.config.save_dir, 'confusion_matrix'), exist_ok=True)
        np.save(
            os.path.join(self.config.save_dir,
            'confusion_matrix/{}.npy'.format(update_step)), np_confusion_matrix)
    
    def get_writer(self):
        return self.writer
        
    def _get_samples(self, dataset, augment=True, batch_size=32, balance_phase_sampling=False):
        """Get a dataset sample.

        Args:
          dataset: a agents.transporter.Dataset (train or validation)
          augment: if True, perform data augmentation.

        Returns:
          tuple of data for training:
            (input_image, phase vector), 
            where input_image is a dictionary of rgb and depth tensors
            and phase vector is a one-hot encoding vector
        """

        batch = dataset.sample_batch(
            batch_size, 
            balance_phase_sampling=balance_phase_sampling,
            num_phases=self.num_phases)
        ## batch is a list of (obs, phase) tuple
        img, phase = zip(*batch)
        #print('phase', phase)

        ## visualise the first rgb image
        # import matplotlib.pyplot as plt
        # plt.imshow(img[3]['rgb'])
        # plt.show()

        ## image is a tuple of dictionary that has rgb and depth
        ## convert it to dictionary of rgb and depth, and convert it to tensor
        img_ = {
            'rgb': torch.tensor([x['rgb'] for x in img]).permute(0, 3, 1, 2), 
            'depth': torch.tensor([x['depth'] for x in img]).unsqueeze(-1).permute(0, 3, 1, 2),
           
        }
        if 'mask' in img[0]:
            img_['mask'] = torch.tensor([x['mask'] for x in img]).unsqueeze(-1).permute(0, 3, 1, 2)
        
        if 'contour' in img[0]:
            img_['contour'] = torch.tensor([x['contour'] for x in img]).unsqueeze(-1).permute(0, 3, 1, 2)

        phase = torch.tensor(phase)
        

        ## Phase is batch of numbers, I want to convert it to one-hot encoding
        phase = torch.nn.functional.one_hot(phase, num_classes=self.num_phases).float()
        #print('phase', phase)

        img  = self.transform(img_, augment=augment)
        

        if self.config.output_obs == 'binary_mask':
            ### transform 1 channel mask to 2 chnnel mask, 
            # where the second channel is the complement of the first channel

            img['binary_mask'] = img['mask']
            img['binary_mask'] = torch.cat([img['binary_mask'], 1 - img['binary_mask']], dim=1)
        
        elif self.config.output_obs == 'binary_contour':
            img['binary_contour'] = img['contour']
            img['binary_contour'] = torch.cat([img['binary_contour'], 1 - img['binary_contour']], dim=1)
        elif self.config.output_obs == 'gray+mask':
            img['gray+mask'] = torch.cat([img['gray'], img['mask']], dim=1)


        ret_obs = {
            'input': img[self.config.input_obs],
           
        }

        if self.config.output_obs is not None:
            ret_obs['output'] = img[self.config.output_obs]

        # rgb = img['rgb']/255.0 - 0.5
        # rgb = rgb.permute(0, 3, 1, 2)

        ## print input output shape
        print('input shape', ret_obs['input'].shape)
        if 'output' in ret_obs:
            print('output shape', ret_obs['output'].shape)

        return ret_obs, phase
    
    def init(self, information):
        ## init all policies
        for agent in self.phase_agents.values():
            agent.init(information)

    def update(self, information, action):
        pass

    def _init_dataset_from_policy(self, policy, arena, mode='train'):
        dataset = Dataset(os.path.join(self.config.save_dir, '{}_dataset'.format(mode)), 
                        tuple(self.config.in_shape[:2]), save_contour=self.config.save_contour)

        if mode == 'train':
            arena.set_train()
        else:
            arena.set_eval()

        max_episode = self.config.num_train_episodes if mode == 'train' else self.config.num_val_episodes
        #print('max_episode', max_episode)

        qbar = tqdm(total=max_episode, desc='Collecting {} data from policy ...'.format(mode))
        logging.debug('dataset.n_episodes {}'.format(dataset.n_episodes))
        qbar.update(dataset.n_episodes)
        qbar.refresh()

        episode_id = 0
        while dataset.n_episodes < max_episode:
            episodes = []
            policy.reset()
            info = arena.reset()
            policy.init(info)
            done = False
            while not done and not policy.terminate():
                action = policy.act(info)
                phase = policy.get_phase()
                if action is None:
                    break
                #print('obs keys', info['observation'].keys())
                if phase in self.config.phases.keys():
                    # from matplotlib import pyplot as plt
                    # plt.imshow(info['observation']['rgb'])
                    # plt.show()
                    # plt.imshow(info['observation']['contour'])
                    # plt.show()
                    episodes.append((info['observation'], self.config.phases[phase]))
                    if mode == 'train':
                        self.num_samples_each_phase[phase] += 1
                
                info = arena.step(action)
                policy.update(info, action)
                done = info['done']
                if info['success']:
                    break

            if info['success'] and 'success' in self.config.phases.keys():
                episodes.append((info['observation'], self.config.phases['success']))
                if mode == 'train':
                    self.num_samples_each_phase[phase] += 1
            
            dataset.add(episode_id, episodes)
            qbar.update(1)
            episode_id += 1

        return dataset
    
    def save_checkpoint(self, checkpoint=-1):
        """
            Save the checkpoint of the model to the config.save_dir/checkpoints/phase_prediction_{checkpoint}.pth
        """

        path = os.path.join(self.config.save_dir, 'checkpoints')
        os.makedirs(path, exist_ok=True)
        file_dir = os.path.join(path, 'phase_predictor_{}.pth'.format(checkpoint))
        self.phase_predictor.save(file_dir)


    def save(self, path=None):
        
        """
            If the path is given, save the model to the path.
            If the path is not given, save the model to the config.save_dir/checkpoints/phase_prediction.pth
        """

        # Implement save prediction network with its checkpoint
        if path is None:
            path = self.config.save_dir
            path = os.path.join(path, 'checkpoints')
        os.makedirs(path, exist_ok=True)

        file_dir = os.path.join(file_dir, 'phase_predictor.pth')
        self.phase_predictor.save(file_dir)

    def load(self, path=None):
        """
            load the model from the path, if the path is given.
            load the last checkpoint model from the config.save_dir/checkpoints, 
            if the path is not given, return the checkpoint number.
        """
        
        if path is None:
            path = self.config.save_dir
            path = os.path.join(path, 'checkpoints')
            os.makedirs(path, exist_ok=True)
            ## find the largest checkpoint
            checkpoint = -1
            for file in os.listdir(path):
                if 'phase_predictor_' in file:
                    current_checkpoint = int(file.split('.')[0].split('_')[-1])
                    checkpoint = max(checkpoint, current_checkpoint)
            file_dir = os.path.join(path, 'phase_predictor_{}.pth'.format(checkpoint))
        else:
            file_dir = os.path.join(path, 'phase_predictor.pth')

        if os.path.exists(file_dir):
            print('Loading checkpoint from {}'.format(file_dir))
            self.phase_predictor.load(file_dir)

            return checkpoint
        else:
            return -1
    
    def load_checkpoint(self, checkpoint):
        path = os.path.join(self.config.save_dir, 'checkpoints')
        file_dir = os.path.join(path, 'phase_predictor_{}.pth'.format(checkpoint))
        if os.path.exists(file_dir):
            print('Loading checkpoint from {}'.format(file_dir))
            self.phase_predictor.load(file_dir)
        else:
            raise FileNotFoundError('Checkpoint file not found')

    def set_train(self):
        self.phase_predictor.train_mode()


    def set_eval(self):
        self.phase_predictor.eval_mode()

    def get_name(self):
        return "PhasePredictionAgent"

    def _load_phase_agents(self, config):
        # Load different agents for each phase
        self.phase_agents = {}
        for phase_name, phase_params in config.phase_agents.items():
            if 'config' in phase_params:
                agent_config = ag_ar.retrieve_config(
                    phase_params.name, 
                    phase_params.arena, 
                    phase_params.config, 
                    phase_params.log_dir
                )
                self.phase_agents[phase_name] = ag_ar.build_agent(
                    phase_params.name, 
                    config=agent_config)
            else:
                self.phase_agents[phase_name] = ag_ar.build_agent(
                    phase_params.name,
                    phase_params.arena,
                    phase_params.param)
            ## if the agent is the instanceof TrainableAgent, load the model
            if isinstance(self.phase_agents[phase_name], TrainableAgent):
                if 'checkpoint' in phase_params:
                    self.phase_agents[phase_name].load_checkpoint(phase_params.checkpoint)
                else:
                    self.phase_agents[phase_name].load()

    def _decide_phase(self, state):
        #print('fixed_window_size', self.fixed_window_size)
        if self.window_step < self.fixed_window_size and (self.current_phase is not None):
            self.window_step += 1
            print('window step', self.window_step)
            print('decided phase', self.current_phase)
            return self.current_phase
        else:
            self.window_step = 0
        
        img = self.transform(
            {'rgb': state['observation']['rgb'].transpose(2, 0, 1).copy(),
             'depth': state['observation']['depth'].transpose(2, 0, 1).copy()}, 
            augment=True, 
            single=True)
        
        # save rgb and contour image

        cv2.imwrite('TS-rgb.png', state['observation']['rgb'])
        cv2.imwrite('TS-contour.png', state['observation']['contour']*255)

        probs = self.phase_predictor.predict(img[self.config.input_obs])[0]
        prediction_probs = [probs]

        if self.phase_discount:
            probs = probs * self.phase_weights
            prediction_weigthed_probs = [probs]


        
        if self.config.decide_mode == 'max':
            phase = np.argmax(probs)
        elif self.config.decide_mode == 'sample':
            phase = np.random.choice(np.arange(len(probs)), p=probs)
        elif self.config.decide_mode == 'threshold_sample':
            if np.max(probs) > 0.6:
                phase = np.argmax(probs)
            else:
                phase = np.random.choice(np.arange(len(probs)), p=probs)
        elif self.config.decide_mode == 'vote':
            predictions = [np.argmax(probs)]
            


            for _ in range(4):
                img = self.transform(
                    {'rgb': state['observation']['rgb'].transpose(2, 0, 1).copy(),
                     'depth': state['observation']['depth'].transpose(2, 0, 1).copy()
                     }, 
                    augment=True, 
                    single=True)
        
                probs = self.phase_predictor.predict(img[self.config.input_obs])[0]
                prediction_probs.append(probs)
                if self.phase_discount:
                    probs = probs * self.phase_weights
                    prediction_weigthed_probs.append(probs)
                predictions.append(np.argmax(probs))
            
            phase = max(set(predictions), key=predictions.count)
        else:
            raise NotImplementedError
        
        if self.phase_discount:
            if phase != self.config.phases[self.current_phase]:
                self.phase_weights = np.zeros(self.num_phases)
                self.phase_weights[phase] = 1.0
            else:
                tmp_phase_weight = self.phase_weights[phase] * self.decay_rate
                print('tmp_phase_weight', tmp_phase_weight)
                self.phase_weights = (1-tmp_phase_weight)/(self.num_phases - 1) * np.ones(self.num_phases)
                self.phase_weights[phase] = tmp_phase_weight

        print('prediction probs', prediction_probs)
        if self.phase_discount:
            print('prediction weighted probs', prediction_weigthed_probs)
        print('decided phase', phase)
       
        return self.num2phase[phase]