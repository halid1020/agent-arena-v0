import os
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader
from collections import namedtuple
import pytorch_lightning.utilities.seed as seed_utils
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from .flownet.models import FlowNet
from .flownet.dataset import FlowDataset
from .picknet.models import FlowPickSplitModel
from .picknet.dataset import PickNetDataset


from agent_arena import TrainableAgent
from agent_arena.utilities.utils import TrainWriter
import cv2

Experience = namedtuple('Experience', ('obs', 'goal', 'act', 'rew', 'nobs', 'done'))


class FabricFlowNetAdapter(TrainableAgent):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'fabricflownet'
        self.flownet = FlowNet(** self.config.flownet).to(self.config.device)
        
        if config.picknet.net_cfg.model_type == 'split':
            self.picknet = FlowPickSplitModel(
                **self.config.picknet.net_cfg
                ).to(self.config.device)
        else:
            raise NotImplementedError
        
        self.state = {}
        self.writer = TrainWriter(self.config.save_dir)
        self.action_mode = config['action_mode'] if 'action_mode' in config else "step-wise"
        self.action_horizon = config['action_horizon'] if 'action_horizon' in config else 15
        self.action_repeat = config['action_repeat'] if 'action_repeat' in config else 1
        
    def train(self, update_steps, arena = None):
        

        
        # Train FlowNet
        print('Training FlowNet')
        seed_utils.seed_everything(self.config.flownet.seed)
        flownet_train_loader, flownet_val_loader = self._get_flownet_dataloaders(self.config.flownet)
        csv_logger = pl_loggers.CSVLogger(save_dir=os.path.join(self.config.save_dir, 'flownet_csv'))
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(self.config.save_dir, 'flownet_tb'))
        chkpt_cb = ModelCheckpoint(monitor='loss/val', save_last=True, save_top_k=-1, every_n_val_epochs=10)
        flownet_trainer = pl.Trainer(
            gpus=[0],
            logger=[csv_logger, tb_logger],
            max_epochs=self.config.flownet.epoch,
            check_val_every_n_epoch=self.config.flownet.check_val_every_n_epoch,
            log_every_n_steps=len(flownet_train_loader) if len(flownet_train_loader) < 50 else 50,
            callbacks=[chkpt_cb])
        flownet_trainer.fit(self.flownet, flownet_train_loader, flownet_val_loader)

        # Train PickNet
        print('Training PickNet')
        seed_utils.seed_everything(self.config.picknet.seed)

        picknet_train_loader, picknet_val_loader = self._get_picknet_dataloaders(self.config.picknet)

        csv_logger = pl_loggers.CSVLogger(save_dir=os.path.join(self.config.save_dir, 'picknet_csv'))
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(self.config.save_dir, 'picknet_tb'))

        chkpt_cb = ModelCheckpoint(monitor='loss1/val', save_last=True, save_top_k=-1, every_n_val_epochs=10)
        trainer = pl.Trainer(gpus=[0],
                            logger=[csv_logger, tb_logger],
                            max_epochs=self.config.picknet.epochs,
                            check_val_every_n_epoch=self.config.picknet.check_val_every_n_epoch, # TODO change to every k steps
                            log_every_n_steps=len(picknet_train_loader) if len(picknet_train_loader) < 50 else 50,
                            callbacks=[chkpt_cb])
        trainer.fit(self.picknet, picknet_train_loader, picknet_val_loader)


    def act(self, information):
        
        start = time.time()
        depth = information['observation']['depth']
        if self.is_terminate:
            print('fabricflownet no-op')
            return information['no_op']
        
        mask = information['observation']['mask']
        mask = mask.reshape(*mask.shape, 1)
        
        if self.action_mode == 'step-wise':
            idx = int(self.step/self.action_repeat)
            if 'goals' in information:
                goal_depth = information['goals'][idx]['depth']
                goal_mask = information['goals'][idx]['mask']
            elif 'goal' in information:
                goal_depth = information['goal']['depth']
                goal_mask = information['goal']['mask']
                
        elif self.action_mode == 'final-goal':
            goal_depth = information['goals'][-1]['depth']
            goal_mask = information['goals'][-1]['mask']
        else:
            raise NotImplementedError
        
        
        goal_mask = goal_mask.reshape(*goal_mask.shape, 1)

        depth = depth*mask #*255
        goal_depth = goal_depth*goal_mask #*255
        depth += self.config.depth_offset
        goal_depth += self.config.depth_offset

        depth = cv2.resize(depth, (200, 200))
        curr_img = torch.tensor(depth).unsqueeze(0).to(self.config.device)
        goal_depth = cv2.resize(goal_depth, (200, 200))
        goal_depth = torch.tensor(goal_depth).unsqueeze(0).to(self.config.device)

        inp = torch.cat([curr_img, goal_depth]).unsqueeze(0)
        flow_out = self.flownet(inp)

        self.state['flow'] = flow_out.detach().cpu().numpy().squeeze().transpose(1,2,0)
        # print('flow_out shape:', self.state['flow'].shape)
        # from .utils import plot_flow
        # # creat an ax
        # ax = plt.gca()
        # plot_flow(ax, self.state['flow'])
        # plt.savefig('flow.png')
        # plt.close()

        # mask flow
        flow_out[0,0,:,:][inp[0,0,:,:] == 0] = 0
        flow_out[0,1,:,:][inp[0,0,:,:] == 0] = 0
        self.state['mask_flow'] = flow_out.detach().cpu().numpy()

       
        action, _ = self.picknet.get_action(flow_out, 
                curr_img.unsqueeze(0), goal_depth.unsqueeze(0))
        action = (action.astype(np.float32)/200)*2 -1

        action = action.reshape(-1, 2)
        # swap x and y
        action = action[:, [1, 0]]
        action = action.reshape(-1, 4)

        if self.config.only_allow_one_picker:
            action = action[0]

        action = action.reshape(*information['action_space'].shape)
        

        print('fabric flow net action', action)

        self.state['inference_time'] = time.time() - start
        
        return action

    def terminate(self):
        return self.is_terminate


    def load(self, path=None):
        load_dir = f"{self.config.save_dir}/checkpoints/picknet/weights"
        ## find the largest iteration
        load_iter = 0
        for file in os.listdir(load_dir):
            if file.startswith('first_'):
                load_iter_tmp = int(file.split('_')[1].split('.')[0])
                load_iter = max(load_iter, load_iter_tmp)
        
        self.load_checkpoint(load_iter)
        return load_iter

    # TODO: to be done.
    def save(self):
        pass
    
    def load_checkpoint(self, load_iter):
        first_path = f"{self.config.save_dir}/checkpoints/picknet/weights/first_{load_iter}.pt"
        second_path = f"{self.config.save_dir}/checkpoints/picknet/weights/second_{load_iter}.pt"

        # self.picknet = FlowPickSplitModel(
        #     s_pick_thres=self.config.single_pick_thresh,
        #     a_len_thres=self.config.action_len_thresh).cuda()
        self.picknet.first.load_state_dict(torch.load(first_path))
        self.picknet.second.load_state_dict(torch.load(second_path))

        # flow model
        # self.flownet = FlowNet(input_channels=2).cuda()
        flownet_dir = f"{self.config.save_dir}/checkpoints/flownet/horz5_epoch=1588.ckpt"
        checkpt = torch.load(flownet_dir)
        self.flownet.load_state_dict(checkpt['state_dict'])

        print('Successfully loaded picknet and flownet models {}'.format(load_iter))

    
    def set_train(self):
        self.picknet.train()
        self.flownet.train()
    
    def set_eval(self):
        self.picknet.eval()
        self.flownet.eval()

    def get_writer(self) -> TrainWriter:
        return self.writer

    def get_phase(self):
        return 'fold'

    def get_state(self):
        return self.state

    def init(self, information):
        pass

    def update(self, information, action):
        self.step += 1
        if self.action_mode == 'step-wise' and self.step >= len(information['goals']):
            self.is_terminate = True
        
        if self.step >= self.action_horizon:
            self.is_terminate = True

    def reset(self):
        self.state = {}
        self.step = 0
        self.is_terminate = False

    def get_action_type(self):
        return 'default'


    def _get_flownet_dataloaders(cfg):
        # Get training samples
        trainpath = f'{cfg.base_path}/{cfg.train_name}'
        trainfs = sorted(['_'.join(fn.split('_')[0:2])
                    for fn in os.listdir(f'{trainpath}/actions')])
        if cfg['max_train_samples'] != None:
            trainfs = trainfs[:cfg['max_train_samples']]
            print(f"Max training set: {len(trainfs)}")

        # Get validation samples
        valpath = f'{cfg.base_path}/{cfg.val_name}'
        valfs = sorted(['_'.join(fn.split('_')[0:2])
                    for fn in os.listdir(f'{valpath}/actions')])
        if cfg['max_val_samples'] != None:
            valfs = valfs[:cfg['max_val_samples']]
            print(f"Max val set: {len(valfs)}")
        
        # Get camera params
        train_camera_params = np.load(f"{trainpath}/camera_params.npy", allow_pickle=True)[()]
        val_camera_params = np.load(f"{valpath}/camera_params.npy", allow_pickle=True)[()]
        np.testing.assert_equal(val_camera_params, train_camera_params)
        camera_params = train_camera_params

        # Get datasets
        train_data = FlowDataset(cfg, trainfs, camera_params, stage='train')
        val_data = FlowDataset(cfg, valfs, camera_params, stage='val')
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, 
                                  num_workers=cfg.workers, persistent_workers=cfg.workers>0)
        val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, 
                                num_workers=cfg.workers, persistent_workers=cfg.workers>0)
        return train_loader, val_loader
    
    def _get_flownet_dataloaders(cfg):
            # Get training samples
        trainpath = f'{cfg.base_path}/{cfg.train_name}'
        trainfs = sorted(['_'.join(fn.split('_')[0:2])
                    for fn in os.listdir(f'{trainpath}/actions')])
        if cfg.max_buf != None:
            trainfs = trainfs[:cfg.max_buf]
            print(f"Max training set: {len(trainfs)}")

        # Get validation samples
        valpath = f'{cfg.base_path}/{cfg.val_name}'
        valfs = sorted(['_'.join(fn.split('_')[0:2])
                    for fn in os.listdir(f'{valpath}/actions')])
        if cfg.max_buf != None:
            valfs = valfs[:cfg.max_buf]
            print(f"Max val set: {len(valfs)}")

        # Get camera params
        train_camera_params = np.load(f"{cfg.base_path}/{cfg.train_name}/camera_params.npy", allow_pickle=True)[()]
        val_camera_params = np.load(f"{cfg.base_path}/{cfg.val_name}/camera_params.npy", allow_pickle=True)[()]
        np.testing.assert_equal(val_camera_params, train_camera_params)
        camera_params = train_camera_params

        train_data = PickNetDataset(camera_params, cfg, trainfs, mode='train', pick_pt=cfg.net_cfg.pick)
        val_data = PickNetDataset(camera_params, cfg, valfs, mode='test', pick_pt=cfg.net_cfg.pick)
        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, persistent_workers=cfg.workers>0)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=cfg.workers, persistent_workers=cfg.workers>0)

        return train_loader, val_loader
