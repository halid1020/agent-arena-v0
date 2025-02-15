import os.path as osp
import json
from dotmap import DotMap

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
import pyrealsense2 as rs

from agent_arena import TrainableAgent
import cv2

from agent.vcd.utils.utils import (
    pc_reward_model, voxelize_pointcloud,
)
from agent.vcd.vc_edge import VCConnection
from agent.vcd.vc_dynamics import VCDynamics
from agent.vcd.rs_planner import RandomShootingUVPickandPlacePlanner

class VCDAdapter(TrainableAgent):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'vcd'
        #self.writer = SummaryWriter()
        

        

        self.vcd_edge = self.load_edge_model(config.edge_model_path)
        self.vcdynamics = self.load_dynamics_model(config, None, self.vcd_edge)

        self.planner =  RandomShootingUVPickandPlacePlanner(
            config.shooting_number, config.delta_y, config.pull_step, config.wait_step,
            dynamics=self.vcdynamics,
            reward_model=pc_reward_model,
            num_worker=config.num_worker,
            move_distance_range=config.move_distance_range,
            gpu_num=config.gpu_num,
            delta_y_range=config.delta_y_range,
            image_size=(config.camera_height, config.camera_width),
            # matrix_world_to_camera=self.matrix_world_to_camera,
            task=config.task,
            normalize_info={'xz_mean': [0, 0.02]}
        )
    
    def load_edge_model(self, edge_model_path):
        if edge_model_path is not None:
            edge_model_dir = osp.dirname(edge_model_path)
            edge_model_vv = json.load(open(osp.join(edge_model_dir, 'best_state.json')))
            edge_model_vv['eval'] = 1
            edge_model_vv['n_epoch'] = 1
            edge_model_vv['edge_model_path'] = edge_model_path
            edge_model_config = DotMap(edge_model_vv)

            vcd_edge = VCConnection(edge_model_config)
            print('edge GNN model successfully loaded from ', edge_model_path, flush=True)
        else:
            print("no edge GNN model is loaded")
            vcd_edge = None

        return vcd_edge


    def load_dynamics_model(self, config, env, vcd_edge):
        model_vv_dir = osp.dirname(config.partial_dyn_path)
        model_vv = json.load(open(osp.join(model_vv_dir, 'best_state.json')))

        model_vv[
            'fix_collision_edge'] = config.fix_collision_edge  # for ablation that train without mesh edges, if True, fix collision edges from the first time step during planning; If False, recompute collision edge at each time step
        model_vv[
            'use_collision_as_mesh_edge'] = config.use_collision_as_mesh_edge  # for ablation that train with mesh edges, but remove edge GNN at test time, so it uses first-time step collision edges as the mesh edges
        model_vv['train_mode'] = 'vsbl'
        model_vv['use_wandb'] = False
        model_vv['eval'] = 1
        model_vv['load_optim'] = False
        model_vv['pred_time_interval'] = config.pred_time_interval
        model_vv['cuda_idx'] = config.cuda_idx
        model_vv['partial_dyn_path'] = config.partial_dyn_path
        config = DotMap(model_vv)

        vcdynamics = VCDynamics(config, vcd_edge=vcd_edge, env=env)
        return vcdynamics



    def train(self, update_steps, arena = None):
        print('No training for VCD adapter')

    def act(self, state):
        

        #cloth_mask = state['observation']['mask']
        no_op = state['no_op']
        pointcloud = state['pointcloud'] # world-frame point cloud of cloth ## TODO: go st andrews to test this out.
        
        
        voxel_pc = voxelize_pointcloud(pointcloud, self.config.voxel_size)
        observable_particle_indices = np.zeros(len(voxel_pc), dtype=np.int32)
        vel_history = np.zeros((len(observable_particle_indices), self.config.n_his * 3), dtype=np.float32)

        # stop if the cloth is dragged out-of-view
        if len(voxel_pc) == 0:
            print("cloth dragged out of camera view!")
            return no_op
        

        data = {
            'pointcloud': voxel_pc.astype(np.float32),
            'vel_his': vel_history,
            'picker_position': None,
            #'action': env.action_space.sample(),  # action will be replaced by sampled action later
            #'picked_points': picked_points,
            #'scene_params': scene_params,
            'partial_pc_mapped_idx': observable_particle_indices,
            # 'matrix_world_to_camera': matrix_world_to_camera,
        }

        # do planning
        action_sequence, model_pred_particle_pos, model_pred_shape_pos, cem_info, predicted_edges \
            = self.planner.get_action(data)
        

        # set picker to start pos (pick pos), world(robot base) frame
        pick_pos, place_pos = cem_info['start_pos'], cem_info['after_pos']


        return pick_pos, place_pos


    def terminate(self):
        return self.is_terminate


    def load(self, path=None):
        pass

    def save(self):
        pass
    
    def load_checkpoint(self, load_iter):
        pass

    
    def set_train(self):
        pass

    def set_eval(self):
        pass

    def get_writer(self) -> SummaryWriter:
        return self.writer

    def get_phase(self):
        pass

    def get_state(self):
        return {}

    def init(self, information):
        pass

    def update(self, information, action):
        pass

    def reset(self):
        pass

    def get_action_type(self):
        return 'default'
    



