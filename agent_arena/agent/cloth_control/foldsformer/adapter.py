import os
import json
from dotmap import DotMap
import torch
import numpy as np
from einops import rearrange
import imageio
import cv2

from agent_arena import TrainableAgent
from agent_arena.utilities.utils import TrainWriter
from .utils.setup_model import get_configs, setup_model
from .utils.visual import nearest_to_mask

class FoldsformerAdapter(TrainableAgent):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'vcd'
        self.writer = TrainWriter()
        
        self.device = config.device
        
        model_config_path = os.path.join(os.environ['AGENT_ARENA_PATH'], "agent", "cloth_control", "foldsformer", "train.yaml")
        configs = get_configs(model_config_path)
        self.trained_model_path = os.path.join(os.environ['AGENT_ARENA_PATH'], "data", "foldsformer", "trained model", "foldsformer_eval.pth")
        self.net = setup_model(configs)
        self.net = self.net.to(self.device)
        self.img_size = 224

        self.set_goals()

    
    def set_goals(self):
        self.goal_config = {
            'corners-edge-inward-folding': {
                'alias': 'CornersEdgesInward',
                'frames_idx': [0, 1, 2, 3, 4],
                'steps': 4
            },
            'all-corner-inward-folding': {
                'alias': 'AllCornersInward',
                'frames_idx': [0, 1, 2, 3, 4],
                'steps': 4
            },
            'cross-folding': {
                'alias': 'DoubleStraight',
                'frames_idx': [0, 1, 2, 3, 3],
                'steps': 3
            },
            'diagonal-cross-folding': {
                'alias': 'DoubleTriangle',
                'frames_idx': [0, 1, 1, 2, 2],
                'steps': 2
            }
        }

        demo_dir = os.path.join(os.environ['AGENT_ARENA_PATH'], "data", "foldsformer", "demo")

        for task, config in self.goal_config.items():
            task_dir = os.path.join(demo_dir, config['alias'])
            depth_dir = os.path.join(task_dir, "depth")
            goal_frames = []
            for i in config['frames_idx']:
                frame = imageio.imread(os.path.join(depth_dir, str(i) + ".png")) / 255
                frame = torch.FloatTensor(self.orginal_preprocess(frame)).unsqueeze(0).unsqueeze(0)
                goal_frames.append(frame)
            
            self.goal_config[task]['goal_frames'] = torch.cat(goal_frames, dim=0)
        
    def original_get_mask(self, depth):
        mask = depth.copy()
        mask[mask > 0.646] = 0
        mask[mask != 0] = 1
        return mask

    def orginal_preprocess(self, depth):
        mask = self.original_get_mask(depth)
        depth = depth * mask
        return depth


    def train(self, update_steps, arena = None):
        print('No training for Foldsformer adapter')

    def act(self, state):

        depth = state['observation']['depth']
        mask = state['observation']['mask']

        ## resize to img_size
        depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.float), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.9).astype(np.bool8) 

        task = state['task']
        goal_frames = self.goal_config[task]['goal_frames']

        # normalise depth
        # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

        # ## make depth to 0.55 to 0.65
        # depth = depth * 0.1 + 0.55
        mask_depth = depth*mask

        current_state = torch.FloatTensor(mask_depth).unsqueeze(0).unsqueeze(0)
        current_frames = torch.cat((current_state, goal_frames), dim=0).unsqueeze(0)
        current_frames = rearrange(current_frames, "b t c h w -> b c t h w")
        current_frames = current_frames.to(self.device)
        

        pickmap, placemap = self.net(current_frames)
        pickmap = torch.sigmoid(torch.squeeze(pickmap))
        placemap = torch.sigmoid(torch.squeeze(placemap))
        pickmap = pickmap.detach().cpu().numpy()
        placemap = placemap.detach().cpu().numpy()

        test_pick_pixel = np.array(np.unravel_index(pickmap.argmax(), pickmap.shape))
        test_place_pixel = np.array(np.unravel_index(placemap.argmax(), placemap.shape))

        #mask = state['observation']['mask'] 
        test_pick_pixel_mask = nearest_to_mask(test_pick_pixel[0], test_pick_pixel[1], mask)
        test_pick_pixel[0], test_pick_pixel[1] = test_pick_pixel_mask[1], test_pick_pixel_mask[0]
        test_place_pixel[0], test_place_pixel[1] = test_place_pixel[1], test_place_pixel[0]
        
        pixel_action =  np.array([test_pick_pixel[0], test_pick_pixel[1], test_place_pixel[0], test_place_pixel[1]])

        action = pixel_action/self.img_size * 2 - 1
        return action.reshape(-1, 4)


    def terminate(self):
        return self.step > self.target_steps

    def load(self, path=None):
        self.net.load_state_dict(torch.load(self.trained_model_path)["model"])
        print(f"load trained model from {self.trained_model_path}")
        return -1

    def save(self):
        pass
    
    def load_checkpoint(self, load_iter):
        pass

    
    def set_train(self):
        pass

    def set_eval(self):
        self.net.eval()

    def get_writer(self) -> TrainWriter:
        return self.writer

    def get_phase(self):
        return "fold"

    def get_state(self):
        return {}

    def init(self, information):
        self.target_steps = self.goal_config[information['task']]['steps']
        

    def update(self, information, action):
        self.step += 1

    def _reset(self):
        self.step = 0

    def get_action_type(self):
        return 'default'
    



