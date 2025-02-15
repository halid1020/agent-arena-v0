import numpy as np
import cv2
from utilities.torch_utils import *
from tqdm import tqdm

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

class OpenSTL_Adapter():

    def __init__(self, config):
        ### Args will be a dot map
        parser = create_parser()

        args = parser.parse_args(config.command.split(" "))
        config = args.__dict__
        
        assert args.config_file is not None, "Config file is required for testing"
        config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method', 'batch_size', 'val_batch_size'])
        config['test'] = True

        self.exp = BaseExperiment(args)
        self.exp.test()
        self.args = args
    
    def unroll_action_from_cur_state(self, actions):
        B, T = actions.shape[:2]

        ### Pad the actions with no-po so that it can be fed into the model
        ## TODO: make this more general
        actions = np.concatenate([actions, np.ones((B, self.args.pre_seq_length-T, *actions.shape[2:]))], axis=1)

        images = np.asarray(self.past_images[-self.args.pre_seq_length:])

        #### Repeat the numpy image and make into shape (B, self.args.pre_seq_length, 3, H, W)
        images = np.repeat(images, B, axis=0).reshape(B, self.args.pre_seq_length, *images.shape[1:])

        ### Predict the future images
        preds = []
        bs = self.args.batch_size
        for i in tqdm(range(0, B, bs)):
            batchx = {
                'image': np_to_ts(images[i: i+bs], self.args.device).float()/255.0,
                'future_action': np_to_ts(actions[i: i+bs], self.args.device).float()
            }
            pred = ts_to_np(self.exp.method._predict(batchx))[:, :T]
                            
            preds.append(pred)

        preds = (np.concatenate(preds, axis=0).transpose(0, 1, 3, 4, 2)*255.0).astype(np.uint8)

        return preds

    def visual_reconstruct(self, pred_traj):
        return pred_traj

    def init(self, state):
        self.past_images = []
        image = state['observation']['image']
        H, W = self.args.in_shape[2:]
        image = cv2.resize(image, (H, W), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)

        for i in range(self.args.pre_seq_length):
            self.past_images.append(image)

    def update_state(self, state,  action):
        image = state['observation']['image']
        H, W = self.args.in_shape[2:]
        image = cv2.resize(image, (H, W), interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)

        self.past_images.append(image)