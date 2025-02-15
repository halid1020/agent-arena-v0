import torch
import numpy as np

from agent_arena.agent.utilities.torch_utils import ts_to_np
from agent_arena.utilities.transform.pick_and_place_transformer import PickAndPlaceTransformer

def gaussian_kernel(size, sigma):
    x = torch.arange(size).float()
    y = torch.arange(size).float()
    x, y = torch.meshgrid(x, y)
    x = x - size//2 
    y = y - size//2
    dist = x.pow(2) + y.pow(2)
    kernel = torch.exp(-dist / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def create_heatmap(pixel_positions, size, sigma, kernel_size):
    # Create empty heatmap
    heatmap = torch.zeros(size).to(pixel_positions.device)

    # Convert pixel positions to heatmap indices
    indices = (pixel_positions + 1) / 2 * torch.tensor(size).to(pixel_positions.device)
    indices = ts_to_np(indices.long())

    # Create kernel and add to heatmap at corresponding indices
    kernel = gaussian_kernel(kernel_size, sigma).to(pixel_positions.device)
    k_size = kernel_size
    
    x, y =  indices[1], indices[0]
    x_min, x_max = max(0, x - k_size // 2), min(size[0], x + k_size // 2 + 1)
    y_min, y_max = max(0, y - k_size // 2), min(size[1], y + k_size // 2 + 1)

    k_x_min, k_x_max = max(0, k_size//2 - (x-x_min)), min(k_size, k_size // 2 + (x_max-x))
    k_y_min, k_y_max = max(0, k_size//2 - (y-y_min)), min(k_size, k_size // 2 + (y_max-y))

    kernel_slice = kernel[k_x_min:k_x_max, k_y_min:k_y_max]
    heatmap[x_min:x_max, y_min:y_max] += kernel_slice

    return heatmap

 ## Heatmap dimension Any*2*H*W
def get_action_from_heatmap(heatmap):
    original_shape = heatmap.shape[:-3]
    H, W = heatmap.shape[-2:]
    heatmap = heatmap.reshape(-1, 2, heatmap.shape[-2], heatmap.shape[-1])



    # Convert the indices to x-y coordinates
    pick_idx = np.argmax(heatmap[:, 0].reshape(-1, H*W), axis=1)
    
    pick_action = np.stack([np.asarray(np.unravel_index(idx, (H, W))) for idx in pick_idx])
    tmp = pick_action.copy()
    pick_action[:, 0], pick_action[:, 1] = tmp[:, 1], tmp[:, 0] 
    pick_action = pick_action/np.asarray([H, W])*2 - 1
    
    place_idx = np.argmax(heatmap[:, 1].reshape(-1, H*W), axis=1)
    place_action = np.stack([np.asarray(np.unravel_index(idx, (H, W))) for idx in place_idx])
    tmp = place_action.copy()
    place_action[:, 0], place_action[:, 1] = tmp[:, 1], tmp[:, 0]
    place_action = place_action/np.asarray([H, W])*2 - 1


    action = np.concatenate((pick_action, place_action), axis=1).reshape(original_shape + (-1,))
    return action

class PickAndPlaceHeatmapTransformer(PickAndPlaceTransformer):
    def __init__(self,  config=None):
        super().__init__(config)

    def __call__(self, sample, train=True):
        # batch is assumed to have the shape B*T*C*H*W
        sample = super().__call__(sample, train)

        actions = sample['action']
        T = actions.shape[0]
        H, W = self.config.heatmap_size

        # assuming the action trajectory is stored in a PyTorch tensor called actions
        pick_heatmaps = []
        place_heatmaps = []

        pick_heatmaps = [create_heatmap(actions[:, :2][t], size=(H, W), sigma=self.config.sigma, kernel_size=self.config.kernel_size) \
            for t in range(T)]
        place_heatmaps = [create_heatmap(actions[:, 2:][t], size=(H, W), sigma=self.config.sigma, kernel_size=self.config.kernel_size) \
            for t in range(T)]

        sample['pick_heatmap'] = torch.stack(pick_heatmaps) * self.config.scale
        sample['place_heatmap'] = torch.stack(place_heatmaps) * self.config.scale

        return sample


        
    def post_transform(self, sample):
        pick_heatmap = sample['pick_heatmap']

        place_heatmap = sample['place_heatmap']
        sample['action'] = get_action_from_heatmap(
            ts_to_np(torch.concat([pick_heatmap.unsqueeze(1),place_heatmap.unsqueeze(1)], dim=1)))
        
        sample['action'] = torch.from_numpy(sample['action']).float()

        return super().post_transform(sample)