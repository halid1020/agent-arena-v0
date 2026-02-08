import torch

class NormalSampler:

    def sample(self, state, horizon, action_dim):

        return torch.randn((1, horizon, action_dim))

class MaskPickUniformSampler:

    def sample(self, state, horizon, action_dim):
        mask = state['mask'][-1][0]
        target_id = len(state['mask'])
        #print('horizon:', horizon)
        #print('sampler mask:', mask.shape)
        H = mask.shape[0]
        ## get the dimension of the masks, 
        ## masks is in horizon*H*W

        ### for each horizon, sample points from the mask
        picks = torch.randn((horizon, 2))
        #print('sampler picks:', picks.shape)

        indices = torch.nonzero(mask)
        num_points = indices.shape[0]
        idx = torch.randint(0, num_points, (1,))[0]
        

        target_pick = indices[idx]
        #print('pre target_pick:', target_pick)

        picks[target_id] = target_pick/H * 2 -1
        #print('target pick', picks[target_id])

        #print('target pick:', target_pick)
        
        #print('sampler picks:', picks.shape)
        ## uniformly sample places form -1 to 1
        places = torch.randn((horizon, 2))
        #print('sampler places:', places.shape)

        sample_action = torch.cat([picks, places], dim=-1).unsqueeze(0)
        #print('sample action:', sample_action.shape)
        #print('action_dim:', action_dim)
        return sample_action


ActionSampler = {
    'normal': NormalSampler,
    'mask_pick_uniform': MaskPickUniformSampler
}