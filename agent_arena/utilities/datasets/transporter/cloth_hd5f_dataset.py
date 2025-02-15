import numpy as np
import matplotlib.pyplot as plt

from agent_arena.utilities.datasets.cloth_vision_pick_and_place_hd5f_dataset import ClothVisionPickAndPlaceHDF5Dataset

class TransporterCltohHd5fDataset:
   

    def __init__(self, **kwargs): 
        self.sample_set = []
        ## We assume the dataset already filter the success episodes
        self._dataset = ClothVisionPickAndPlaceHDF5Dataset(**kwargs)
        self.n_episodes = self._dataset._N
        self.sample_nc_tresholds = kwargs['sample_nc_thresholds'] if 'sample_nc_thresholds' in kwargs else [(0.0, 1.0)]
        self.filter_no_op = kwargs['filter_no_op'] if 'filter_no_op' in kwargs else False

    def load(self, episode_id, images=True, cache=False):
        loaded_episode = self._dataset._load(episode_id)

        episode = []

        for i in range(len(loaded_episode['action'])):
            obs = {
                'color': loaded_episode['rgb'][i],
                'depth': loaded_episode['depth'][i],
            }
            actions = loaded_episode['action'][i].reshape(-1, 2, 3)[:, :, :2]
            
            load_actions = actions.copy()
            load_actions[:, :, 0] = actions[:, :, 1]
            load_actions[:, :, 1] = actions[:, :, 0]

            episode.append((
                obs, 
                ## make ndarry to float numpy array
                load_actions.reshape(-1, 4).astype(np.float32), #action
                loaded_episode['normalised_coverage'][i], #reward
                None))
            
        return episode, None
    
    def __len__(self):
        return self._dataset.__len__()
    
    def sample(self, images=True, cache=False):
        """Uniformly sample from the dataset.

        Args:
          images: load image data if True.
          cache: load data from memory if True.

        Returns:
          sample: randomly sampled (obs, act, reward, info) tuple.
          goal: the last (obs, act, reward, info) tuple in the episode.
        """

        # Choose random episode.
        

        # Return random observation action pair (and goal) from episode.

        while True:
            if len(self.sample_set) > 0:  # pylint: disable=g-explicit-length-test
                episode_id = np.random.choice(self.sample_set)
            else:
                episode_id = np.random.choice(range(self.n_episodes))
            episode, _ = self.load(episode_id, images, cache)

            i = np.random.choice(range(len(episode) - 1))
            #print('i', i)
            sample, goal = episode[i], episode[-1]
            #print('reward', sample[2])
            can_sample = False

            if self.filter_no_op and np.min(np.absolute(sample[1])) >= 0.99:
                #print('filter no op')
                continue

            for sample_nc_treshold in self.sample_nc_tresholds:
                if sample_nc_treshold[0] <= sample[2] <= sample_nc_treshold[1]:
                    can_sample = True
                    break

            if can_sample:
                break

        ## Plot sample and goal
        # print('sample[1]', sample[1],  np.min(np.absolute(sample[1])))
        
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(sample[0]['color'])
        # ax[1].imshow(goal[0]['color'])
        # plt.show()

        return sample, goal