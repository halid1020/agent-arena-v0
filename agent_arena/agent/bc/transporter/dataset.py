# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens
"""Image dataset."""

import os
import pickle
import cv2

import numpy as np
from agent_arena.arena.raven import tasks
from agent_arena.arena.raven.tasks import cameras
from matplotlib import pyplot as plt


# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


class Dataset:
    """A simple image dataset class."""

    def __init__(self, path, swap_action=False, img_shape=None, 
        save_mask=False, save_contour=False, n_sample=-1):
        """A simple RGB-D image dataset."""
        self.path = path
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.n_sample = n_sample
        self.swap_action = swap_action
        self.img_shape = img_shape
        self.save_mask = save_mask
        self.save_contour = save_contour

        # Track existing dataset if it exists.
        color_path = os.path.join(self.path, 'action')
        if os.path.exists(color_path):
            for fname in sorted(os.listdir(color_path)):
                if '.pkl' in fname:
                    seed = int(fname[(fname.find('-') + 1):-4])
                    self.n_episodes += 1
                    self.max_seed = max(self.max_seed, seed)

        self._cache = {}

    def add(self, seed, episode):
        """Add an episode to the dataset.

        Args:
          seed: random seed used to initialize the episode.
          episode: list of (obs, act, reward, info) tuples.
        """
        color, depth, action, reward, info = [], [], [], [], []
        if self.save_mask:
            mask = []
        if self.save_contour:
            contour = []
        
        for obs, act, r, i in episode:
            #print('color', obs['color'].shape)
            if 'color' in obs:
                color.append(obs['color'])
            else:
                color.append(obs['rgb'])
            if self.save_mask:
                mask.append(obs['mask'].reshape(*obs['mask'].shape[:2], -1).astype(np.float32))
            if self.save_contour:
                contour.append(obs['contour'].reshape(*obs['contour'].shape[:2], -1).astype(np.float32))

            depth.append(obs['depth'])
            action.append(act)
            reward.append(r)
            info.append(i)

        # for imgs in color:
        #     #show the image
        #     for img in imgs:
        #         plt.imshow(img)
        #         plt.show()

        # # resize color and depth according img_shape

        ## check if color is a tuple
        if (not isinstance(color[0], tuple)) and (not isinstance(depth[0], list)):
            #print('color[0]', color[0])
            color = [cv2.resize(img, self.img_shape) for img in color]
            depth = [cv2.resize(img, self.img_shape) for img in depth]
            if self.save_mask:
                mask = [cv2.resize(img, self.img_shape) for img in mask]
                mask = np.float32(mask)
            if self.save_contour:
                contour = [cv2.resize(img, self.img_shape) for img in contour]
                contour = np.float32(contour)
            color = np.uint8(color)
            depth = np.float32(depth)


        def dump(data, field):
            field_path = os.path.join(self.path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
            with open(os.path.join(field_path, fname), 'wb') as f:
                pickle.dump(data, f)

        dump(color, 'color')
        dump(depth, 'depth')
        dump(action, 'action')
        dump(reward, 'reward')
        dump(info, 'info')
        #print('saving')
        if self.save_mask:
            print('save mask', mask[0].shape)
            dump(mask, 'mask')
        if self.save_contour:
            print('save contour', contour[0].shape)
            dump(contour, 'contour')


        self.n_episodes += 1
        self.max_seed = max(self.max_seed, seed)

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.sample_set = episodes

    def load(self, episode_id, images=True, cache=False):
        """Load data from a saved episode.

        Args:
          episode_id: the ID of the episode to be loaded.
          images: load image data if True.
          cache: load data from memory if True.

        Returns:
          episode: list of (obs, act, reward, info) tuples.
          seed: random seed used to initialize the episode.
        """

        def load_field(episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self.path, field)
            data = pickle.load(open(os.path.join(path, fname), 'rb'))
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self.path, 'action')
        for fname in sorted(os.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                color = load_field(episode_id, 'color', fname)
                depth = load_field(episode_id, 'depth', fname)
                action = load_field(episode_id, 'action', fname)
                if self.save_mask:
                    mask = load_field(episode_id, 'mask', fname)
                # else:
                #     mask = depth < 1.499

                if self.save_contour:
                    contour = load_field(episode_id, 'contour', fname)

                if self.swap_action:
                    new_action = []
                    #print('before swap', action[0])
                    for act in action:
                        if act is not None:
                            new_act = act.copy()
                            new_act[:, 0], new_act[:, 1] = act[:, 1], act[:, 0]
                            new_act[:, 2], new_act[:, 3] = act[:, 3], act[:, 2]
                            new_action.append(new_act)
                        else:
                            new_action.append(act)
                   
                    action = new_action
                    #print('after swap', action[0])
                    ## action is a list of 2d array with shape 1, 4
                    ## I want to swap the first and second column, and third and fourth column
                reward = load_field(episode_id, 'reward', fname)
                info = load_field(episode_id, 'info', fname)
                
                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {'color': color[i],
                           'depth': depth[i]} if images else {}
                    
                    if self.save_mask:
                        obs['mask'] = mask[i]
                    if self.save_contour:
                        obs['contour'] = contour[i]

                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed
            
        print('data of episode id {} is not in the the path'.format(episode_id))
        exit()

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
        while True:
            sample = self.n_sample if self.n_sample != -1 else self.n_episodes
            #print('sampel numbers', sample)
            if len(self.sample_set) > 0:  # pylint: disable=g-explicit-length-test
                #print('here')
                episode_id = np.random.choice(self.sample_set[:sample])
            else:
                episode_id = np.random.choice(range(sample))
            
            # print('episode id', episode_id)
            # print('sample', sample)
            
            episode, _ = self.load(episode_id, images, cache)
            #print('len of episode', len(episode))
            if len(episode) > 1:
                break

        # Return random observation action pair (and goal) from episode.
        i = np.random.choice(range(len(episode) - 1))
        sample, goal = episode[i], episode[-1]
        #print('here')
        return sample, goal


def load_data(FLAGS, only_test=False):
    test_path = os.path.join(FLAGS.data_dir, f'{FLAGS.task}-test')
    train_path = os.path.join(FLAGS.data_dir, f'{FLAGS.task}-train')

    if FLAGS.verbose:
        if not only_test:
            print(f"Loading trainset from {train_path}")
        print(f"Loading testset  from {test_path}")

    if only_test:
        return Dataset(test_path)

    return Dataset(train_path), Dataset(test_path)