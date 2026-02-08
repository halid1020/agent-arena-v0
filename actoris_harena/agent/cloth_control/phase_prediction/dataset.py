# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens
"""Image dataset."""

import os
import pickle
import cv2

import numpy as np
from arena.raven import tasks
from arena.raven.tasks import cameras
from matplotlib import pyplot as plt



class Dataset:
    """A simple image dataset class."""

    def __init__(self, path, img_shape=None, save_contour=False, save_mask=False):
        """A simple RGB-D image dataset."""
        self.path = path
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.img_shape = img_shape
        self.save_contour = save_contour
        self.save_mask = save_mask

        # Track existing dataset if it exists.
        color_path = os.path.join(self.path, 'phases')
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
        rgb, depth, mask, phases = [], [], [], []
        if self.save_mask:
            mask = []
        if self.save_contour:
            contour = []

        for obs, phase in episode:
            #print('obs', obs.keys())
            rgb.append(obs['rgb'])
            depth.append(obs['depth'])
            
            phases.append(phase)
            if self.save_mask:
                mask.append(obs['mask'])
            if self.save_contour:
                contour.append(obs['contour'])
            #self.n_steps += 1


        def dump(data, field):
            field_path = os.path.join(self.path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
            with open(os.path.join(field_path, fname), 'wb') as f:
                pickle.dump(data, f)

        dump(rgb, 'rgb')
        dump(depth, 'depth')
        
        if self.save_mask:
            dump(mask, 'mask')
        if self.save_contour:
            dump(contour, 'contour')

        dump(phases, 'phases')

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
            if not os.path.exists(path):
                return None, False
            data = pickle.load(open(os.path.join(path, fname), 'rb'))
            if cache:
                self._cache[episode_id][field] = data
            return data, True

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self.path, 'phases')
        for fname in sorted(os.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                rgb, _ = load_field(episode_id, 'rgb', fname)
                depth, _ = load_field(episode_id, 'depth', fname)
                mask, mask_exist = load_field(episode_id, 'mask', fname)
                contour, contour_exist = load_field(episode_id, 'contour', fname)
                phases, _ = load_field(episode_id, 'phases', fname)
                
                # Reconstruct episode.
                episode = []
                for i in range(len(phases)):
                    ## resize image to self.img_shape
                    #print('rgb before', rgb[i].shape)
                    rgb_ = cv2.resize(rgb[i], self.img_shape)
                    #print('rgb after', rgb_.shape)
                    #print('depth', depth[i].shape)
                    depth_ = cv2.resize(depth[i], self.img_shape)
                    ### if mask is 2D, then add on more dimension

                    if mask_exist:
                        if len(mask[i].shape) == 2:
                            mask[i] = np.expand_dims(mask[i], axis=-1)
                        #print('mask', mask[i].shape)
                        
                        mask_ = cv2.resize(mask[i].astype(np.float32), self.img_shape)
                        mask_ = (mask_ > 0.5).astype(np.float32)
                    else:
                        mask_ = depth_ < 1.499
                    
                    if contour_exist:
                        if len(contour[i].shape) == 2:
                            contour[i] = np.expand_dims(contour[i], axis=-1)
                        contour_ = cv2.resize(contour[i].astype(np.float32), self.img_shape)
                        #contour_ = (contour_ > 0.5).astype(np.float32)

                    obs = {'rgb': rgb_,
                           'depth': depth_} if images else {}
                    obs['mask'] = mask_
                    if contour_exist and images:
                        obs['contour'] = contour_

                    episode.append((obs, phases[i]))
                return episode, seed

    def sample(self, images=True, cache=False):
        """Uniformly sample from the dataset.

        Args:
          images: load image data if True.
          cache: load data from memory if True.

        Returns:
          sample: randomly sampled (obs, phase) tuple.
        """

        # Choose random episode.
        if len(self.sample_set) > 0:  # pylint: disable=g-explicit-length-test
            episode_id = np.random.choice(self.sample_set)
        else:
            episode_id = np.random.choice(range(self.n_episodes))
        episode, _ = self.load(episode_id, images, cache)

        # Return random observation action pair (and goal) from episode.
        i = np.random.choice(range(len(episode) - 1))
        sample = episode[i]
        return sample

    def sample_batch(self, 
            batch_size=1, 
            images=True, 
            cache=False, 
            balance_phase_sampling=False,
            num_phases=None):
        """Uniformly sample a batch from the dataset.

        Args:
          batch_size: number of samples to return.
          images: load image data if True.
          cache: load data from memory if True.

        Returns:
          batch: randomly sampled (obs, phase) tuple.
        """
        batch = []
        if balance_phase_sampling:
            assert num_phases is not None

            num_sample_per_phase = batch_size // num_phases
            sampled_num_phases = [0 for _ in range(num_phases)]
            
            # sample but the number for each phases will not different more than 1
            while True:
                obs, phase = self.sample(images, cache)
                if phase >= num_phases:
                    continue
                if sampled_num_phases[phase] < num_sample_per_phase:
                    batch.append((obs, phase))
                    sampled_num_phases[phase] += 1
                if len(batch) == num_sample_per_phase * num_phases:
                    break
            
            # sample the rest
            while len(batch) < batch_size:
                batch.append(self.sample(images, cache))
            
            return batch

        for _ in range(batch_size):
            
            while True:
                obs, phase = self.sample(images, cache)
                if phase < num_phases:
                    break
            
            batch.append((obs, phase))
        return batch

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