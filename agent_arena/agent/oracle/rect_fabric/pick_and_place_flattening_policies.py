import cv2
import numpy as np
import random 
from scipy.spatial import ConvexHull #, convex_hull_plot_2d
import logging
import gym

from ...oracle.random_pick_and_place_policy import RandomPickAndPlacePolicy
from agent_arena.agent.utilities.utils import *


class RectFabricPickAndPlaceCornerBiasedPolicy(RandomPickAndPlacePolicy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('corner-biased-random-pick-and-place')

    def act(self, state):
        action = super().act(state)#
        action[0, :2] = self._sample_pick_action_using_harris_detector(state['observation']['rgb'][:, :, :3].astype(np.uint8))
        self.action_type = 'corner-biased-random-pick-and-place'
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action.copy().reshape(*self.action_dim)


    def _sample_pick_action_using_harris_detector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)

        dst = cv2.dilate(dst,None)

        H, W = image.shape[:2]
        #image[dst>0.01*dst.max()]=[0,0,255]

        ys, xs = np.where(dst>0.01*dst.max())

        
        num = len(xs)
        i = random.randint(0, num-1)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (0 <= ys[i]+dx < H) and (0 < xs[i]+dy <  W):
                    image[ys[i]+dx][xs[i]+dy] = [0,0,255]
        

        x, y = (xs[i]/W)*2-1, (ys[i]/H)*2-1
        
        return np.asarray([x, y])

class RectFabricClothMaskSmallDragPolicy(RandomPickAndPlacePolicy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_types.append('cloth-mask-small-drag')
        self.drag_radius = kwargs.get('drag_radius', 0.1)

    def act(self, state):
        cloth_mask = state['observation']['mask']
        arena = state['arena']
        action = np.zeros((1, 4))
        #cloth_mask = environment.get_cloth_mask()
        ## sample a point from the cloth mask
        ys, xs = np.where(cloth_mask)
        num = len(xs)
        i = random.randint(0, num-1)
        x, y = (xs[i]/cloth_mask.shape[1])*2-1, (ys[i]/cloth_mask.shape[0])*2-1

        action[0, :2] = np.asarray([x, y])
        action[0, 2:] = np.asarray([x, y]) + self.drag_radius*np.random.randn(2)
        
        return self.hueristic_z(arena, action.copy()).reshape(*self.action_space.shape)