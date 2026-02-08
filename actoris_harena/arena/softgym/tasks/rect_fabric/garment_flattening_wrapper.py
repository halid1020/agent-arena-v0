import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from pyswarm import pso
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist

import logging

from arena.softgym.task_wrappers.task_wrapper import TaskWrapper
from arena.softgym.task_wrappers.metrics_utils import get_wrinkle_pixel_ratio

def rotate_particles_1_axis(particles, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), 0, np.sin(angle_radians)],
        [0, 1, 0],
        [-np.sin(angle_radians), 0, np.cos(angle_radians)]
    ])
    return np.dot(particles, rotation_matrix)

class GarmentFlatteningWrapper(TaskWrapper):
    def __init__(self, env, canonical=False, 
                 domain='mono-square-fabric', initial='crumple'):
        self.env = env
        self.canonical = canonical
        self.successes = []
        self.domain = domain
        self.initial = initial
        self.task_name = 'flattening'
        

    def _process_info(self, info):
        info['goal'] = self.get_goal()
        info['success'] = self.success()
        #info['cloth_size'] = self.env.get_cloth_size()
        return info

    def reset(self, episode_config=None):
        info =  self.env.reset(episode_config)
        self.cur_coverage = info['normalised_coverage']
        info = self._process_info(info)
        self.successes = [info['success']]
        self.goal = self.get_goal()
        self._save_goal()
        return info

    def wait_until_stable(self):
        info = self.env.wait_until_stable()
        info = self._process_info(info)
        info['reward'] = self.env.get_normalised_coverage()
        return info
        
    ### It recives the velocity-grasp signal as action.
    def step(self, action):
        info = self.env.step(action)
        info = self._process_info(info)

        self.last_coverage = self.cur_coverage
        self.cur_coverage = info['normalised_coverage']
        info['reward'] = self.reward(action)
        self.successes.append(info['success'])
        return info
    
    def reward(self, action):

        max_distance = self._get_max_keypoint_distance()

        reward = 1 if max_distance < 0.05 else max(1 - max_distance, 0.0)
        return reward
    
    def get_goals(self):
        return [self.get_goal()]
    
    def get_goal(self):
        
        return self.env.get_flatten_observation()
    
    # def _get_keypoint_distances(self):

    #     nam2keypoints = self.env.get_name2keypoints()

    #     left_keypoints = []
    #     right_keypoints = []

    #     for name, keypoint in nam2keypoints.items():
    #         if 'left' in name:
    #             left_keypoints.append(keypoint)
    #             # replace the left with right
    #             right_name = name.replace('left', 'right')
    #             right_keypoints.append(nam2keypoints[right_name])
        
    #     flattend_particles = self.env.get_flatten_positions()
    #     left_flattened_particles = flattend_particles[left_keypoints]
    #     right_flattened_particles = flattend_particles[right_keypoints]

    #     current_partciels = self.env.get_particle_positions()
    #     left_current_particles = current_partciels[left_keypoints]
    #     right_current_particles = current_partciels[right_keypoints]

    #     left_distance = np.linalg.norm(left_flattened_particles - left_current_particles, axis=1)
    #     right_distance = np.linalg.norm(right_flattened_particles - right_current_particles, axis=1)
    #     distance = np.concatenate([left_distance, right_distance])

    #     left_distance_1 = np.linalg.norm(left_flattened_particles - right_current_particles, axis=1)
    #     right_distance_1 = np.linalg.norm(right_flattened_particles - left_current_particles, axis=1)
    #     distance_1 = np.concatenate([left_distance_1, right_distance_1])

    #     return distance, distance_1

    def _get_keypoint_distances(self):
        nam2keypoints = self.env.get_name2keypoints()

        left_keypoints = []
        right_keypoints = []

        for name, keypoint in nam2keypoints.items():
            if 'left' in name:
                left_keypoints.append(keypoint)
                # replace the left with right
                right_name = name.replace('left', 'right')
                right_keypoints.append(nam2keypoints[right_name])
        
        flattend_particles = self.env.get_flatten_positions()
        current_particles = self.env.get_particle_positions()

        left_flattened_particles = flattend_particles[left_keypoints]
        right_flattened_particles = flattend_particles[right_keypoints]

        left_current_particles = current_particles[left_keypoints]
        right_current_particles = current_particles[right_keypoints]

        distances = []

        for angle in range(0, 360, 5):
            rotated_left_flattened = rotate_particles_1_axis(left_flattened_particles, angle)
            rotated_right_flattened = rotate_particles_1_axis(right_flattened_particles, angle)

            left_distance = np.linalg.norm(rotated_left_flattened - left_current_particles, axis=1)
            right_distance = np.linalg.norm(rotated_right_flattened - right_current_particles, axis=1)
            distance = np.concatenate([left_distance, right_distance])

            left_distance_1 = np.linalg.norm(rotated_left_flattened - right_current_particles, axis=1)
            right_distance_1 = np.linalg.norm(rotated_right_flattened - left_current_particles, axis=1)
            distance_1 = np.concatenate([left_distance_1, right_distance_1])

            distances.append(distance)
            distances.append(distance_1)

        return distances


    def _get_max_keypoint_distance(self):
        distances = self._get_keypoint_distances()
        return min([ max(dis)for dis in distances ])
        #return min(max(d1), max(d2))

    def _get_mean_keypoint_distance(self):
        distances = self._get_keypoint_distances()
        return np.min([np.mean(dis) for dis in distances])
    
    def _get_std_keypoint_distance(self):
        distances = self._get_keypoint_distances()
        stds = [np.std(dis) for dis in distances]
        means = [np.mean(dis) for dis in distances]
        # return the std of the minimum mean distance
        return stds[np.argmin(means)]
        ## chose the minimum mean d1 and d2, then get the std on the choosen one
        #return np.std(d1) if np.mean(d1) < np.mean(d2) else np.std(d2)
    
    def success(self):

        max_distance = self._get_max_keypoint_distance()
        #print('max_distance', max_distance)
        nc = self.env.get_normalised_coverage()
        print('mx distance', max_distance, 'nc', nc)
        return max_distance < 0.08 and nc > 0.99
    
    def get_steps2sucess(self):

        ## return the index of first success form the self.successes array, if there is no True return -1
        return np.argmax(self.successes) if np.any(self.successes) else -1
    
    def evaluate(self,
            metrics = ['normalised_improvement', 'normalised_coverage', 'wrinkle_pixel_ratio',
                       'max_keypoint_distance', 'mean_keypoint_distance', 'std_keypoint_distance',
                       'canonical_IoU','success', 'steps2sucess'],
        ):

        target_coverage = self.env.get_flattened_coverage()
        initial_coverage = self.env.get_initial_coverage()/target_coverage
        current_coverage = self.env.get_coverage()/target_coverage
        target_coverage = 1.0

        ### go over metrics and compute them
        results = {}
        for metric in metrics:
            if metric == 'normalised_improvement':

                ## if the target_coverage and curre_coverage are the same, then the normalised_improvement is 1
                if current_coverage >= 1.0 or abs(target_coverage-current_coverage) < 1e-2:
                    results[metric] = 1.0
                elif initial_coverage >= 1.0 or abs(target_coverage-initial_coverage) < 1e-2:
                    results[metric] = 0.0
                else:
                    ni = (current_coverage - initial_coverage)/(target_coverage - initial_coverage)
                    results[metric] = max(min(1.0, ni), 0.0)

            elif metric == 'normalised_coverage':
                results[metric] = self.env.get_normalised_coverage()
            elif metric == 'wrinkle_pixel_ratio':
                results[metric] = get_wrinkle_pixel_ratio(
                    self.render(mode='rgb'), 
                    self.get_cloth_mask(resolution=(128, 128)))
            elif metric == 'success':
                results[metric] = self.success()
            elif metric == 'canonical_IoU':
                results[metric] = self.get_flatten_canonical_IoU()
            elif metric == 'steps2sucess':
                results[metric] = self.get_steps2sucess()
            elif metric == 'max_keypoint_distance':
                results[metric] = self._get_max_keypoint_distance()
            elif metric == 'mean_keypoint_distance':
                results[metric] = self._get_mean_keypoint_distance()
            elif metric == 'std_keypoint_distance':
                results[metric] = self._get_std_keypoint_distance()
            else:
                raise NotImplementedError


        return results
    
    def get_canonical_hausdorff_distance(self):
        mask = self.get_cloth_mask(resolution=(128, 128))
        canonical_mask = self.get_canonical_mask(resolution=(128, 128))
        hausdorff_distance = directed_hausdorff(mask, canonical_mask)[0]

        return hausdorff_distance
    
    def get_canonical_chamfer_distance(self):
        mask1 = self.get_cloth_mask(resolution=(128, 128))
        mask2 = self.get_canonical_mask(resolution=(128, 128))
        points1 = np.transpose(np.where(mask1))
        points2 = np.transpose(np.where(mask2))

        chamfer_distance = np.sum(np.min(cdist(points1, points2), axis=1)) + np.sum(np.min(cdist(points2, points1), axis=1))

        return chamfer_distance

    ## Version 2
    def get_maximum_IoU(self):
        mask = self.get_cloth_mask(resolution=(128, 128))
        canonical_mask = self.get_canonical_mask(resolution=(128, 128))

        x0 = np.array([0, 0, 0])  # Initial guess for rotation angle, translation x, and translation y

        bounds = [(-180, 180), (-63, 63), (-63, 63)]

        result = differential_evolution(self.calculate_IoU, bounds, args=(mask, canonical_mask.copy()),
                                        disp=True, maxiter=100, workers=1)

        optimal_angle = result.x[0]
        optimal_translation = result.x[1:]

        final_mask = self.rotate_and_translate_image(canonical_mask.copy(), optimal_angle, optimal_translation)

        # plt.imshow(final_mask)
        # plt.show()

        return -result.fun

    def calculate_IoU(self, x, mask, target_mask):
        angle = x[0]
        translation = x[1:]

        transformed_mask = self.rotate_and_translate_image(target_mask, angle, translation)

        intersection = np.sum(np.logical_and(mask, transformed_mask))
        union = np.sum(np.logical_or(mask, transformed_mask))
        IoU = intersection / union

        return -IoU  # Minimize the negative IoU

    def rotate_and_translate_image(self, image, angle, translation):
        translated_mask = np.roll(image, (int(translation[0]), int(translation[1])), axis=(0, 1))
        rotated_and_translated_image = self.rotate_image(translated_mask, angle)

        return rotated_and_translated_image

    def rotate_image(self, image, angle):
        angle_rad = np.radians(angle)
        rotated_image = np.zeros_like(image)
        center = (image.shape[0] / 2, image.shape[1] / 2)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                x, y = np.dot(rotation_matrix, (i - center[0], j - center[1])) + center
                x = int(round(x))
                y = int(round(y))
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    rotated_image[x, y] = image[i, j]

        return rotated_image

        
    def rotate_and_translate_image(self, image, angle, translation):
        translated_mask = np.roll(image, (int(translation[0]), int(translation[1])), axis=(0, 1))
        rotated_and_translated_image = self.rotate_image(translated_mask, angle)

        return rotated_and_translated_image
    
    def rotate_image(self, image, angle):
        angle_rad = np.radians(angle)
        rotated_image = np.zeros_like(image)
        center = (image.shape[0] / 2, image.shape[1] / 2)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                x, y = np.dot(rotation_matrix, (i - center[0], j - center[1])) + center
                x = int(round(x))
                y = int(round(y))
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    rotated_image[x, y] = image[i, j]

        return rotated_image

         

    

    def get_wrinkle_pixel_ratio(self, particles=None):
        rgb = self.render(mode='rgb')
        rgb = cv2.resize(rgb, (128, 128))
        mask = self.get_cloth_mask(resolution=(128, 128))

        if mask.dtype != np.uint8:  # Ensure mask has a valid data type (uint8)
            mask = mask.astype(np.uint8)


        # Use cv2 edge detection to get the wrinkle ratio.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # plt.imshow(edges)
        # plt.show()

        masked_edges = cv2.bitwise_and(edges, mask)
        # plt.imshow(masked_edges)
        # plt.show()

        wrinkle_ratio = np.sum(masked_edges) / np.sum(mask)

        return wrinkle_ratio
    

