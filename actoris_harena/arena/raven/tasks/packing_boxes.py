# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Packing task."""

import os

import numpy as np
from ...raven.tasks.task import Task
from actoris_harena.agent.bc.transporter.utils import utils

import pybullet as p


class PackingBoxes(Task):
    """Packing task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20

    def reset(self, env):
        super().reset(env)
        
        self._add_instance(env)

    def _add_instance(self, env):
        # Add container box.
        zone_size = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        os.remove(container_urdf)

        margin = 0.01
        min_object_dim = 0.05
        bboxes = []

        class TreeNode:

            def __init__(self, parent, children, bbox):
                self.parent = parent
                self.children = children
                self.bbox = bbox  # min x, min y, min z, max x, max y, max z

        def KDTree(node):
            size = node.bbox[3:] - node.bbox[:3]

            # Choose which axis to split.
            split = size > 2 * min_object_dim
            if np.sum(split) == 0:
                bboxes.append(node.bbox)
                return
            split = np.float32(split) / np.sum(split)
            split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

            # Split along chosen axis and create 2 children
            cut_ind = np.random.rand() * \
                (size[split_axis] - 2 * min_object_dim) + \
                node.bbox[split_axis] + min_object_dim
            child1_bbox = node.bbox.copy()
            child1_bbox[3 + split_axis] = cut_ind - margin / 2.
            child2_bbox = node.bbox.copy()
            child2_bbox[split_axis] = cut_ind + margin / 2.
            node.children = [
                TreeNode(node, [], bbox=child1_bbox),
                TreeNode(node, [], bbox=child2_bbox)
            ]
            KDTree(node.children[0])
            KDTree(node.children[1])

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = TreeNode(None, [], bbox=np.array(root_size))
        KDTree(root)

        colors = [utils.COLORS[c] for c in utils.COLORS if c != 'brown']

        # Add objects in container.
        object_points = {}
        object_ids = []
        bboxes = np.array(bboxes)
        object_template = 'box/box-template.urdf'
        for bbox in bboxes:
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            pose = (position, (0, 0, 0, 1))
            pose = utils.multiply(zone_pose, pose)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)
            os.remove(urdf)
            object_ids.append((box_id, (0, None)))
            icolor = np.random.choice(range(len(colors)), 1).squeeze()
            p.changeVisualShape(box_id, -1, rgbaColor=colors[icolor] + [1])
            object_points[box_id] = self.get_object_points(box_id)

        # Randomly select object in box and save ground truth pose.
        object_volumes = []
        true_poses = []
        # self.goal = {'places': {}, 'steps': []}
        for object_id, _ in object_ids:
            true_pose = p.getBasePositionAndOrientation(object_id)
            object_size = p.getVisualShapeData(object_id)[0][3]
            object_volumes.append(np.prod(np.array(object_size) * 100))
            pose = self.get_random_pose(env, object_size)
            p.resetBasePositionAndOrientation(object_id, pose[0], pose[1])
            true_poses.append(true_pose)
            # self.goal['places'][object_id] = true_pose
            # symmetry = 0  # zone-evaluation: symmetry does not matter
            # self.goal['steps'].append({object_id: (symmetry, [object_id])})
        # self.total_rewards = 0
        # self.max_steps = len(self.goal['steps']) * 2

        # Sort oracle picking order by object size.
        # self.goal['steps'] = [
        #     self.goal['steps'][i] for i in
        # .    np.argsort(-1 * np.array(object_volumes))
        # ]

        self.goals.append((
            object_ids, np.eye(
                len(object_ids)), true_poses, False, True, 'zone',
            (object_points, [(zone_pose, zone_size)]), 1))

 
    # def reward(self):
    #     """Custom, restorative reward specifically for the PackingBoxes task."""
    #     reward, info = 0, {}

    #     # 1. Cache the ENTIRE goal tuple the first time we see it.
    #     # This contains the target poses the Oracle needs to fix mistakes.
    #     if getattr(self, 'full_goal', None) is None and len(self.goals) > 0:
    #         self.full_goal = self.goals[0]

    #     # Failsafe: if somehow empty and never cached
    #     if getattr(self, 'full_goal', None) is None:
    #         return 0, {}

    #     # 2. Extract parameters from the cached full_goal
    #     objs, matches, targs, replace, rotations, metric, params, max_reward = self.full_goal
    #     obj_pts, zones = params

    #     # 3. ALWAYS evaluate the physical state of the blocks
    #     zone_pts, total_pts = 0, 0
    #     for zone_pose, zone_size in zones:
    #         for obj_id in obj_pts:
    #             pts = obj_pts[obj_id]
                
    #             if pts.shape[1] == 0:
    #                 continue
                    
    #             obj_pose = p.getBasePositionAndOrientation(obj_id)
    #             world_to_zone = utils.invert(zone_pose)
    #             obj_to_zone = utils.multiply(world_to_zone, obj_pose)
    #             pts_local = np.float32(utils.apply(obj_to_zone, pts))
                
    #             if len(zone_size) > 1:
    #                 valid_pts = np.logical_and.reduce([
    #                     pts_local[0, :] > -zone_size[0] / 2, pts_local[0, :] < zone_size[0] / 2,
    #                     pts_local[1, :] > -zone_size[1] / 2, pts_local[1, :] < zone_size[1] / 2,
    #                     pts_local[2, :] > -zone_size[2] / 2, pts_local[2, :] < zone_size[2] / 2
    #                 ])

    #             zone_pts += np.sum(np.float32(valid_pts))
    #             total_pts += pts.shape[1]
        
    #     # Calculate current exact physical reward
    #     step_reward = max_reward * (zone_pts / total_pts) if total_pts > 0 else 0

    #     # Calculate delta reward (negative if it drifted!)
    #     reward = step_reward - getattr(self, '_rewards', 0)
    #     self._rewards = step_reward 

    #     # 4. The "Wake Up" Logic: Pop or Restore the goal based on the physical state
    #     is_packed = np.abs(max_reward - step_reward) < 0.05
        
    #     if is_packed and len(self.goals) > 0:
    #         # Everything is packed nicely. Pop the goal so the Oracle executes a no-op.
    #         self.progress += max_reward  
    #         self.goals.pop(0)
            
    #     elif not is_packed and len(self.goals) == 0:
    #         # A block drifted out! The task is no longer solved.
    #         # Restore the goal so the Oracle wakes up and fixes it.
    #         self.progress -= max_reward  # Revoke progress so we don't double-count later
    #         self.goals.append(self.full_goal)

    #     return reward, info
