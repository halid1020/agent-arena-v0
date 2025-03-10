# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Sweeping task."""

import numpy as np
from ...raven.tasks import primitives
from ...raven.tasks.grippers import Spatula
from ...raven.tasks.task import Task
from agent_arena.agent.bc.transporter.utils import utils


class SweepingPiles(Task):
    """Sweeping task."""

    def __init__(self):
        super().__init__()
        self.ee = Spatula
        self.max_steps = 20
        self.primitive = primitives.Push()

    def reset(self, env):
        super().reset(env)
        self._add_instance(env)

    def _add_instance(self, env):
        # Add goal zone.
        zone_size = (0.12, 0.12, 0)
        zone_pose = self.get_random_pose(env, zone_size)
        env.add_object('zone/zone.urdf', zone_pose, 'fixed')

        # Add pile of small blocks.
        obj_pts = {}
        obj_ids = []
        for _ in range(50):
            rx = self.bounds[0, 0] + 0.15 + np.random.rand() * 0.2
            ry = self.bounds[1, 0] + 0.4 + np.random.rand() * 0.2
            xyz = (rx, ry, 0.01)
            theta = np.random.rand() * 2 * np.pi
            xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            obj_id = env.add_object('block/small.urdf', (xyz, xyzw))
            obj_pts[obj_id] = self.get_object_points(obj_id)
            obj_ids.append((obj_id, (0, None)))

        # Goal: all small blocks must be in zone.
        # goal = Goal(list(obj_pts.keys()), [0] * len(obj_pts), [zone_pose])
        # metric = Metric('zone', (obj_pts, [(zone_pose, zone_size)]), 1.)
        # self.goals.append((goal, metric))
        self.goals.append((obj_ids, np.ones((50, 1)), [zone_pose], True, False,
                           'zone', (obj_pts, [(zone_pose, zone_size)]), 1))
