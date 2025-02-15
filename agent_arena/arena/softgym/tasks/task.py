import os

from agent_arena.arena.task import Task
import matplotlib.pyplot as plt
import numpy as np

class Task(Task):

    
    def evaluate(self, arena):
        raise NotImplementedError
    
    def reward(self, arena):
        raise 0
    
    def _save_goal(self, arena):
        eid = arena.get_episode_id()
        mode = arena.get_mode()
        task = self.task_name
        arena_name = arena.get_name()
        if not os.path.exists(self._get_goal_path(arena_name, task, eid, mode)):
            os.makedirs(self._get_goal_path(arena_name, task, eid, mode))
        plt.imsave(self._get_goal_path(arena_name, task, eid, mode) + '/rgb.png', self.goal['rgb'])
        np.save(self._get_goal_path(arena_name, task, eid, mode) + '/depth.npy', self.goal['depth'])

    def _get_goal_path(self, arena, task, eid, mode):
        
        goal_dir = f'{os.environ["AGENT_ARENA_PATH"]}/../data/goals/{arena}/{task}/{mode}/eid_{eid}'
        return goal_dir
    
        # if self.domain == 'mono-square-fabric':
        #     return '{}/../task_wrappers/rect_fabric/goals/{}/{}'\
        #         .format(os.environ['SOFTGYM_PATH'], self.task_name, self.domain)
        
        # return '{}/../task_wrappers/rect_fabric/goals/{}/{}/initial_{}/{}_eid_{}'\
        #     .format(os.environ['SOFTGYM_PATH'], self.task_name, self.domain, self.initial, mode, eid)