import os
import numpy as np
import matplotlib.pyplot as plt
import logging


from arena.softgym.task_wrappers.rect_fabric.folding_wrapper \
    import FoldingWrapper
import api as ag_ar


class OneStepFoldingWrapper(FoldingWrapper):
    """
        Pick a random coner and put the corner on any point in the area, where the area is defined as the
        arc of the circle with the center at the corner and the radius is the length of the 1.5 time diagonal of the fabric.
        This environment must be a goal-condition environment.
        We also save the goal partciels, and the pick-and-place action for generating the goal.

    """

    def __init__(self, env, canonical=False, 
                 domain='mono-square-fabric', 
                 initial='canonical', action='pixel-pick-and-place(1)'):
        super().__init__(env, canonical)
        self.env = env
        self.domain = domain
        self.initial = initial
        self.task_name = 'one-step-folding'
        #print('initial', initial)
        self.oracle_policy = ag_ar.build_agent(
                'oracle-rect-fabric|action:{},task:one-step-folding,strategy:expert'.format(action)
            )
        self.action = action
        # assert initial in ['flatten', 'flatten-v1' 'canonical'], \
        # 'The initial state of the one-step-folding task must be flatten, flatten-v1, or canonial.'
        
    def reset(self, episode_config=None):
        info_ = self.env.reset(episode_config)
        episode_config = self.env.get_episode_config()

        # ## TODO: debug this.
        self.load_goals(self.env.get_episode_id(), self.env.get_mode())

        info_ = self.env.reset(episode_config)

        return self._process_info(info_)
    
    def evaluate(self,
            metrics = [
                'mean_particle_distance', 
                'largest_particle_distance', 
                'canonical_IoU',
                'success']
        ):

        ### go over metrics and compute them
        results = {}
        for metric in metrics:
            if metric == 'mean_particle_distance':
                if self.goals is not None:
                    results[metric] = self._mean_particle_distance()
            elif metric == 'largest_particle_distance':
                if self.goals is not None:
                    results[metric] = self._largest_particle_distance()
            elif metric == 'canonical_IoU':
                if self.goals is not None:
                    results[metric] = self._get_canonical_IoU()
            elif metric == 'success':
                results[metric] = self.success()
            else:
                raise NotImplementedError


        return results
    
    # def _generate_goals(self):
    #     logging.info('[softgym, one-step folding, load gaols] generate goal')
    #     episode_config = self.env.get_episode_config()
    #     attempts = 0

    #     while True:
    #         info = self.env.reset(episode_config)
    #         #episode_config = self.env.get_episode_config()
    #         info = self._process_info(info)
    #         self.oracle_policy.reset()
    #         self.oracle_policy.init(info)

    #         actions = []
        
    #         #print('info largest_particle_distance {}'.format(info['largest_particle_distance']))
    #         info['seed'] = episode_config['eid'] + attempts*1000
    #         action = self.oracle_policy.act(info, self)
    #         actions.append(action)
    #         #print('action shape', action.shape)
    #         info = self.step(action)
    #         self.oracle_policy.update(info, action)
            
    #         particles = self.get_particle_positions()

    #         info = self.env.reset(episode_config)
    #         self.oracle_policy.reset(info)
    #         print('goal actions', actions)
    #         for action in actions:
    #             info = self.step(action)
    #             self.oracle_policy.update(info, action)
            
    #         replicate_particles = self.get_particle_positions()

    #         lpd = np.max(np.linalg.norm(particles - replicate_particles, axis=1))
    #         if lpd < 0.005:
    #             break

    #         attempts += 1
            

    #     self.goal = info['observation']
    #     self.goal['action'] = np.array(actions)
    #     self.goal['particle'] = self.get_particle_positions()

    
    # def _get_goal_path(self, eid, mode):

    #     """
    #         Ovewrite from the TaskWrapper class of SoftGym, as all domain we follow the following structure
    #         for saving the goals.
    #     """
        
    #     return '{}/../task_wrappers/rect_fabric/goals/{}/{}/initial_{}/{}_eid_{}'\
    #         .format(os.environ['SOFTGYM_PATH'], self.task_name, self.domain, self.initial, mode, eid)
    
    # def _load_goal(self, eid, mode):
        
    #     """
    #         Ovewrite from the TaskWrapper class of SoftGym to include the action and the particle.
    #     """
    #     print('load eid', eid)
    #     print('load mode', mode)

    #     goal = super()._load_goal(eid, mode)
    #     goal['action'] = np.load(self._get_goal_path(eid, mode) + '/action.npy')
    #     goal['particle'] = np.load(self._get_goal_path(eid, mode) + '/particle.npy')
        
        
    #     return goal
    
    # def _save_goal(self):
    #     """
    #         Save goals, rgb, depth, and the action and the particle.
    #     """
    #     eid = self.env.get_episode_id()
    #     mode = self.env.get_mode()
       
        
    #     if not os.path.exists(self._get_goal_path(eid, mode)):
    #         os.makedirs(self._get_goal_path(eid, mode))
    #     plt.imsave(self._get_goal_path(eid, mode) + '/rgb.png', self.goal['rgb'])
    #     np.save(self._get_goal_path(eid, mode) + '/depth.npy', self.goal['depth'])
    #     np.save(self._get_goal_path(eid, mode) + '/mask.npy', self.goal['mask'])
    #     np.save(self._get_goal_path(eid, mode) + '/action.npy', self.goal['action'])
    #     np.save(self._get_goal_path(eid, mode) + '/particle.npy', self.goal['particle'])
    

    def success(self):
        """
            Check if the largest partcile distance is smaller than a value.
            Check if the IoU between the current state and the goal is larger than a value.
        """
        is_success = self._largest_particle_distance() < 0.005
        if self.canonical:
            is_success = is_success and self._get_canonical_IoU() >= 0.7
        
        return is_success
    
    def _get_particle_distances(self):
        """
            Return the distance between the goal particles and the current particles in meters.
        """

        ## Particles are in the form of N*(x, z, y)
        cur_particles = self.env.get_particle_positions()
        goal_particles = self.goals[-1]['particle']

        # Get the particle wise distance
        return np.linalg.norm(cur_particles - goal_particles, axis=1)

    def _largest_particle_distance(self):
        """
            Compare the goal particles and the current particles, and find the largest distance.
        """
        if self.goals is None:
            return np.nan
    
        return np.max(self._get_particle_distances())
    
    def _mean_particle_distance(self):
        """
            Compare the goal particles and the current particles, and find the mean distance.
        """
        if self.goals is None:
            return np.nan
        
        return np.mean(self._get_particle_distances())