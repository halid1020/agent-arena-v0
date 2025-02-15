import numpy as np
import cv2
from matplotlib import pyplot as plt
from policies.mpc_cem import MPC_CEM


### The dynamic model must provide unroll_action, init_state and update_state functions
### Cost function should be choose.

def L2_cost(trajs, goal_image):
    final_image = trajs[:, -1]
    goal_image = np.expand_dims(goal_image, axis=0)

    costs = np.sum(np.abs(final_image - goal_image), axis=(1,2,3))

    # ### Get the index of lowest cost and plot the corresponding final image
    # min_cost_idx = np.argmin(costs)
    # plt.imshow(final_image[min_cost_idx])
    # plt.show()
    # ## print the range of the image
    # print('min and max of final image', np.min(final_image[min_cost_idx]), np.max(final_image[min_cost_idx]))
    # print('min and max of goal image', np.min(goal_image), np.max(goal_image))

    ### Get the index of highest cost and plot the corresponding final image
    # max_cost_idx = np.argmax(costs)
    # plt.imshow(final_image[max_cost_idx])
    # plt.show()
    return costs

class VisualMPC_CEM(MPC_CEM):

    def __init__(self, model, cost_fn, **kwargs):
        super().__init__(model, cost_fn, **kwargs)
        
        self.image_dim = kwargs['image_dim']

    def initialise_cost_fn(self, cost_fn, **kwargs):
        if cost_fn == 'L2':
            self.cost_fn = L2_cost
        else:
            raise NotImplementedError

    def set_goal_image(self, goal_image):
        self.goal_image = cv2.resize(goal_image, self.image_dim)
    
    
    def _predict_and_eval(self, actions):

        actions = actions.reshape(self.candidates, self.planning_horizon, *self.action_dim)
        pred_trajs = self.model.unroll_action_from_cur_state(actions)
        
        images = self.model.visual_reconstruct(pred_trajs)

        print('imaes shape', images.shape)

        ## Resize the images to the same size as goal image
        images = np.asarray([cv2.resize(image, self.image_dim) for image in images.reshape(-1, *images.shape[2:])])\
            .reshape(self.candidates, self.planning_horizon, *self.image_dim, 3)

        costs = self.cost_fn(images, self.goal_image)
        
        return np.array(costs), pred_trajs
    