from agent_arena import Agent
import numpy as np
import cv2


from agent_arena.utilities.logger.logger_interface import Logger

class PixelHumanOnePicker(Agent):
    
        def __init__(self, config):
            super().__init__(config)
            #self.config = config
            self.name = "human-pixel-pick-and-place"
            self.logger = Logger()

        def act(self, state, update=False):
            """
            Pop up a window shows the RGB image, and user can click on the image to
            produce normalised pick-and-place action ranges from [-1, 1]
            """
            state = state[0]
            rgb = state['observation']['rgb']
            goal_rgb = state['goal']['rgb']

            if 'workspace_mask' in state['observation']:
                workspace = state['observation']['workspace_mask'].reshape(*rgb.shape[:2], -1)
                alpha = 0.5
                rgb = alpha * (1.0*rgb/255.0) + (1-alpha) * workspace.astype(np.float32)
                rgb = (rgb * 255).astype(np.uint8)

                goal_rgb = alpha * (1.0*goal_rgb/255.0) + (1-alpha) * workspace.astype(np.float32)
                goal_rgb = (goal_rgb * 255).astype(np.uint8)
                

            ## make it bgr to rgb using cv2
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            goal_rgb = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2RGB)

            ## resize shorter side to 512
            # H, W = rgb.shape[:2]
            # if H < W:
            #     rgb = cv2.resize(rgb, (int(W/H*512), 512))
            #     goal_rgb = cv2.resize(goal_rgb, (int(W/H*512), 512))
            # else:
            #     rgb = cv2.resize(rgb, (512, int(H/W*512)))
            #     goal_rgb = cv2.resize(goal_rgb, (512, int(H/W*512)))
            
            # put img and goal_img side by side
            img = np.concatenate([rgb, goal_rgb], axis=1)
            
            # Store click coordinates
            clicks = []
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicks.append((x, y))
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                    cv2.imshow('Click Pick and Place Points', img)
            
            cv2.imshow('Click Pick and Place Points', img)
            cv2.setMouseCallback('Click Pick and Place Points', mouse_callback)
            
            while len(clicks) < 2:
                cv2.waitKey(1)
            
            cv2.destroyAllWindows()
            
            # Normalize the coordinates to [-1, 1]
            height, width = rgb.shape[:2]
            pick_y, pick_x = clicks[0]
            place_y, place_x = clicks[1]

            print(f"Pick: ({pick_x}, {pick_y}), Place: ({place_x}, {place_y})")

            print(f"Obs Height: {height}, Width: {width}")
            
            normalized_action = np.array([
                (pick_x / height) * 2 - 1,
                (pick_y / width) * 2 - 1,
                (place_x / height) * 2 - 1,
                (place_y / width) * 2 - 1
            ])
            
            action = {
                'pick_0': normalized_action[:2],
                'place_0': normalized_action[2:],
               
            }
            action['norm-pixel-pick-and-place'] = action.copy()
            return [action]
            
            
        def get_phase(self):
            return "default"
        

        def reset(self, arena_ids):
            self.internal_states = {
                arena_id: [] for arena_id in arena_ids
            }
        
        def init(self, state):
            pass
        
        def update(self, state, action):
            pass
    