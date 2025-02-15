from agent_arena import Agent
import numpy as np
import cv2

class PixelHumanTwoPicker(Agent):
    
    def __init__(self):
        self.name = "human-pixel-pick-and-place-two"

    def act(self, state):
        """
        Pop up a window shows the RGB image, and user can click on the image to
        produce normalised pick-and-place actions for two objects, ranges from [-1, 1]
        """
        rgb = state['observation']['rgb']
        goal_rgb = state['goal']['rgb']

        ## make it bgr to rgb using cv2
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        goal_rgb = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2RGB)

        ## resize
        rgb = cv2.resize(rgb, (512, 512))
        goal_rgb = cv2.resize(goal_rgb, (512, 512))
        
        # Create a copy of the image to draw on
        img = rgb.copy()

        # put img and goal_img side by side
        img = np.concatenate([img, goal_rgb], axis=1)
        
        # Store click coordinates
        clicks = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                if len(clicks) % 2 == 1:  # Pick action (odd clicks)
                    color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
                    cv2.circle(img, (x, y), 5, color, -1)
                else:  # Place action (even clicks)
                    color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
                    cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                cv2.imshow('Click Pick and Place Points (4 clicks needed)', img)
        
        cv2.imshow('Click Pick and Place Points (4 clicks needed)', img)
        cv2.setMouseCallback('Click Pick and Place Points (4 clicks needed)', mouse_callback)
        
        while len(clicks) < 4:
            cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        
        # Normalize the coordinates to [-1, 1]
        height, width = rgb.shape[:2]
        pick1_x, pick1_y = clicks[0]
        place1_x, place1_y = clicks[1]
        pick2_x, pick2_y = clicks[2]
        place2_x, place2_y = clicks[3]
        
        normalized_action1 = [
            (pick1_x / width) * 2 - 1,
            (pick1_y / height) * 2 - 1,
            (place1_x / width) * 2 - 1,
            (place1_y / height) * 2 - 1
        ]
        
        normalized_action2 = [
            (pick2_x / width) * 2 - 1,
            (pick2_y / height) * 2 - 1,
            (place2_x / width) * 2 - 1,
            (place2_y / height) * 2 - 1
        ]
        
        return {
            'pick_0': normalized_action1[:2],
            'place_0': normalized_action1[2:],
            'pick_1': normalized_action2[:2],
            'place_1': normalized_action2[2:],
        }
        
    def get_phase(self):
            return "default"

    def get_state(self):
        return {}
    
    def success(self):
        return False

    def terminate(self):
        return False
    

    def _reset(self):
        pass
    
    def init(self, state):
        pass
    
    def update(self, state, action):
        pass