from agent_arena import Agent
import numpy as np
import cv2

from .pick_and_place.pixel_human_two_picker import PixelHumanTwoPicker
from .pick_and_fling.pixel_human import PixelHuman

class PixelMultiPrimitive(Agent):
    
    def __init__(self):
        self.name = "human-pixel-multi-primitive"
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-pick-and-place"
        ]
        self.primitive_instances = [
            PixelHuman(),
            PixelHumanTwoPicker()]

    def act(self, state):
        """
        Allow user to choose a primitive, then delegate to the chosen primitive's act method.
        Keeps asking until a valid choice is made.
        """
        while True:
            print("Choose a primitive:")
            for i, primitive in enumerate(self.primitive_names):
                print(f"{i + 1}. {primitive}")
        
            try:
                choice = int(input("Enter the number of your choice: ")) - 1
                if 0 <= choice < len(self.primitive_names):
                    chosen_primitive = self.primitive_names[choice]
                    self.current_primitive = self.primitive_instances[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        action = self.current_primitive.act(state)

        return {
            chosen_primitive: action
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
