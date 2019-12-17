from .controller import Controller
import numpy as np

class RandomController(Controller):
    def __init__(self, env, controller_cfg):
        self.env = env
        self.cfg = controller_cfg

    def reset(self):
        print("Resetting Random Controller Not Needed, but passed")
        return

    def get_action(self, state):
        action = self.env.action_space.sample().astype(float)
        return action
