from .controller import Controller
import numpy as np

class RandomController(Controller):
    def __init__(self, env, controller_cfg):
        super(RandomController, self).__init__(controller_cfg)
        self.env = env
        self.cfg = controller_cfg

    def reset(self):
        self.interal = 0
        print("Resetting Random Controller Not Needed, but passed")
        return

    def get_action(self, state):
        action = self.env.action_space.sample().astype(float)
        if self.internal % self.update_period == 0:
            return action, True
        else:
            return action, False
