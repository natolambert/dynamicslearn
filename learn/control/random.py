from .controller import Controller


class RandomController(Controller):
    def __init__(self, env, controller_cfg):
        self.env = env
        self.cfg = controller_cfg

    def reset(self):
        print("Resetting Random Controller Not Needed, but passed")
        return

    def get_action(self, state):
        action = self.env.action_space.sample()
        return action
