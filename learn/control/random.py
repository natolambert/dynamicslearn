from .controller import Controller


class RandomController(Controller):
    def __init__(self, env, controller_cfg):
        self.env = env
        self.cfg = controller_cfg

    def reset(self):
        raise NotImplementedError("TODO")

    def get_action(self, state):
        raise NotImplementedError("TODO")
        return action
