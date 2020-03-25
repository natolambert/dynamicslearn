

class Controller:
    def __init__(self, cfg):
        self.internal = 0
        # self.update_period = cfg.policy.params.period
        self.last_action = None

    def reset(self):
        raise NotImplementedError("Subclass must implement this function")

    def get_action(self, state):
        raise NotImplementedError("Subclass must implement this function")

