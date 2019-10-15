

class Controller:
    def reset(self):
        raise NotImplementedError("Subclass must implement this function")

    def get_action(self, state):
        raise NotImplementedError("Subclass must implement this function")

