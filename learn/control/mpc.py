from .controller import Controller
import torch


class MPController(Controller):
    def __init__(self, env, model, controller_cfg):
        self.env = env
        self.model = model
        self.cfg = controller_cfg
        self.N = self.cfg.params.N
        self.T = self.cfg.params.T
        self.hold = self.cfg.params.hold

        self.low = torch.tensor(self.env.action_space.low)
        self.high = torch.tensor(self.env.action_space.high)
        self.action_sampler = torch.distributions.Uniform(low=self.low, high=self.high)

    def reset(self):
        print("Resetting MPController Not Needed, but passed")
        return

    def get_action(self, state):
        # repeat the state
        state = state.repeat(1,self.N)
        rewards = torch.zeros_like(state)
        if self.hold:
            action_candidates = self.action_sampler.sample(shape=(self.N, 1)).repeat(1, self.T)
        else:
            action_candidates = self.action_sampler.sample(shape=(self.N, self.T))

        # TODO
        for t in range(self.T):
            next_state = 1
        return 0
