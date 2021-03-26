from .controller import Controller
import torch


class MPController(Controller):
    def __init__(self, env, model, controller_cfg):
        super(MPController, self).__init__(controller_cfg)
        controller_cfg = controller_cfg[controller_cfg.policy.mode]
        self.env = env
        self.model = model
        if torch.cuda.is_available(): self.model.cuda()
        self.cfg = controller_cfg
        self.N = self.cfg.params.N
        self.T = self.cfg.params.T
        self.hold = self.cfg.params.hold

        self.yaw = controller_cfg.params.mode == 'yaw'

        self.low = torch.tensor(self.env.action_space.low, dtype=torch.float32)
        self.high = torch.tensor(self.env.action_space.high, dtype=torch.float32)
        self.action_sampler = torch.distributions.Uniform(low=self.low, high=self.high)

        self.internal = 0
        self.yaw_actions = torch.tensor([
            [1500.0, 1500.0, 1500.0, 1500.0],
            [2000, 1000, 1000, 2000],
            [2000, 2000, 1000, 1000],
            [1000, 2000, 2000, 1000],
            [1000, 1000, 2000, 2000],
        ])

        self.yaw_actions = torch.tensor([
            [1500.0, 1500.0, 1500.0, 1500.0],

            # [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],
            [2000, 1000, 1000, 2000],

            [2000, 2000, 1000, 1000],
            [2000, 2000, 1000, 1000],
            [2000, 2000, 1000, 1000],
            [2000, 2000, 1000, 1000],
            [2000, 2000, 1000, 1000],
            [2000, 2000, 1000, 1000],
            [2000, 2000, 1000, 1000],
            [2000, 2000, 1000, 1000],
            # [1000, 2000, 2000, 1000],

            [1000, 2050, 2050, 1000],
            [1000, 2050, 2050, 1000],
            [1000, 2050, 2050, 1000],
            [1000, 2050, 2050, 1000],
            [1000, 2050, 2050, 1000],
            [1000, 2050, 2050, 1000],
            [1000, 2050, 2050, 1000],
            [1000, 2050, 2050, 1000],

            [1000, 1000, 2050, 2050],
            [1000, 1000, 2050, 2050],
            [1000, 1000, 2050, 2050],
            [1000, 1000, 2050, 2050],
            [1000, 1000, 2050, 2050],
            [1000, 1000, 2050, 2050],
            [1000, 1000, 2050, 2050],
            [1000, 1000, 2050, 2050],
        ])

        # self.yaw_actions = torch.tensor([
        #     [1500.0, 1500.0, 1500.0, 1500.0],
        #     [2000, 1000, 1000, 2000],
        #     [2000, 1000, 1000, 2000],
        #     [2000, 1000, 1000, 2000],
        #     [2000, 1000, 1000, 2000],
        #     [2000, 1000, 1000, 2000],
        #     [2000, 1000, 1000, 2000],
        #     [2000, 2000, 1000, 1000],
        #     [2000, 2000, 1000, 1000],
        #     [2000, 2000, 1000, 1000],
        #     [2000, 2000, 1000, 1000],
        #     [2000, 2000, 1000, 1000],
        #     [2000, 2000, 1000, 1000],
        #     [1000, 2000, 2000, 1000],
        #     [1000, 2000, 2000, 1000],
        #     [1000, 2000, 2000, 1000],
        #     [1000, 2000, 2000, 1000],
        #     [1000, 2000, 2000, 1000],
        #     [1000, 2000, 2000, 1000],
        #     [1000, 1000, 2000, 2000],
        #     [1000, 1000, 2000, 2000],
        #     [1000, 1000, 2000, 2000],
        #     [1000, 1000, 2000, 2000],
        #     [1000, 1000, 2000, 2000],
        #     [1000, 1000, 2000, 2000],
        # ])

        # self.yaw_actions = torch.tensor([
        #     [900, 900, 900, 900],
        #     [1350, 450, 450, 1350],
        #     [1350, 450, 450, 1350],
        #     [1350, 450, 450, 1350],
        #     [1350, 450, 450, 1350],
        #     [450, 1350, 1350, 450],
        #     [450, 1350, 1350, 450],
        #     [450, 1350, 1350, 450],
        #     [450, 1350, 1350, 450],
        #     [1350, 1350, 450, 450],
        #     [1350, 1350, 450, 450],
        #     [1350, 1350, 450, 450],
        #     [1350, 1350, 450, 450],
        #     [450, 450, 1350, 1350],
        #     [450, 450, 1350, 1350],
        #     [450, 450, 1350, 1350],
        #     [450, 450, 1350, 1350],
        # ])
        # self.yaw_actions = torch.tensor([
        #     [900, 900, 900, 900],
        #     [3000, 0, 0, 3000],
        #     [3000, 0, 0, 3000],
        #     [3000, 0, 0, 3000],
        #     [3000, 0, 0, 3000],
        #     [0, 3000, 3000, 0],
        #     [0, 3000, 3000, 0],
        #     [0, 3000, 3000, 0],
        #     [0, 3000, 3000, 0],
        #     [3000, 3000, 0, 0],
        #     [3000, 3000, 0, 0],
        #     [3000, 3000, 0, 0],
        #     [3000, 3000, 0, 0],
        #     [0, 0, 3000, 3000],
        #     [0, 0, 3000, 3000],
        #     [0, 0, 3000, 3000],
        #     [0, 0, 3000, 3000],
        # ])

    def reset(self):
        self.interal = 0
        print("Resetting MPController Not Needed, but passed")
        return

    def get_action(self, state, metric=None):
        # if self.internal % self.update_period == 0:
        # repeat the state
        state0 = torch.tensor(state).repeat(self.N, 1)
        states = torch.zeros(self.N, self.T + 1, state.shape[-1])
        states[:, 0, :] = state0
        rewards = torch.zeros(self.N, self.T)
        if self.yaw:

            action = self.yaw_actions[self.internal + 1]
            self.internal += 1
            self.internal %= 32
            return action

            # # constrained yaw action below
            # action_idx = torch.randint(0, 5, (self.N, self.T), dtype=torch.long)
            # action_candidates = torch.zeros((self.N, self.T, len(self.low)))
            # for i in range(action_candidates.shape[0]):
            #     act = []
            #     for t in range(self.T):
            #         act.append(self.yaw_actions[action_idx[i, t]])
            #     action_candidates[i, :, :] = torch.stack(act)

        else:
            if self.hold:
                action_candidates = self.action_sampler.sample(sample_shape=(self.N, 1)).repeat(1, self.T, 1)
            else:
                action_candidates = self.action_sampler.sample(sample_shape=(self.N, self.T))
        if False: #torch.cuda.is_available:
            states.cuda()
            rewards.cuda()
            action_candidates.cuda()
        # TODO
        for t in range(self.T):
            action_batch = action_candidates[:, t, :]
            state_batch = states[:, t, :]
            # if torch.cuda.is_available():
            #     state_batch.cuda()
            # next_state_batch = state_batch + torch.tensor(self.model.predict(state_batch, action_batch)).float()
            next_state_batch = state_batch + self.model.predict(state_batch, action_batch)
            states[:, t + 1, :] = next_state_batch
            if metric is not None:
                rewards[:, t] = metric(next_state_batch, action_batch)
            else:
                rewards[:, t] = self.env.get_reward_torch(next_state_batch, action_batch)

        # TODO compute rewards
        cumulative_reward = torch.sum(rewards, dim=1)

        if False:
            # Basic waterfall plot
            import plotly.graph_objects as go
            fig = go.Figure()
            # Create and style traces
            for i, vec in enumerate(states[:, :, 0]):
                if i < 500:
                    fig.add_trace(go.Scatter(y=vec))
            fig.show()

        best = torch.argmax(cumulative_reward)
        actions_seq = action_candidates[best, :, :]
        best_action = actions_seq[0]

        self.last_action = best_action
        self.internal += 1
        return best_action
        # else:
        #     self.internal += 1
        #     best_action = self.last_action
        #     return best_action, False
