from .controller import Controller
import torch


class MPController(Controller):
    def __init__(self, env, model, controller_cfg):
        super(MPController, self).__init__(controller_cfg)
        controller_cfg = controller_cfg[controller_cfg.policy.mode]
        self.env = env
        self.model = model
        self.cfg = controller_cfg
        self.N = self.cfg.params.N
        self.T = self.cfg.params.T
        self.hold = self.cfg.params.hold

        self.yaw = controller_cfg.params.mode

        self.low = torch.tensor(self.env.action_space.low, dtype=torch.float32)
        self.high = torch.tensor(self.env.action_space.high, dtype=torch.float32)
        self.action_sampler = torch.distributions.Uniform(low=self.low, high=self.high)

        self.yaw_actions = torch.tensor([
            [1500, 1500, 1500, 1500],
            [2000, 1000, 1000, 2000],
            [1000, 2000, 2000, 1000],
            [2000, 2000, 1000, 1000],
            [1000, 1000, 2000, 2000],
        ])
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

            action_idx = torch.randint(0,5,(self.N, self.T),dtype=torch.long)
            action_candidates = torch.zeros((self.N, self.T,len(self.low)))
            for i in range(action_candidates.shape[0]):
                act = []
                for t in range(self.T):
                    act.append(self.yaw_actions[action_idx[i,t]])
                action_candidates[i,:,:] = torch.stack(act)
        else:
            if self.hold:
                action_candidates = self.action_sampler.sample(sample_shape=(self.N, 1)).repeat(1, self.T, 1)
            else:
                action_candidates = self.action_sampler.sample(sample_shape=(self.N, self.T))

        # TODO
        for t in range(self.T):
            action_batch = action_candidates[:, t, :]
            state_batch = states[:, t, :]
            next_state_batch = state_batch + torch.tensor(self.model.predict(state_batch, action_batch)).float()
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
            for i, vec in enumerate(states[:,:,0]):
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
