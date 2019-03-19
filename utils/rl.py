import numpy as np
import torch
import torch.optim as optim

import rlkit.rlkit.torch.pytorch_util as ptu
from rlkit.rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.rlkit.torch.sac.policies import MakeDeterministic
from rlkit.rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm

"""
Classic cart-pole system implemented by Rich Sutton et al. 
-> Modified by Nathan Lambert to be a continuous environment

Changes:
1. actions:
    self.action_space = spaces.box(low=np.array([-1.0]), high=np.array([1.0]))
2. forces:
    force = self.force_mag*action
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding

class CartPoleContEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: box([-1,1])
        Num	Action 
        -1	Push cart to the left max force
        1	Push cart to the right max force
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        # self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag*action
        # force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot *
                theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length *
                                                                  (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset = cartheight/4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


"""
NOTE: Adapted from Vitchyr Pong's rlkit implementation of SAC
need to implement a modified building of me-sac on top of RLKIT with the 
  policy improvement iteration stop
"""


class METwinSAC(TorchRLAlgorithm):
    """
    SAC with the twin architecture from TD3 running on a model ensemble.
      the ME part adds a training loop that looks for improvement across the models
    """

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vf,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            train_policy_with_reparameterization=True,
            soft_target_tau=1e-2,
            policy_update_period=1,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,

            eval_policy=None,
            exploration_policy=None,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            **kwargs
    ):
        if eval_policy is None:
            if eval_deterministic:
                eval_policy = MakeDeterministic(policy)
            else:
                eval_policy = policy
        super().__init__(
            env=env,
            exploration_policy=exploration_policy or policy,
            eval_policy=eval_policy,
            **kwargs
        )
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.soft_target_tau = soft_target_tau
        self.policy_update_period = policy_update_period
        self.target_update_period = target_update_period
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = - \
                    np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf = vf.copy()
        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        v_pred = self.vf(obs)
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(obs,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        Alpha Loss (if applicable)
        """
        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi +
                                             self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_v_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = torch.min(
            self.qf1(obs, new_actions),
            self.qf2(obs, new_actions),
        )
        v_target = q_new_actions - alpha*log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        policy_loss = None
        if self._n_train_steps_total % self.policy_update_period == 0:
            """
            Policy Loss
            """
            if self.train_policy_with_reparameterization:
                policy_loss = (alpha*log_pi - q_new_actions).mean()
            else:
                log_policy_target = q_new_actions - v_pred
                policy_loss = (
                    log_pi * (alpha*log_pi - log_policy_target).detach()
                ).mean()
            mean_reg_loss = self.policy_mean_reg_weight * \
                (policy_mean**2).mean()
            std_reg_loss = self.policy_std_reg_weight * \
                (policy_log_std**2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.vf, self.target_vf, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            if policy_loss is None:
                if self.train_policy_with_reparameterization:
                    policy_loss = (log_pi - q_new_actions).mean()
                else:
                    log_policy_target = q_new_actions - v_pred
                    policy_loss = (
                        log_pi * (log_pi - log_policy_target).detach()
                    ).mean()

                mean_reg_loss = self.policy_mean_reg_weight * \
                    (policy_mean**2).mean()
                std_reg_loss = self.policy_std_reg_weight * \
                    (policy_log_std**2).mean()
                pre_tanh_value = policy_outputs[-1]
                pre_activation_reg_loss = self.policy_pre_activation_weight * (
                    (pre_tanh_value**2).sum(dim=1).mean()
                )
                policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
                policy_loss = policy_loss + policy_reg_loss

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
        ]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot['qf1'] = self.qf1
        snapshot['qf2'] = self.qf2
        snapshot['policy'] = self.policy
        snapshot['vf'] = self.vf
        snapshot['target_vf'] = self.target_vf
        return snapshot


class QLearner():
	def __init__(self, dynamics_model, dynamics_data):
		self.batch_size = 10
		self.num_actions = 81
		self.state_size = 9
		self.reward_size = 1
		self.eps = 0.5
		# Make target Q and current Q networks
		self.target_q = nn.Sequential(nn.Linear(self.state_size, 64), nn.ReLU(
		), nn.Linear(64, 64), nn.Linear(64, self.num_actions))
		self.current_q = nn.Sequential(nn.Linear(self.state_size, 64), nn.ReLU(
		), nn.Linear(64, 64), nn.Linear(64, self.num_actions))
		self.buffer = []
		self.dyn_data = dynamics_data
		self.dyn_model = dynamics_model
		self.dyn_model.eval()
		self.action_dict = self.init_discretized_actions()
		self.gamma = 1
		self.fail_loss = 1000
		x = torch.randn(self.batch_size, self.state_size)
		self.criterion = nn.MSELoss(reduce=True)
		self.optimizer = torch.optim.SGD(self.current_q.parameters(), lr=0.01)
		# print(x)
		# print(self.target_q(x))
		self.initial_state = self.sample_random_state()

	def init_discretized_actions(self):
		"""
		The real action space has values [m1, m2, m3, m4, vbat]. To keep the action space
		discrete in our implementation, we allow each mi to take on only one of three values,
		and we keep the vbat constant. This function initializes an action_dict that maps
		an index (from 0 to 80) to its corresponding permutation in the real action space.
		"""

		motor_discretized = [[30000, 35000, 40000], [30000, 35000, 40000], [
		    30000, 35000, 40000], [30000, 35000, 40000]]
		action_dict = dict()
		for ac, action in enumerate(list(itertools.product(*motor_discretized))):
			action = action + (3757,)
			action_dict[ac] = np.asarray(action)
		return action_dict

	def get_next_state(self, s, a):
		"""
		We use our dynamics model (trained beforehand) to predict the next state
		given a state s and an action a. 
		Returns: resulting state [numpy array]
		"""
		return self.dyn_model.predict(s, a) + s

	def state_failed(self, s):
		"""
		Check whether a state has failed, so the trajectory should terminate.
		This happens when either the roll or pitch angle > 30
		Returns: failure flag [bool]
		"""
		if abs(s[3]) > 30.0 or abs(s[4]) > 30.0:
			return True

	def get_reward_state(self, s_next):
		"""
		Returns reward of being in a certain state.
		"""
		pitch = s_next[3]
		roll = s_next[4]
		if self.state_failed(s_next):
			return -1 * self.fail_loss
		loss = pitch**2 + roll**2
		# print(loss)
		# print(pitch, roll)
		reward = 100 - loss  # This should be positive. TODO: Double check
		return reward

	def sample_random_state(self):
		"""
		Samples random state from previous logs. Ensures that the sampled
		state is not in a failed state.
		"""
		state_failed = True
		while state_failed:
			row_idx = np.random.randint(self.dyn_data.shape[0])
			random_state = self.dyn_data.iloc[row_idx, 12:21].values
			state_failed = self.state_failed(random_state)
		return random_state

	def sample_random_action(self):
		"""
		Samples a completely random action.
		"""
		ac = np.random.randint(81)  # Returns an integer from 0 to 80
		a = self.action_dict[ac]
		return a, ac

	def add_transition_to_buffer(self, s, ac):
		"""
		Adds a transition to the buffer based on s and a. No explicit passing in of s_next
		Returns: transition [Transition] containing the s_next.
		"""
		a = self.action_dict[ac]
		s_next = self.get_next_state(s, a)
		r = self.get_reward_state(s_next)
		r = np.atleast_1d(r)
		new_transition = Transition(s=torch.from_numpy(s), a=torch.from_numpy(a),
                              a_index=ac, s_next=torch.from_numpy(s_next), r=torch.from_numpy(r))
		self.buffer.append(new_transition)
		return new_transition

	def sample_mini_batch(self, size):
		"""
		Samples a mini_batch of size=size from the reply buffer.
		Returns: mini_batch
		"""
		buffer_length = len(self.buffer)
		mini_batch = np.random.choice(self.buffer, size, p=np.repeat(
		    1.0 / buffer_length, buffer_length), replace=False)
		return mini_batch

	def mini_batch_to_stacked(self, mini_batch):
		"""
		Takes a mini-batch of transitions and stacks each s, a, s_next, and r as tensors
		"""
		b_states = torch.cat([t.s.unsqueeze(0) for t in mini_batch], dim=0)
		b_actions = torch.cat([t.a.unsqueeze(0) for t in mini_batch], dim=0)
		b_action_indices = torch.LongTensor([t.a_index for t in mini_batch])
		b_next_states = torch.cat([t.s_next.unsqueeze(0) for t in mini_batch], dim=0)
		b_rewards = torch.cat([t.r for t in mini_batch], dim=0)

		return b_states, b_actions, b_action_indices, b_next_states, b_rewards

	def compute_y(self, mini_batch):
		"""  
		Computes the targets for training from the target Q network
		"""
		b_states, b_actions, b_action_indices, b_next_states, b_rewards = self.mini_batch_to_stacked(
		    mini_batch)
		q = self.target_q(b_next_states.float()).detach()
		max_a = torch.max(q, dim=1)[0]
		max_a = max_a.unsqueeze(1)
		b_y = b_rewards.unsqueeze(1).float() + self.gamma * max_a.float()
		return b_y

	def gradient_step(self, mini_batch):
		"""
		Takes one gradient step in training the Q network
		"""
		b_states, b_actions, b_action_indices, b_next_states, b_rewards = self.mini_batch_to_stacked(
		    mini_batch)

		# Get y
		y = self.compute_y(mini_batch)
		curr_q = self.current_q(b_states.float()).gather(
		    1, b_action_indices.view(-1, 1))
		# loss = self.criterion(curr_q, y) # y is the target
		loss = nn.functional.smooth_l1_loss(curr_q, y)

		# print("Difference", curr_q[0], y[0], curr_q[0] - y[0])
		# print(curr_q.size(), y.size())
		self.optimizer.zero_grad()  # Clear previous gradients
		loss.backward()
		self.optimizer.step()
		return loss

	def save_target_parameters(self):
		"""
		Updates the target Q network parameters with those of the current Q network
		"""
		self.target_q = copy.deepcopy(self.current_q)

	def get_best_action(self, state):
		"""
		Chooses best action from state based on the current Q network policy
		"""
		q = self.current_q(torch.from_numpy(state).float())
		best_action_index = torch.max(q, dim=0)[1].item()
		best_action = self.action_dict[best_action_index]
		return best_action, best_action_index

	def evaluate(self, horizon, rollouts):
		# Do a bunch of random rollouts and calculate the mean loss
		total_reward = 0
		for r in range(rollouts):
			s = self.sample_random_state()
			s = self.initial_state
			rollout_reward = 0

			for h in range(horizon):
				# Get best action
				# best_action, best_ac = self.get_best_action(s)
				best_action, best_ac = self.sample_random_action()
				s_next = self.get_next_state(s, best_action)
				rollout_reward += self.get_reward_state(s_next)
				s = s_next
				if self.state_failed(s_next):
					print("Failed", h)
					break
			total_reward += rollout_reward / horizon  # Average reward over steps
		return total_reward / rollouts

	def train(self):
		mini_batch_size = 100
		n = 5000  # Number of times to run before upddating target q
		num_target_updates = 100
		# how many times we want to update target_q * n
		total_iterations = n * num_target_updates

		# First take a bunch of random actions so B has at least mini_batch_size samples
		for i in range(mini_batch_size):
			sampled_state = self.sample_random_state()
			sampled_action, ac = self.sample_random_action()
			self.add_transition_to_buffer(sampled_state, ac)

		counter = 0
		x = []
		r = []
		losses = []
		state = self.sample_random_state()
		traj_len = 0

		while counter < total_iterations:

			# Take action and add to B (epsilon greedy)
			p = np.random.uniform(0, 1)
			if p > self.eps:
				best_action, best_ac = self.get_best_action(state)
			else:
				best_action, best_ac = self.sample_random_action()

			transition = self.add_transition_to_buffer(state, best_ac)
			traj_len += 1

			# If this resulted in a failed trajectory, initialize new state
			if self.state_failed(transition.s_next):
				state = self.sample_random_state()
				state = self.initial_state
				print("Traj len", traj_len)
				traj_len = 0

				# state = self.initial_state

			# Sample minibatch from B uniformly
			mini_batch = self.sample_mini_batch(mini_batch_size)

			# Do one gradient step
			loss = self.gradient_step(mini_batch)

			if counter % n == 0:
				# Update target parameters
				print("\n == Updating target == ", counter/n, "/", num_target_updates)
				self.save_target_parameters()
				avg_reward = self.evaluate(100, 1)
				x.append(counter / n)
				r.append(avg_reward)
				print("Average reward", avg_reward)
				state = self.sample_random_state()
				state = self.initial_state
				print("Training loss", loss.item())
				losses.append(loss)

			counter += 1

		# Plot returns
		plt.plot(x, r, label="Evaluated rewards")
		plt.plot(x, losses, label="Training losses")
		plt.legend()
		plt.show()
