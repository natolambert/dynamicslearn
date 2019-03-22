import torch
import gym
from gym.spaces import Box
import pickle
import numpy as np
import random
from utils.sim import explore_pwm_equil
from utls.nn import predict_nn_v2

class QuadEnv(gym.Env):
	"""
	Gym environment for our custom system (Crazyflie quad).

	"""
	def __init__(self):
		gym.Env.__init__(self)

		# load dynamics model
		# self.dyn_nn = torch.load("_models/temp/2018-11-18--22-38-43.9_no_battery_dynamics_stack3_.pth")
		self.dyn_nn = torch.load(
			"_models/temp/2019-03-22--09-29-48.4_mfrl_ens_stack3_.pth")
		self.dyn_nn.eval()

		# load trained data for bounds on evironment
		# data_file = open("_models/temp/2018-11-18--22-38-43.9_no_battery_dynamics_stack3_--data.pkl",'rb')
		data_file = open(
			"_models/temp/2019-03-22--09-29-48.4_mfrl_ens_stack3_--data.pkl", 'rb')

		self.num_stack = 3
		self.state = None

		df = pickle.load(data_file)
		self.dyn_data = df

		# generate equilibrium data
		self.equil_act = explore_pwm_equil(df)
		self.init_act = np.tile(self.equil_act,self.num_stack)

		# set action bounds
		self.act_low = np.min(
			df[['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0']].values)
		self.act_high = np.max(
			df[['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0']].values)
		print("Actions are PWMS between:")
		print(self.act_low)
		print(self.act_high)

		# fit distributions to state data for initialization purposes / state space
		state_data = df[['omega_x0', 'omega_y0', 'omega_z0', 'pitch0', 'roll0',
                   'yaw0', 'lina_x0', 'lina_y0', 'lina_z0']]
		self.state_means = np.mean(state_data,axis=0).values
		self.state_vars = np.var(state_data, axis=0).values
		self.state_mins = np.min(state_data).values
		self.state_maxs = np.max(state_data).values

		print("Running on State Data Distribution ======")
		print("Means: ")
		print(np.mean(state_data, axis=0))
		print("Variances: ")
		print(np.var(state_data, axis=0))

		self.action_space = Box(
			low=np.tile(self.act_low,self.num_stack), 
			high=np.tile(self.act_high, self.num_stack))
		self.observation_space = Box(
			low=np.tile(self.state_mins, self.num_stack),
			high=np.tile(self.state_maxs, self.num_stack))

	def state_failed(self, s):
		"""
		Check whether a state has failed, so the trajectory should terminate.
		This happens when either the roll or pitch angle > 30
		Returns: failure flag [bool]
		"""
		if abs(s[3]) > 30.0 or abs(s[4]) > 30.0:
			return True
		return False

	def get_reward_state(self, s_next):
		"""
		Returns reward of being in a certain state.
		"""
		pitch = s_next[3]
		roll = s_next[4]
		if self.state_failed(s_next):
			# return -1 * 1000
			return 0
		loss = pitch**2 + roll**2

		reward = 1800 - loss # This should be positive. TODO: Double check
		# reward = 1
		return reward		

	def sample_random_state(self):
		"""
		Samples random state from previous logs. Ensures that the sampled
		state is not in a failed state.
		"""
		state_failed = True
		acceptable = False
		while not acceptable:
			# random_state = self.observation_space.sample()

			row_idx = np.random.randint(self.dyn_data.shape[0])
			random_state = self.dyn_data.iloc[row_idx, 12:12 + 9*self.num_stack].values
			random_state_s = self.dyn_data.iloc[row_idx, 12:12 + 9*self.num_stack].values
			random_state_a = self.dyn_data.iloc[row_idx, 12 + 9*self.num_stack + 4 :12 + 9*self.num_stack + 4 + 4*(self.num_stack -1)].values
			random_state = np.append(random_state_s, random_state_a)

			# if abs(random_state[3]) < 5 and abs(random_state[4]) < 5:
			# 	acceptable = True

			state_failed = self.state_failed(random_state)
			if not state_failed:
				acceptable = True
		return random_state

	def set_init_state(self):
		"""
		Returns a reasonable initial conditions:
		- low euler angles
		- sample accelerations from distribution (likely to get normal value)
		"""

		# sample proportional to data distribution
		samples = np.random.rand(9)
		generated_state = self.state_means+samples*np.sqrt(self.state_vars)

		# clamp to low Euler angles
		generated_state[3:6] = np.clip(generated_state[3:6],min=-5,max=5)

		# tile array, init state repeats x3
		init_state = np.tile(generated_state,self.num_stack)
		self.state = init_state
		return init_state

		

	def next_state(self, state, action):
		# Note that the states defined in env are different

		# predict_nn_v2

		state_dynamics = state[:9*self.n]
		action_dynamics = np.append(action, state[9*self.n : 9*self.n + 4 * (self.n - 1)])
		state_change = self.dyn_nn.predict(state_dynamics, action_dynamics)

		next_state = state[:9] + state_change
		past_state = state[:9*(self.n - 1)]

		new_state = np.concatenate((next_state, state[:9*(self.n - 1)]))
		new_action = np.concatenate((action, state[9*self.n: 9*self.n + 4*(self.n - 2)])) #

		return np.concatenate((new_state, new_action))


	def step(self, action):
		new_state = self.next_state(self.state, action)
		self.state = new_state
		reward = self.get_reward_state(new_state)
		done = False
		if self.state_failed(new_state):
			done = True
		return self.state, reward, done, {}

	def reset(self):
		# self.state = self.sample_random_state()
		self.state = self.set_init_state()
		return self.state


# class QuadSpace():


