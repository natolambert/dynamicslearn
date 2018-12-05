import torch
import gym
from gym.spaces import Box
import pickle
import numpy as np


class QuadEnv(gym.Env):
	"""
	Gym environment for our custom system (Crazyflie quad).

	"""
	def __init__(self):
		gym.Env.__init__(self)
		# self.dyn_nn = torch.load("_models/temp/2018-11-18--22-38-43.9_no_battery_dynamics_stack3_.pth")
		self.dyn_nn = torch.load("_models/temp/2018-12-04--21-17-34.2_quad2_stack3_.pth")

		self.dyn_nn.eval()

		# data_file = open("_models/temp/2018-11-18--22-38-43.9_no_battery_dynamics_stack3_--data.pkl",'rb')
		data_file = open("_models/temp/2018-12-04--21-17-34.2_quad2_stack3_--data.pkl",'rb')

		self.n = 3
		self.state = None

		df = pickle.load(data_file)
		# print("\n===UNSTACKED===")
		# s = df.iloc[2, 12:21].values
		# a = df.iloc[2, 21:26].values
		# print("\n Prediction")
		# print(self.dyn_nn.predict(s, a))

		print(df.columns.values)

		r = 500
		print("State Before")
		print(df.iloc[r, 12:12+9].values)
		print("State Change")
		print(df.iloc[r, :9].values)
		print("State After")
		print(df.iloc[r+1, 12:12+9].values)
		print("Is this the same")
		print(df.iloc[r, :9].values + df.iloc[r, 12:12+9].values)
		print("Compare to State Before")
		print(df.iloc[r+1, 12+9:12+9+9].values)

		s_stacked = df.iloc[2, 12:12+9*self.n].values
		v_bat = df.iloc[2,-1]
		# full_state = np.append(s_stacked, v_bat)
		# print(full_state)

		# print("\n V_BAT")
		# print(full_state.shape)
		a_stacked = df.iloc[2, 12+9*self.n:12+9*self.n+4*self.n].values
		print(a_stacked)
		print("\n Prediction")
		print(self.dyn_nn.predict(s_stacked, a_stacked))


		# assert(False)

		v_bats = df.iloc[:, -1]
		print(min(v_bats), max(v_bats))

		self.dyn_data = df

		all_states = df.iloc[:, 12:12+9].values
		all_actions = df.iloc[:, 12+9*self.n:12+9*self.n + 4].values


		min_state_bounds = [min(all_states[:, i]) for i in range(len(all_states[0,:]))]
		max_state_bounds = [max(all_states[:, i]) for i in range(len(all_states[0,:]))]
		min_action_bounds = [min(all_actions[:, i]) for i in range(len(all_actions[0,:]))]
		max_action_bounds = [max(all_actions[:, i]) for i in range(len(all_actions[0,:]))]

		low_state_s = np.tile(min_state_bounds, self.n)
		low_state_a = np.tile(min_action_bounds,self.n - 1)
		high_state_s = np.tile(max_state_bounds, self.n)
		high_state_a = np.tile(max_action_bounds,self.n - 1)


		self.action_space = Box(low=np.array(min_action_bounds), high=np.array(max_action_bounds))
		self.observation_space = Box(low=np.append(low_state_s, low_state_a), \
		 	high=np.append(high_state_s, high_state_a))

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
			random_state = self.dyn_data.iloc[row_idx, 12:12 + 9*self.n].values
			random_state_s = self.dyn_data.iloc[row_idx, 12:12 + 9*self.n].values
			random_state_a = self.dyn_data.iloc[row_idx, 12 + 9*self.n + 4 :12 + 9*self.n + 4 + 4*(self.n -1)].values
			random_state = np.append(random_state_s, random_state_a)

			# if abs(random_state[3]) < 5 and abs(random_state[4]) < 5:
			# 	acceptable = True

			state_failed = self.state_failed(random_state)
			if not state_failed:
				acceptable = True
		return random_state

	def next_state(self, state, action):
		# Note that the states defined in env are different
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
		self.state = self.sample_random_state()
		return self.state


# class QuadSpace():


