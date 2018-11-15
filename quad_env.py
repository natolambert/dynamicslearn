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
		# self.dyn_nn = torch.load("_models/temp/2018-11-01--16-20-11.5--Min error-21.66078d=_50Hz_try_stack1_.pth")
		# self.dyn_nn = torch.load("_models/temp/2018-11-08--21-47-50.7--Min error-25.442549d=_50Hz_try_stack4_.pth")
		self.dyn_nn = torch.load("_models/temp/2018-11-14--10-52-20.9_output_model_stack3_.pth")

		self.dyn_nn.eval()
		# data_file = open("_models/temp/2018-11-01--16-20-11.5--Min error-21.66078d=_50Hz_try_stack1_--data.pkl",'rb')
		# data_file = open("_models/temp/2018-11-08--21-47-50.7--Min error-25.442549d=_50Hz_try_stack4_--data.pkl",'rb')
		data_file = open("_models/temp/2018-11-14--10-52-20.9_output_model_stack3_--data.pkl",'rb')

		self.n = 3

		df = pickle.load(data_file)
		# print("\n===UNSTACKED===")
		# s = df.iloc[2, 12:21].values
		# a = df.iloc[2, 21:26].values
		# print("\n Prediction")
		# print(self.dyn_nn.predict(s, a))

		print(df.columns.values)
		s_stacked = df.iloc[2, 12:12+9*self.n].values
		v_bat = df.iloc[2,-1]
		full_state = np.append(s_stacked, v_bat)
		print(full_state)

		print("\n V_BAT")
		print(full_state.shape)
		a_stacked = df.iloc[2, 12+9*self.n:12+9*self.n+4*self.n].values
		print(a_stacked)
		print("\n Prediction")
		print(self.dyn_nn.predict(full_state, a_stacked))


		assert(False)

		v_bats = df.iloc[:, -1]
		print(min(v_bats), max(v_bats))


		self.dyn_data = df
		self.action_space = Box(low=np.tile([30000, 30000, 30000, 30000],self.n), high=np.tile([50000, 50000, 50000, 50000],self.n))
		self.observation_space = Box(low=np.append(np.tile([-330, -350, -60, -30, -30, -140, -8, -8, 5], self.n), 3500), \
		 	high=np.append(np.tile([350, 370, 140, 30, 30, 160, 5, 7, 20], self.n), 4000))

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
			return -1 * 1000
		loss = pitch**2 + roll**2
		# print(loss)
		# print(pitch, roll)
		reward = 100 - loss # This should be positive. TODO: Double check
		return reward		

	def sample_random_state(self):
		"""
		Samples random state from previous logs. Ensures that the sampled
		state is not in a failed state.
		"""
		state_failed = True
		while state_failed:
			row_idx = np.random.randint(self.dyn_data.shape[0])
			random_state = self.dyn_data.iloc[row_idx, 12:12 + 9*self.n].values
			battery = self.dyn_data.iloc[row_idx, -1]
			random_state = np.append(random_state, battery)
			state_failed = self.state_failed(random_state)
		return random_state

	def next_state(self, state, action):
		state_change = self.dyn_nn.predict(state, action)
		recent_state = state[:9]
		new_recent_state = recent_state + state_change
		return np.append(new_recent_state, state[:-9])
		# return self.dyn_nn.predict(state, action) + state

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


