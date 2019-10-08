import torch
import gym
from gym.spaces import Box
import pickle
import numpy as np
import random
from utils.sim import explorepwm_equil
from utils.nn import predict_nn_v2

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
		self.equil_act = explorepwm_equil(df)
		# self.init_act = np.tile(self.equil_act,self.num_stack)
		self.init_act = self.equil_act

		# set action bounds
		act_data = df[['m1pwm_0', 'm2pwm_0', 'm3pwm_0', 'm4pwm_0']]
		self.act_low = np.min(act_data.values, axis=0)
		self.act_high = np.max(act_data.values, axis=0)
		self.act_means = np.mean(act_data, axis=0).values
		self.act_vars = np.var(act_data, axis=0).values
		# print("Actions are PWMS between:")
		# print(self.act_low)
		# print(self.act_high)
		# print(self.act_means)
		# print(self.act_vars)

		# fit distributions to state data for initialization purposes / state space
		state_data = df[['omega_x0', 'omega_y0', 'omega_z0', 'pitch0', 'roll0',
                   'yaw0', 'lina_x0', 'lina_y0', 'lina_z0']]
		self.state_means = np.mean(state_data,axis=0).values
		self.state_vars = np.var(state_data, axis=0).values
		self.state_mins = np.min(state_data).values
		self.state_maxs = np.max(state_data).values

		# generate valeus form normalization in states during training
		# uses NormalizedBoxEnv
		self.act_means = np.tile(self.act_means, self.num_stack-1)
		self.act_std = np.tile(np.sqrt(self.act_vars), self.num_stack-1)

		self.norm_means = np.concatenate((np.tile(self.state_means,self.num_stack), self.act_means))
		self.norm_stds = np.concatenate(
			(np.tile(np.sqrt(self.state_vars), self.num_stack), self.act_std))

		print("Running on State Data Distribution ======")
		print("Means: ")
		print(np.mean(state_data, axis=0))
		print("Variances: ")
		print(np.var(state_data, axis=0))

		# define action as 4*num_stack and state as 12*num_stack
		# self.action_space = Box(
		# 	low=np.tile(self.act_low,self.num_stack), 
		# 	high=np.tile(self.act_high, self.num_stack))
		# self.observation_space = Box(
		# 	low=np.tile(self.state_mins, self.num_stack),
		# 	high=np.tile(self.state_maxs, self.num_stack))

		# define action as 4 and state as 4*(num_stack-1)+12*num_stack
		self.action_space = Box(
			low=self.act_low,
			high=self.act_high)

		state_low = np.concatenate((np.tile(self.state_mins, self.num_stack), 
						np.tile(self.act_low, self.num_stack-1)))
		state_high = np.concatenate((np.tile(self.state_mins, self.num_stack),  
						np.tile(self.act_high, self.num_stack-1)))
		self.observation_space = Box(
			low=state_low,
			high=state_high)

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
		Our loss function is a "inverted Huber loss".
		- Huber loss is linear outside of a quadratic inner region
		- ours is quadratic outside of a linear region in pitch and roll
		cost = -(c*p^2)  if p < a, for pitch and roll

		TODO: We will have to add loss when the mean PWM is below a certain value
		"""

		pitch = s_next[3]
		roll = s_next[4]
		if self.state_failed(s_next):
			# return -1 * 1000
			return -1000

		a1 = 1
		a2 = 1
		lin_pitch = 5
		lin_roll = 5
		if pitch > lin_pitch:
			loss_pitch = a1*pitch**2
		else:
			loss_pitch = a1*abs(pitch)

		if roll > lin_roll:
			loss_roll = a2*roll**2
		else:
			loss_roll = a2*abs(roll)

		loss_angles = loss_pitch+loss_roll

		# add a loss term for if the past actions were too low
		if True:
			lambda_act = .01
			# loss act should be a scaled difference between the mean of the
			# 	past actions and the mean of the equilibrium actions
			past_act_mean = np.mean(self.state[9*self.num_stack:])
			eq_mean = np.mean(self.equil_act)
			diff = eq_mean-past_act_mean
			loss_act = lambda_act*max(0, diff)

			loss = loss_angles+loss_act
			
		else:
			loss = loss_angles

		return -loss		

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
		generated_state[3:6] = np.clip(generated_state[3:6],-5,5)

		# tile array, init state repeats x3
		init_state = np.tile(generated_state,self.num_stack)

		# add actions if num_stack >1
		if self.num_stack >1:
			init_state = np.concatenate(
				(init_state, np.tile(self.equil_act, self.num_stack-1)))
		self.state = init_state
		return init_state

		

	def next_state(self, state, action):
		# Note that the states defined in env are different

		state_dynamics = state[:9*self.num_stack]
		action_dynamics = np.append(action, state[9*self.num_stack : 9*self.num_stack + 4 * (self.num_stack - 1)])
		state_change = self.dyn_nn.predict(state_dynamics, action_dynamics)

		next_state = state[:9] + state_change
		past_state = state[:9*(self.num_stack - 1)]

		new_state = np.concatenate((next_state, state[:9*(self.num_stack - 1)]))
		new_action = np.concatenate((action, state[9*self.num_stack: 9*self.num_stack + 4*(self.num_stack - 2)])) #

		return np.concatenate((new_state, new_action))


	def step(self, action):
		# print(action)
		s = self.state[:9*self.num_stack]
		a = np.concatenate((action, self.state[9*self.num_stack:]))
		new_state = predict_nn_v2(self.dyn_nn, s, a)
		# new_state = self.next_state(self.state, action)
		self.state = np.concatenate((new_state, self.state[9:]),axis=0)
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


