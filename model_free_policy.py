import numpy as np
# from sklearn.preprocessing import StandardScaler
from datetime import datetime
from datetime import timedelta
import struct
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import torch
import torch.nn as nn
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import itertools
import copy
	

model = "_models/temp/2018-10-23--15-28-38.2--Min error-765.235234375d=_150Hz_newnet_.pth"
dyn_nn = torch.load(model)
dyn_nn.eval()

X = np.array([])

data_file = open("_models/temp/2018-10-23--15-28-38.2--Min error-765.235234375d=_150Hz_newnet_data.pkl",'rb')
df = pickle.load(data_file)
print(df.shape)
row_idx = 0

state = df.iloc[row_idx, 12:21].values
action = df.iloc[row_idx, 21:26].values
change = df.iloc[row_idx, 0:9].values

# print(state)
print(type(state))
print(np.shape(state))
print(type(action))
print(np.shape(action))
print("ACTION")
print(action)
# print(df.columns.values)
print(df['vbat'].mean())
# print("Predicted")
# print(dyn_nn.predict(state, action))
# print("Actual")
# print(change)

mins = []
maxs = []
for i in range(9):

	states = df.iloc[:, 12+i].values
	# print(states.shape)
	# print(min(states))
	mins.append(min(states))
	maxs.append(max(states))

print(mins)
print(maxs)


class Transition():
	def __init__(self, s=0, a=0, a_index=0, s_next=0, r=0):
		self.s = s
		self.a = a
		self.a_index = a_index
		self.s_next = s_next
		self.r = r

class QLearner():
	def __init__(self, dynamics_model, dynamics_data):
		self.batch_size = 10
		self.num_actions = 81
		self.state_size = 9
		self.reward_size = 1
		self.eps = 0.5
		# Make target Q and current Q networks
		self.target_q = nn.Sequential(nn.Linear(self.state_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.Linear(64, self.num_actions))
		self.current_q = nn.Sequential(nn.Linear(self.state_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.Linear(64, self.num_actions))
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
		motor_discretized = [[30000, 35000, 40000], [30000, 35000, 40000], [30000, 35000, 40000], [30000, 35000, 40000]]
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
			random_state = self.dyn_data.iloc[row_idx, 12:21].values
			state_failed = self.state_failed(random_state)
		return random_state

	def sample_random_action(self):
		"""
		Samples a completely random action.
		"""
		ac = np.random.randint(81) # Returns an integer from 0 to 80
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
		new_transition = Transition(s=torch.from_numpy(s), a=torch.from_numpy(a), \
			a_index=ac, s_next=torch.from_numpy(s_next), r=torch.from_numpy(r))
		self.buffer.append(new_transition)	
		return new_transition

	def sample_mini_batch(self, size):
		"""
		Samples a mini_batch of size=size from the reply buffer.
		Returns: mini_batch
		"""
		buffer_length = len(self.buffer)
		mini_batch = np.random.choice(self.buffer, size, p=np.repeat(1.0 / buffer_length, buffer_length), replace=False)
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
		b_states, b_actions, b_action_indices, b_next_states, b_rewards = self.mini_batch_to_stacked(mini_batch)
		q  = self.target_q(b_next_states.float()).detach()
		max_a = torch.max(q, dim=1)[0]
		max_a = max_a.unsqueeze(1)
		b_y = b_rewards.unsqueeze(1).float() + self.gamma * max_a.float()
		return b_y

	def gradient_step(self, mini_batch):
		"""
		Takes one gradient step in training the Q network
		"""
		b_states, b_actions, b_action_indices, b_next_states, b_rewards = self.mini_batch_to_stacked(mini_batch)
		
		# Get y
		y = self.compute_y(mini_batch)
		curr_q = self.current_q(b_states.float()).gather(1, b_action_indices.view(-1, 1))
		# loss = self.criterion(curr_q, y) # y is the target
		loss = nn.functional.smooth_l1_loss(curr_q, y)

		# print("Difference", curr_q[0], y[0], curr_q[0] - y[0])
		# print(curr_q.size(), y.size())
		self.optimizer.zero_grad() # Clear previous gradients
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
		q  = self.current_q(torch.from_numpy(state).float())
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
			total_reward += rollout_reward / horizon # Average reward over steps
		return total_reward / rollouts


	def train(self):
		mini_batch_size = 100
		n = 5000 # Number of times to run before upddating target q
		num_target_updates = 100
		total_iterations = n * num_target_updates # how many times we want to update target_q * n

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

# qlearn = QLearner(dyn_nn, df)
# print("\n ====== TRAINING ====== \n")
# qlearn.train()


