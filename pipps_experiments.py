from pipps_psuedocode import PIPPS_policy
import gym
from model_general_nn import GeneralNN
import torch.nn as nn
import torch
import numpy as np
from utils.nn import *

env = gym.make('CartPole-v0')
# double bounds
env.unwrapped.theta_threshold_radians = 2*env.unwrapped.theta_threshold_radians
env.unwrapped.x_threshold = 2*env.unwrapped.x_threshold

observations = []
actions = [] 
rewards = []
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        # env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        observations.append(observation)
        actions.append([action])
        rewards.append(reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
o = np.array(observations)
actions = np.array(actions)

# shape into trainable set
d_o = o[1:,:]-o[:-1,:]
actions = actions[:-1,:]
o = o[:-1,:]

print('---')
print("X has shape: ", np.shape(o))
print("U has shape: ", np.shape(actions))
print("dX has shape: ", np.shape(d_o))
print('---')

ensemble = False

nn_params = {                           # all should be pretty self-explanatory
    'dx': np.shape(o)[1],
    'du': np.shape(actions)[1],
    'dt': np.shape(d_o)[1],
    'hid_width': 250,
    'hid_depth': 2,
    'bayesian_flag': True,
    'activation': Swish(),
    'dropout': 0.0,
    'split_flag': False,
    'pred_mode': 'Delta State',
    'ensemble': ensemble
}

train_params = {
    'epochs': 28,
    'batch_size': 18,
    'optim': 'Adam',
    'split': 0.8,
    'lr': .00175,  # bayesian .00175, mse:  .0001
    'lr_schedule': [30, .6],
    'test_loss_fnc': [],
    'preprocess': True,
    'noprint': True
}

if ensemble:
    newNN = EnsembleNN(nn_params, 10)
    acctest, acctrain = newNN.train_cust((o, actions, d_o), train_params)

else:
    newNN = GeneralNN(nn_params)
    newNN.init_weights_orth()
    if nn_params['bayesian_flag']:
        newNN.init_loss_fnc(d_o, l_mean=1, l_cov=1)  # data for std,
    acctest, acctrain = newNN.train_cust((o, actions, d_o), train_params)

pipps_nn_params = {                           # all should be pretty self-explanatory
    'dx': np.shape(o)[1],
    'du': np.shape(actions)[1],
    'hid_width': 32,
    'hid_depth': 2,
    'bayesian_flag': True,
    'activation': Swish(),
    'dropout': 0.0,
    'bayesian_flag': False
}

# for the pipps policy update step
policy_update_params = {
    'N': 10,
    'T': 15,
    'learning_rate': 1e-4,
}
PIPPSy = PIPPS_policy(pipps_nn_params, policy_update_params, newNN)
