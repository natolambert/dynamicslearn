from pipps_psuedocode import PIPPS_policy
import gym
from model_general_nn import GeneralNN
import torch.nn as nn
import torch
import numpy as np
from utils.nn import *
from utils.data import *
# from utils.rl import *
import utils.rl

load_params = {
    'delta_state': True,                # normally leave as True, prediction mode
    # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
    'include_tplus1': True,
    # trims high vbat because these points the quad is not moving
    'takeoff_points': 180,
    # if all the euler angles (floats) don't change, it is not realistic data
    'trim_0_dX': False,
    'find_move': True,
    # if the states change by a large amount, not realistic
    'trime_large_dX': True,
    # IMPORTANT ONE: stacks the past states and inputs to pass into network
    'stack_states': 3,
    # adds a column to the dataframe tracking end of trajectories
    'terminals': True
}

# load_iono_txt('data_wiggle.txt', load_params)
# quit()

############ SETUP EXPERIMENT ENV ########################
# env = gym.make('CartPole-v1')
# env = gym.make('MountainCarContinuous-v0')
env = gym.make("CartPoleContEnv-v0")
# double bounds
# env.unwrapped.theta_threshold_radians = 2*env.unwrapped.theta_threshold_radians
# env.unwrapped.x_threshold = 2*env.unwrapped.x_threshold

observations = []
actions = [] 
rewards = []
for i_episode in range(50):
    observation = env.reset()
    rewards = []
    for t in range(100):
        # env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        #reshape obersvation
        observation = observation.reshape(-1,1)
        # print(action)
        observations.append(observation)
        actions.append([action])
        rewards.append(reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Ep reward: ", np.sum(rewards))
            break
    

############ PROCESS DATA ########################
# o = np.array(observations)    # mountain car
o = np.array(observations).squeeze()      # cartpole

actions = np.array(actions).reshape(-1,1)
# shape into trainable set
d_o = o[1:,:]-o[:-1,:]
actions = actions[:-1,:]
o = o[:-1,:]

print('---')
print("X has shape: ", np.shape(o))
print("U has shape: ", np.shape(actions))
print("dX has shape: ", np.shape(d_o))
print('---')

############ TRAIN MODEL ########################
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
    'hid_width': 50,
    'hid_depth': 2,
    'bayesian_flag': True,
    'activation': Swish(),
    'dropout': 0.0,
    'bayesian_flag': False
}

# for the pipps policy update step
policy_update_params = {
    'P': 100,
    'T': 15,
    'learning_rate': 3e-3,
}

############ INITAL PIPPS POLICY ########################
# init PIPPS policy
PIPPSy = PIPPS_policy(pipps_nn_params, policy_update_params, newNN)
PIPPSy.init_weights_orth()  # to see if this helps initial rollouts
# set cost function
"""
Observation:
        Type: Box(4)
        Num	Observation             Min             Max
        0	Cart Position           - 4.8            4.8
        1	Cart Velocity           - Inf            Inf
        2	Pole Angle              - 24 deg        24 deg
        3	Pole Velocity At Tip    - Inf            Inf
"""
def simple_cost_cartpole(vect):
    l_pos = 1
    l_vel = 5
    l_theta = 2
    l_theta_dot = 1
    return l_pos*vect[0]**2 + l_vel*vect[1]**2 \
        + l_theta*vect[2]**2 + l_theta_dot*vect[3]**2 

def simple_cost_car(vect):
    l_pos = 10
    l_vel = 2.5
    return -l_pos*vect[0] #+ l_vel*vect[1]**2 
        

PIPPSy.set_cost_function(simple_cost_cartpole)

# set baseline function
PIPPSy.set_baseline_function(np.mean)

PIPPSy.policy_step(o)
# PIPPSy.viz_comp_graph()
# quit()
############ PIPPS ITERATIONS ########################
P_rollouts = 20
for p in range(P_rollouts):
    print("------ PIPPS Training Rollout", p, " ------")
    observations_new = []
    actions = []
    rewards_fin = []
    for i_episode in range(10):
        observation = env.reset()
        rewards = []
        for t in range(100):
            if p > 10: env.render()
            observation = observation.reshape(-1)
            # print(observation)
            # action = PIPPSy.predict(observation)  # env.action_space.sample()
            action = PIPPSy.forward(torch.Tensor([observation]))  # env.action_space.sample()
            # PIPPSy.viz_comp_graph(action.requires_grad_(True))
            # print(action)
            # action = action.int().data.numpy()[0]
            action = action.data.numpy()
            # print(action)
            
            observation, reward, done, info = env.step(action[0])

            observation = observation.reshape(-1, 1)
            observations_new.append(observation)
            actions.append([action])
            rewards.append(reward)
            

            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                # print(np.sum(rewards))
                rewards_fin.append(np.sum(rewards))
                break
    observations_new = np.array(observations_new).squeeze()
    o = np.concatenate((o,observations_new),0)
    print("New Dataset has shape: ", np.shape(o))
    print("Reward at this iteration: ", np.mean(rewards_fin))
    print('---')
    PIPPSy.policy_step(np.array(o))
    # print('---')

