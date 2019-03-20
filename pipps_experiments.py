from pipps_psuedocode import PIPPS_policy
import gym
from model_general_nn import GeneralNN

from model_ensemble_nn import EnsembleNN

import torch.nn as nn
import torch
import numpy as np
from utils.nn import *
from utils.data import *
from utils.rl import *
# import utils.rl

# saving files
import datetime
import pickle
import time

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

# if not saving, loads an old model
save = False


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
rewards_tot = []
rand_ep = 100 if save else 0
for i_episode in range(rand_ep):
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
            # print("Episode finished after {} timesteps".format(t+1))
            # print("Ep reward: ", np.sum(rewards)) 
            rewards_tot.append(np.sum(rewards))
            break
    

############ PROCESS DATA ########################
# o = np.array(observations)    # mountain car
if save:
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
    print("Mean Random Reward: ", np.mean(rewards_tot))
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
        'epochs': 32,
        'batch_size': 18,
        'optim': 'Adam',
        'split': 0.8,
        'lr': .00375,  # bayesian .00175, mse:  .0001
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

if save:
    dir_str = str('_models/_cartpole/')
    data_name = '_cont_100ep_'
    model_name = 'cartpole_basic'
    date_str = str(datetime.datetime.now())[:-5]
    date_str = date_str.replace(' ', '--').replace(':', '-')    
    # info_str = "_" + model_name + "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
    # + "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
    info_str = "_" + model_name + "_" #+ "stack" + \
    model_name = dir_str + date_str + info_str
    newNN.save_model(model_name + '.pth')
    print('Saving model to', model_name)

    normX, normU, normdX = newNN.getNormScalers()
    with open(model_name+"--normparams.pkl", 'wb') as pickle_file:
        pickle.dump((normX, normU, normdX), pickle_file, protocol=2)
    time.sleep(2)

    # Saves data file
    with open(model_name+"--data.pkl", 'wb') as pickle_file:
      pickle.dump((d_o, actions, o), pickle_file, protocol=2)
    time.sleep(2)
else:
    print("Loading previous model and data")
    model = '_models/_cartpole/2019-03-19--12-32-21.6_cartpole_basic_.pth'
    newNN = torch.load(model)
    newNN.eval()

    data_file = '_models/_cartpole/2019-03-19--12-32-21.6_cartpole_basic_--data.pkl'
    with open(data_file, 'rb') as pickle_file:
        (d_o, actions, o) = pickle.load(pickle_file)

    print('---')
    print("X has shape: ", np.shape(o))
    print("U has shape: ", np.shape(actions))
    print("dX has shape: ", np.shape(d_o))
    print('---')

######### PIPPS PART ###########

pipps_nn_params = {                           # all should be pretty self-explanatory
    'dx': np.shape(o)[1],
    'du': np.shape(actions)[1],
    'hid_width': 100,
    'hid_depth': 2,
    'bayesian_flag': True,
    'activation': Swish(),
    'dropout': 0.2,
    'bayesian_flag': False
}

# for the pipps policy update step
policy_update_params = {
    'P': 20,
    'T': 25,
    'learning_rate': 3e-4,
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
    l_pos = 100
    l_vel = 10
    l_theta = 200
    l_theta_dot = 5
    return 1*(l_pos*(vect[0]**2) + l_vel*vect[1]**2 \
        + l_theta*(vect[2]**2) + l_theta_dot*vect[3]**2)

def simple_cost_car(vect):
    l_pos = 10
    l_vel = 2.5
    return -l_pos*vect[0] #+ l_vel*vect[1]**2 
        

PIPPSy.set_cost_function(simple_cost_cartpole)

# set baseline function
PIPPSy.set_baseline_function(np.mean)
PIPPSy.set_statespace_normal([],[])

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
    for i_episode in range(15):
        observation = env.reset()
        rewards = []
        for t in range(100):
            if p > 10: env.render()
            observation = observation.reshape(-1)
            # print(observation)
            # action = PIPPSy.predict(observation)  # env.action_space.sample()
            # env.action_space.sample()
            action = PIPPSy.forward(torch.Tensor(
                [observation]), normalize=False)
            # PIPPSy.viz_comp_graph(action.requires_grad_(True))
            # print(action)
            # action = action.int().data.numpy()[0]
            action = torch.clamp(action, min = -1, max = 1).data.numpy()
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
    PIPPSy.T +=1 # longer horizon with more data
    # print('---')

