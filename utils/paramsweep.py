# Our infrastucture files
from utils_data import *
from utils_nn import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN
from model_split_nn import SplitModel
from _activation_swish import Swish

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss

# timing etc
import time
import datetime
import os

# Plotting
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import KFold

print('\n')
date_str = str(datetime.datetime.now())[:-5]
date_str = date_str.replace(' ','--').replace(':', '-')
print('Running... paramsweep.py' + date_str +'\n')

load_params ={
    'delta_state': True,                # normally leave as True, prediction mode
    'takeoff_points': 180,              # If not trimming data with fast log, need another way to get rid of repeated 0s
    'trim_0_dX': True,                  # if all the euler angles (floats) don't change, it is not realistic data
    'trime_large_dX': True,             # if the states change by a large amount, not realistic
    'bound_inputs': [20000,65500],      # Anything out of here is erroneous anyways. Can be used to focus training
    'stack_states': 3,                  # IMPORTANT ONE: stacks the past states and inputs to pass into network
    'collision_flag': False,            # looks for sharp changes to tthrow out items post collision
    'shuffle_here': False,              # shuffle pre training, makes it hard to plot trajectories
    'timestep_flags': [],               # if you want to filter rostime stamps, do it here
    'battery' : True,                   # if battery voltage is in the state data
    'terminals': False,                 # adds a column to the dataframe tracking end of trajectories
    'fastLog' : True,                   # if using the software with the new fast log
    'contFreq' : 1                      # Number of times the control freq you will be using is faster than that at data logging
}                                       # for contFreq, use 1 if training at the same rate data was collected at

dir_list = ["_newquad1/fixed_samp/c50_samp300_rand/","_newquad1/fixed_samp/c50_samp300_roll1/","_newquad1/fixed_samp/c50_samp300_roll2/","_newquad1/fixed_samp/c50_samp300_roll3/"]

df = load_dirs(dir_list, load_params)

data_params = {
    'states' : [],                      # most of these are to be implented for easily training specific states etc
    'inputs' : [],
    'change_states' : [],
    'battery' : True                    # Need to include battery here too
}

X, U, dX = df_to_training(df, data_params)

nn_params = {                           # all should be pretty self-explanatory
    'dx' : np.shape(X)[1],
    'du' : np.shape(U)[1],
    'dt' : np.shape(dX)[1],
    'hid_width' : 250,
    'hid_depth' : 2,
    'bayesian_flag' : True,
    'activation': Swish(),
    'dropout' : 0.0,
    'split_flag' : False,
    'pred_mode' : 'Delta State',
    'ensemble' : False
}

train_params = {
    'epochs' : 60,
    'batch_size' : 18,
    'optim' : 'Adam',
    'split' : 0.8,
    'lr': .002,
    'lr_schedule' : [30,.6],
    'test_loss_fnc' : [],
    'preprocess' : True,
    'noprint' : False
}
############# CROSS VALIDATION ##############################################################################
print("----------------------------------------------------------")
print("Running Cross validation for prediction vs stack states")
if True:
    for stack in [4,5,6]:
        load_params ={
            'delta_state': True,                # normally leave as True, prediction mode
            'takeoff_points': 180,              # If not trimming data with fast log, need another way to get rid of repeated 0s
            'trim_0_dX': True,                  # if all the euler angles (floats) don't change, it is not realistic data
            'trime_large_dX': True,             # if the states change by a large amount, not realistic
            'bound_inputs': [20000,65500],      # Anything out of here is erroneous anyways. Can be used to focus training
            'stack_states': stack,                  # IMPORTANT ONE: stacks the past states and inputs to pass into network
            'collision_flag': False,            # looks for sharp changes to tthrow out items post collision
            'shuffle_here': False,              # shuffle pre training, makes it hard to plot trajectories
            'timestep_flags': [],               # if you want to filter rostime stamps, do it here
            'battery' : True,                   # if battery voltage is in the state data
            'terminals': False,                 # adds a column to the dataframe tracking end of trajectories
            'fastLog' : True,                   # if using the software with the new fast log
            'contFreq' : 1                      # Number of times the control freq you will be using is faster than that at data logging
        }                                       # for contFreq, use 1 if training at the same rate data was collected at

        dir_list = ["_newquad1/fixed_samp/c50_samp300_rand/","_newquad1/fixed_samp/c50_samp300_roll1/","_newquad1/fixed_samp/c50_samp300_roll2/","_newquad1/fixed_samp/c50_samp300_roll3/"]

        df = load_dirs(dir_list, load_params)

        data_params = {
            'states' : [],                      # most of these are to be implented for easily training specific states etc
            'inputs' : [],
            'change_states' : [],
            'battery' : True                    # Need to include battery here too
        }

        X, U, dX = df_to_training(df, data_params)

        nn_params = {                           # all should be pretty self-explanatory
            'dx' : np.shape(X)[1],
            'du' : np.shape(U)[1],
            'dt' : np.shape(dX)[1],
            'hid_width' : 250,
            'hid_depth' : 2,
            'bayesian_flag' : True,
            'activation': Swish(),
            'dropout' : 0.0,
            'split_flag' : False,
            'pred_mode' : 'Delta State',
            'ensemble' : False
        }

        train_params = {
            'epochs' : 100,
            'batch_size' : 18,
            'optim' : 'Adam',
            'split' : 0.8,
            'lr': .002,
            'lr_schedule' : [30,.6],
            'test_loss_fnc' : [],
            'preprocess' : True,
            'noprint' : False
        }

        K = 5

        kf = KFold(n_splits=K)
        kf.get_n_splits(X)

        cross_val_err_test = []
        cross_val_err_train = []
        # iterate through the validation sets
        for train_index, test_index in kf.split(X):
            # print(train_index)
            # print(test_index)   # train = data[0]
            X_train, X_test = X[train_index], X[test_index]
            U_train, U_test = U[train_index], U[test_index]
            dX_train, dX_test = dX[train_index], dX[test_index]
            X_k = np.append(X_train, X_test, axis=0)
            U_k = np.append(U_train, U_test, axis=0)
            dX_k = np.append(dX_train, dX_test, axis=0)

            # Initialize
            newNN = GeneralNN(nn_params)
            newNN.init_weights_orth()
            newNN.init_loss_fnc(dX,l_mean = 1,l_cov = 1) # data for std,
            acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

            cross_val_err_test.append(min(acctest))
            cross_val_err_train.append(min(acctrain))
        print("Stack is:", stack, " with a mean test error of: ", np.mean(cross_val_err_test))
        print("Stack is:", stack, " with a mean train error of: ", np.mean(cross_val_err_train))
        print('-----')

print('----------')
######################### HYPER PARAMS ######################################################
quit()
# Setup values
# learns = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 5e-7, 1e-7]
# batches = [15, 25, 32, 50, 100, 200]

df = load_dirs(dir_list, load_params)

data_params = {
    'states' : [],                      # most of these are to be implented for easily training specific states etc
    'inputs' : [],
    'change_states' : [],
    'battery' : True                    # Need to include battery here too
}

X, U, dX = df_to_training(df, data_params)

# Setup DF
d = {"MinTrainLoss" : [],
    "MinTestLoss" : [],
    "LR": [],
    "Depth": [],
    "Batch Size": [],
    "Activation": [],
    "ProbFlag": [],
    "Opt": []
}

results = pd.DataFrame(d)

learns = [.003, .002, .0015]
batches = [15, 18, 20, 23] #32, 45]
# Iteration Loop
i = 0
for l in learns:
    for b in batches:
        for act in [Swish()]:
            for d in [1,2,3]:
                for opt in ["Adam"]:
                    print('-----------------------------------------------------')
                    print('ITERATION: ', i)


                    nn_params = {                           # all should be pretty self-explanatory
                        'dx' : np.shape(X)[1],
                        'du' : np.shape(U)[1],
                        'dt' : np.shape(dX)[1],
                        'hid_width' : 250,
                        'hid_depth' : d,
                        'bayesian_flag' : True,
                        'activation': Swish(),
                        'dropout' : 0.0,
                        'split_flag' : False,
                        'pred_mode' : 'Delta State',
                        'ensemble' : False
                    }

                    train_params = {
                        'epochs' : 155,
                        'batch_size' : b,
                        'optim' : 'Adam',
                        'split' : 0.8,
                        'lr': l,
                        'lr_schedule' : [30,.6],
                        'test_loss_fnc' : [],
                        'preprocess' : True,
                        'noprint' : False
                    }

                    newNN = GeneralNN(nn_params)
                    newNN.init_weights_orth()
                    newNN.init_loss_fnc(dX,l_mean = 1,l_cov = 1) # data for std,

                    # Train
                    try:
                        acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

                    except:
                        acctest = [np.inf]
                        acctrain = [np.inf]

                    best_loss_test = min(acctest)
                    best_loss_train = min(acctrain)

                    d = {"MinTrainLoss" : best_loss_train,
                        "MinTestLoss" : best_loss_test,
                        "LR": l,
                        "Depth": d,
                        "Batch Size": b,
                        "Activation": act,
                        "ProbFlag": True,
                        "Opt":opt
                    }
                    print(d)
                    results = results.append(d, ignore_index=True)
                    i+=1


print(results)
results.to_csv('PARAMSWEEP-2.csv')
print('Saved and Done')
