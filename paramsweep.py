# Our infrastucture files
from utils_data import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN, predict_nn
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
    'delta_state': True,
    'takeoff_points': 5,
    'trim_0_dX': True,
    'trime_large_dX': True,
    'bound_inputs': [20000,65500],
    'stack_states': 3,
    'collision_flag': False,
    'shuffle_here': False,
    'timestep_flags': []
}

dir_list = ["150Hz/sep12_float/", "150Hz/sep13_150/"]
other_dirs = ["/sep13_150_2/","/sep14_150/","/sep14_150_2/","/sep14_150_3/"]
df = load_dirs(dir_list, load_params)

data_params = {
    'states' : [],
    'inputs' : [],
    'change_states' : []
}

X, U, dX = df_to_training(df, data_params)

nn_params = {
    'dx' : np.shape(X)[1],
    'du' : np.shape(U)[1],
    'dt' : np.shape(dX)[1],
    'hid_width' : 500,
    'hid_depth' : 3,
    'bayesian_flag' : True,
    'activation': Swish(),
    'dropout' : 0.0,
    'split_flag' : False,
    'pred_mode' : 'Delta State'
}

train_params = {
    'epochs' : 75,
    'batch_size' : 32,
    'optim' : 'Adam',
    'split' : 0.8,
    'lr': .0003,
    'lr_schedule' : [20,.5],
    'test_loss_fnc' : [],
    'preprocess' : True
}


############# CROSS VALIDATION ##############################################################################

if False:
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
    print(mean(cross_val_err_test))

######################### HYPER PARAMS ######################################################

# Setup values
# learns = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 5e-7, 1e-7]
# batches = [15, 25, 32, 50, 100, 200]


# Setup DF
d = {"MinTrainLoss" : [],
    "MinTestLoss" : [],
    "LR": [],
    "Batch Size": [],
    "Activation": [],
    "ProbFlag": [],
    "Opt": []
}

results = pd.DataFrame(d)

learns = [.001, .0005, .0002]
batches = [20, 32, 45]
# Iteration Loop
i = 0
for l in learns:
    for b in batches:
        for act in [Swish(), nn.ReLU()]:
            for bayes_flag in [True]:
                for opt in ["Adam"]:
                    print('-----------------------------------------------------')
                    print('ITERATION: ', i)


                    nn_params = {
                        'dx' : np.shape(X)[1],
                        'du' : np.shape(U)[1],
                        'dt' : np.shape(dX)[1],
                        'hid_width' : 500,
                        'hid_depth' : 3,
                        'bayesian_flag' : True,
                        'activation': act,
                        'dropout' : 0.0,
                        'split_flag' : False,
                        'pred_mode' : 'Delta State'
                    }

                    train_params = {
                        'epochs' : 3,
                        'batch_size' : b,
                        'optim' : opt,
                        'split' : 0.8,
                        'lr': l,
                        'lr_schedule' : [20,.5],
                        'test_loss_fnc' : [],
                        'preprocess' : True
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
                        "Batch Size": b,
                        "Activation": act,
                        "ProbFlag": bayes_flag,
                        "Opt":opt
                    }
                    print(d)
                    results = results.append(d, ignore_index=True)
                    i+=1


print(results)
results.to_csv('PARAMSWEEP.csv')
print('Saved and Done')
