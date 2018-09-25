

# minimal file to load existing NN and continue training with new data

from dynamics import *
import pickle
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
from dynamics_crazyflie_linearized import CrazyFlie
from utils_plot import *
from utils_data import *
from model_general_nn import GeneralNN, predict_nn
import torch
from torch.nn import MSELoss
import time
import datetime
from model_split_nn import SplitModel
import os
# Plotting
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
import csv
from enum import Enum
import pandas as pd

from sklearn.model_selection import KFold

delta = True
input_stack = 4
X_rl, U_rl, dX_rl = stack_dir("/sep12_float/", delta = delta, input_stack = input_stack)
X_rl_2, U_rl_2, dX_rl_2 = stack_dir("/sep13_150/", delta = delta, input_stack = input_stack)
X_rl_3, U_rl_3, dX_rl_3 = stack_dir("/sep13_150_2/", delta = delta, input_stack = input_stack)
X_rl_4, U_rl_4, dX_rl_4 = stack_dir("/sep14_150_2/", delta = delta, input_stack = input_stack, takeoff=True)
X_rl_5, U_rl_5, dX_rl_5 = stack_dir("/sep14_150_3/", delta = delta, input_stack = input_stack, takeoff=True)


# quit()
X = X_rl
U = U_rl
dX = dX_rl
# X = np.concatenate((X_rl,X_rl_2),axis=0)
# U = np.concatenate((U_rl,U_rl_2),axis=0)
# dX = np.concatenate((dX_rl,dX_rl_2),axis=0)

# X = np.concatenate((X_rl,X_rl_2,X_rl_3),axis=0)
# U = np.concatenate((U_rl,U_rl_2, U_rl_3),axis=0)
# dX = np.concatenate((dX_rl,dX_rl_2, dX_rl_3),axis=0)

# X = np.concatenate((X_rl,X_rl_2,X_rl_3,X_rl_4),axis=0)
# U = np.concatenate((U_rl,U_rl_2, U_rl_3, U_rl_4),axis=0)
# dX = np.concatenate((dX_rl,dX_rl_2, dX_rl_3, dX_rl_4),axis=0)
#
X = np.concatenate((X_rl,X_rl_2,X_rl_3,X_rl_4, X_rl_5),axis=0)
U = np.concatenate((U_rl,U_rl_2, U_rl_3, U_rl_4, U_rl_5),axis=0)
dX = np.concatenate((dX_rl,dX_rl_2, dX_rl_3, dX_rl_4, dX_rl_5),axis=0)

w = 500     # Network width
e = 60     # number of epochs
# lr = 3e-3    # learning rate
b = 32
l = .0005
dims = [0,1,2,3,4,5,6,7,8]
depth = 3
prob_flag = True
#
dX = dX[:, 3:6]     # Only Euler Angles
# noisy = dX + np.random.uniform(-0.176056338/2, 0.176056338/2, size =np.shape(dX))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        # m.bias.data.fill_(0.01)

############# CROSS VALIDATION ##############################################################################

K = 5
n, d_o = np.shape(dX)
_, dx = np.shape(X)
_, du = np.shape(U)
# data = (X, U, dX)
kf = KFold(n_splits=K)
kf.get_n_splits(X)
print(kf)

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
    newNN = GeneralNN(n_in_input = du,
                        n_in_state = dx,
                        hidden_w=w,
                        n_out = d_o,
                        state_idx_l=dims,
                        prob=True,
                        input_mode = 'Stacked Data',
                        pred_mode = 'Delta State',
                        depth=depth,
                        activation="Swish",
                        B = 1.0,
                        outIdx = dims,
                        dropout=0.0,
                        split_flag = False)

    newNN.features.apply(init_weights)
    if prob_flag:
        newNN.loss_fnc.scalers = torch.Tensor(np.std(dX,axis=0))


    # Train
    acctest, acctrain = newNN.train_cust((X_k, U_k, dX_k),
                        learning_rate = l,
                        epochs=e,
                        batch_size = b,
                        optim="Adam",
                        split=0.8)


    cross_val_err_test.append(min(acctest))
    cross_val_err_train.append(min(acctrain))
print(mean(cross_val_err_test))
quit()
########################################################################################################


# Setup values
# learns = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 5e-7, 1e-7]
# batches = [15, 25, 32, 50, 100, 200]


# Setup DF
d = {"MinTrainLoss" : [], "MinTestLoss" : [], "LR": [], "Batch Size": [],  "Activation": [], "ProbFlag": [], "Opt": []}
results = pd.DataFrame(d)
print(results)


learns = [.001, .0005, .0002]
batches = [20, 32, 45]
# Iteration Loop
i = 0
for l in learns:
    for b in batches:
        for act in ["Swish", "Softsign"]:
            for boooool in [True, False]:
                for opt in ["Adam"]:
                    print('-----------------------------------------------------')
                    print('ITERATION: ', i)
                    print('Running| l:', l, ' | b:', b, ' | Activation: ', act, '| Prob: ', boooool, '|Opt: ', opt)


                    n, d_o = np.shape(dX_rl)
                    _, dx = np.shape(X_rl)
                    _, du = np.shape(U_rl)



                    # Initialize
                    newNN = GeneralNN(n_in_input = du,
                                        n_in_state = dx,
                                        hidden_w=w,
                                        n_out = d_o,
                                        state_idx_l=dims,
                                        prob=boooool,
                                        input_mode = 'Stacked Data',
                                        pred_mode = 'Delta State',
                                        depth=depth,
                                        activation=act,
                                        B = 1.0,
                                        outIdx = dims,
                                        dropout=0.0,
                                        split_flag = False)

                    newNN.features.apply(init_weights)
                    if prob_flag:
                        newNN.loss_fnc.scalers = torch.Tensor(np.std(dX_rl,axis=0))


                    # Train
                    try:
                        acctest, acctrain = newNN.train_cust((X_rl, U_rl, dX_rl),
                                            learning_rate = l,
                                            epochs=e,
                                            batch_size = b,
                                            optim=opt,
                                            split=0.8)
                    except:
                        acctest = [np.inf]
                        acctrain = [np.inf]

                    best_loss_test = min(acctest)
                    best_loss_train = min(acctrain)
                    d = {"MinTrainLoss" : best_loss_train, "MinTestLoss" : best_loss_test, "LR": l, "Batch Size": b,  "Activation": act, "ProbFlag": boooool, "Opt":opt}
                    print(d)
                    results = results.append(d, ignore_index=True)
                    i+=1


print(results)
results.to_csv('PARAMSWEEP.csv')
print('Saved and Done')
