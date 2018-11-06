# Our infrastucture files
from utils_data import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN, predict_nn
from model_split_nn import SplitModel
from _activation_swish import Swish
from model_ensemble_nn import EnsembleNN

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

import argparse

######################################################################

# adding arguments to make code easier to work with
parser = argparse.ArgumentParser(description='Train a Neural Netowrk off Autonomous Data')
parser.add_argument('--log', action='store_true',
                    help='a flag for storing a training log in a txt file')
parser.add_argument('--noprint', action='store_false',
                    help='turn off printing in the terminal window for epochs')
parser.add_argument('--ensemble', action='store_true',
                    help='trains an ensemble of models instead of one network')

args = parser.parse_args()

log = args.log
noprint = args.noprint
ensemble = args.ensemble
######################################################################

print('\n')
date_str = str(datetime.datetime.now())[:-5]
date_str = date_str.replace(' ','--').replace(':', '-')
print('Running... trainNN_RL.py' + date_str +'\n')

load_params ={
    'delta_state': True,                # normally leave as True, prediction mode
    'takeoff_points': 180,              # If not trimming data with fast log, need another way to get rid of repeated 0s
    'trim_0_dX': True,                  # if all the euler angles (floats) don't change, it is not realistic data
    'trime_large_dX': True,             # if the states change by a large amount, not realistic
    'bound_inputs': [20000,65500],      # Anything out of here is erroneous anyways. Can be used to focus training
    'stack_states': 1,                  # IMPORTANT ONE: stacks the past states and inputs to pass into network
    'collision_flag': False,            # looks for sharp changes to tthrow out items post collision
    'shuffle_here': False,              # shuffle pre training, makes it hard to plot trajectories
    'timestep_flags': [],               # if you want to filter rostime stamps, do it here
    'battery' : True,                   # if battery voltage is in the state data
    'terminals': False,                 # adds a column to the dataframe tracking end of trajectories
    'fastLog' : True,                   # if using the software with the new fast log
    'contFreq' : 1                      # Number of times the control freq you will be using is faster than that at data logging
}                                       # for contFreq, use 1 if training at the same rate data was collected at

dir_list = ["c50_samp300_rand/", "c50_samp300_roll1/", "c50_samp300_roll2/", "c50_samp300_roll3/"]
other_dirs = ["150Hz/sep13_150_2/","/150Hzsep14_150_2/","150Hz/sep14_150_3/"]
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
    'ensemble' : ensemble
}

train_params = {
    'epochs' : 120,
    'batch_size' : 32,
    'optim' : 'Adam',
    'split' : 0.8,
    'lr': .002,
    'lr_schedule' : [30,.6],
    'test_loss_fnc' : [],
    'preprocess' : True,
    'noprint' : noprint
}

# log file
if log:
    with open('_training_logs/'+'logfile' + date_str + '.txt', 'w') as my_file:
        my_file.write("Logfile for training run: " + date_str +"\n")
        my_file.write("============================================="+"\n")
        my_file.write("Data Load Params:"+"\n")
        for k, v in load_params.items():
            my_file.write(str(k) + ' >>> '+ str(v) + '\n')
        my_file.write("\n")

        my_file.write("NN Structure Params:"+"\n")
        for k, v in nn_params.items():
            my_file.write(str(k) + ' >>> '+ str(v) + '\n')
        my_file.write("\n")

        my_file.write("NN Train Params:"+"\n")
        for k, v in train_params.items():
            my_file.write(str(k) + ' >>> '+ str(v) + '\n')
        my_file.write("\n")


if ensemble:
    newNN = EnsembleNN(nn_params,5)
    acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

else:
    newNN = GeneralNN(nn_params)
    newNN.init_weights_orth()
    newNN.init_loss_fnc(dX,l_mean = 1,l_cov = 1) # data for std,
    acctest, acctrain = newNN.train_cust((X, U, dX), train_params)


# plot
min_err = np.min(acctrain)
min_err_test = np.min(acctest)

if log:
    with open('_training_logs/'+'logfile' + date_str + '.txt', 'a') as my_file:
        my_file.write("Min test error: " +str(min_err_test)+ "\n")
        my_file.write("Min train error: " +str(min_err)+ "\n")

ax1 = plt.subplot(211)
# ax1.set_yscale('log')
ax1.plot(acctest, label = 'Test Accurcay')
plt.title('Test Accuracy')
ax2 = plt.subplot(212)
# ax2.set_yscale('log')
ax2.plot(acctrain, label = 'Train Accurcay')
plt.title('Training Accuracy')
ax1.legend()
plt.show()

# Saves NN params
dir_str = str('_models/temp/')
data_name = '_50Hz_try_stack1_'
info_str = "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
model_name = dir_str + date_str + info_str
newNN.save_model(model_name + '.pth')
print('Saving model to', model_name)

normX, normU, normdX = newNN.getNormScalers()
with open(model_name+"--normparams.pkl", 'wb') as pickle_file:
  pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
time.sleep(2)

# Saves data file
with open(model_name+"--data.pkl", 'wb') as pickle_file:
  pickle.dump(df, pickle_file, protocol=2)
time.sleep(2)
