# minimal file to load existing NN and continue training with new data

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

print('\n')
date_str = str(datetime.datetime.now())[:-5]
date_str = date_str.replace(' ','--').replace(':', '-')
print('Running... trainNN_RL.py' + date_str +'\n')

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


# Initialize
newNN = GeneralNN(nn_params)
newNN.init_weights_orth()
newNN.init_loss_fnc(dX,l_mean = 1,l_cov = 1) # data for std,
acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

# plot
min_err = min(acctrain)
min_err_test = min(acctest)
ax1 = plt.subplot(211)
ax1.set_yscale('log')
ax1.plot(acctest, label = 'Test Accurcay')
plt.title('Test  Train Accuracy')
ax2 = plt.subplot(212)
# ax2.set_yscale('log')
ax2.plot(acctrain, label = 'Train Accurcay')
plt.title('Training Accuracy')
ax1.legend()
plt.show()

# Saves NN params
dir_str = str('_models/temp/')
data_name = '_150Hz_newnet_'
info_str = "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
model_name = dir_str + date_str + info_str
newNN.save_model(model_name + '.pth')
print('Saving model to', model_name)

normX, normU, normdX = newNN.getNormScalers()
with open(model_name+"--normparams.pkl", 'wb') as pickle_file:
  pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
time.sleep(2)
