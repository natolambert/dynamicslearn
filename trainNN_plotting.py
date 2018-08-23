### REMAKING THE TRAINING CODE ###
# This file is to  plot a nueral network that has already bee ntrained to evaluate

__author__ = 'Nathan Lambert'
__version__ = '1.0'

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

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from enum import Enum

# load a model

# dir_str = str('_models/temp/')
# date_str = str(datetime.datetime.now())[:-5]
# info_str = "w-" + str(w) + "e-" + str(e) + "lr-" + str(lr) + "b-" + str(b) + "d-" + str(data_name) + "p-" + str(prob_flag)
# model_name = dir_str + date_str + info_str
model_name = '_models/temp/2018-08-23--14-21-35.9||w=150e=250lr=7e-06b=32d=2018_08_22_cf1_hover_p=True.pth'
# Load a NN model with:
newNN = torch.load(model_name)

# Load the scaler/dataset variables from training with:
# with open(model_name+"-normparams.pkl", 'wb') as pickle_file:
#   pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
# time.sleep(2)
with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
    normX,normU,normdX = pickle.load(pickle_file)

# Load dataset
print('...Loading Data')
data_dir = '_logged_data/pink-cf1/'
data_name = '2018_08_22_cf1_hover_'
Seqs_X = np.loadtxt(open(data_dir + data_name + 'Seqs_X.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)
Seqs_U = np.loadtxt(open(data_dir + data_name + 'Seqs_U.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)

Seqs_dX = Seqs_X[1:,:]-Seqs_X[:-1,:]
Seqs_X = Seqs_X[:-1]
Seqs_U = Seqs_U[:-1]

# remove repeated euler angles
Seqs_X = Seqs_X[np.all(Seqs_dX[:,3:] !=0, axis=1)]
Seqs_U = Seqs_U[np.all(Seqs_dX[:,3:] !=0, axis=1)]
Seqs_dX = Seqs_dX[np.all(Seqs_dX[:,3:] !=0, axis=1)]

# Seqs_X = np.expand_dims(Seqs_X, axis=0)
# Seqs_U = np.expand_dims(Seqs_U, axis=0)
# data = sequencesXU2array(Seqs_X, Seqs_U)

l = np.shape(Seqs_X)[0]
model_dims = [0,1,2,3,4,5]
plot_model((Seqs_X[:,:], Seqs_U[:,:]), newNN, 0, model_dims = model_dims, delta=True, sort = False)
# plot_model((Seqs_X, Seqs_U), newNN, 1, model_dims = model_dims, delta=True, sort = False)
# plot_model((Seqs_X, Seqs_U), newNN, 2, model_dims = model_dims, delta=True, sort = False)
# plot_model((Seqs_X, Seqs_U), newNN, 3, model_dims = model_dims, delta=True, sort = False)
plot_model((Seqs_X[:,:], Seqs_U[:,:]), newNN, 4, model_dims = model_dims, delta=True, sort = False)
# plot_model((Seqs_X, Seqs_U), newNN, 5, model_dims = model_dims, delta=True, sort = False)
plt.show()
