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

from mpl_toolkits.mplot3d import Axes3D
import csv
from enum import Enum

# load network
model_name = '_models/current_best/2018-08-23--14-21-35.9||w=150e=250lr=7e-06b=32d=2018_08_22_cf1_hover_p=True.pth'
newNN = torch.load(model_name)

with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
    normX,normU,normdX = pickle.load(pickle_file)

new_data = []
# load new data
with open('_logged_data_autonomous/Aug_24th/flight_log-20180824-153552.csv', "rb") as csvfile:
    new_data = np.loadtxt(csvfile, delimiter=",")

def stack_dir(dir):
    '''
    Takes in a directory and saves the compiled tests into one numpy .npz in
    _logged_data_autonomous/compiled/
    '''
    files = os.listdir("_logged_data_autonomous/"+dir)

    X = []
    U = []
    dX = []
    for f in files:
        # print(f)
        # with open(f, "rb") as csvfile:
        #     new_data = np.loadtxt(csvfile, delimiter=",")
        X_t, U_t, dX_t, _, _, _ = trim_load("_logged_data_autonomous/"+dir+f)
        if X == []:
            X = X_t
        else:
            X = np.append(X, X_t, axis=0)

        if U == []:
            U = U_t
        else:
            U = np.append(U, U_t, axis=0)

        if dX == []:
            dX = dX_t
        else:
            dX = np.append(dX, dX_t, axis=0)

    print('Directory: ', dir, ' has additional trimmed datapoints: ', np.shape(X)[0])
    return np.array(X), np.array(U), np.array(dX)

def trim_load(fname):
    '''
    Opens the directed csv file and returns the arrays we want
    '''
    with open(fname, "rb") as csvfile:
        new_data = np.loadtxt(csvfile, delimiter=",")
        X = new_data[:,:6]
        U = new_data[:,6:10]
        Time = new_data[:,10]
        Objv = new_data[:,11]

        # Reduces by length one for training
        dX = X[1:,:]-X[:-1,:]
        X = X[:-1,:]
        U = U[:-1,:]
        Ts = (Time[1:]-Time[:-1])/1000000   # converts deltaT to ms for easy check if data was dropped
        Objv = Objv[:-1]
        Time = Time[:-1]

        # Remove data where the timestep is wrong
        # Remove data if timestep above 10ms
        X = X[np.array(np.where(Ts < 10)).flatten(),:]
        U = U[np.array(np.where(Ts < 10)).flatten(),:]
        dX = dX[np.array(np.where(Ts < 10)).flatten(),:]
        Objv = Objv[np.array(np.where(Ts < 10)).flatten()]
        Ts = Ts[np.array(np.where(Ts < 10)).flatten()]
        Time = Time[np.array(np.where(Ts < 10)).flatten()]

        # Remove data where Ts = 0
        X = X[np.array(np.where(Ts != 0)).flatten(),:]
        U = U[np.array(np.where(Ts != 0)).flatten(),:]
        dX = dX[np.array(np.where(Ts != 0)).flatten(),:]
        Objv = Objv[np.array(np.where(Ts != 0)).flatten()]
        Ts = Ts[np.array(np.where(Ts != 0)).flatten()]
        Time = Time[np.array(np.where(Ts != 0)).flatten()]

        # remove repeated euler angles
        if True:
            X = X[np.all(X[:,3:] !=0, axis=1)]
            U = U[np.all(X[:,3:] !=0, axis=1)]
            dX = dX[np.all(X[:,3:] !=0, axis=1)]

        return np.array(X), np.array(U), np.array(dX), np.array(Objv), np.array(Ts), np.array(Time)

X_rl, U_rl, dX_rl = stack_dir("Aug_24th/")
new_data = np.array(new_data)

# manual data
data_dir = '_logged_data/pink-cf1/'
data_name = '2018_08_22_cf1_activeflight_'
Seqs_X = np.loadtxt(open(data_dir + data_name + 'Seqs_X.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)
Seqs_U = np.loadtxt(open(data_dir + data_name + 'Seqs_U.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)
Seqs_dX = Seqs_X[1:,:]-Seqs_X[:-1,:]
Seqs_X = Seqs_X[:-1]
Seqs_U = Seqs_U[:-1]

# X_merge = np.concatenate((Seqs_X, X_rl), axis=0)
# U_merge = np.concatenate((Seqs_U, U_rl), axis=0)
# continue training the network
# print(np.shape(X_merge))

w = 150     # Network width
e = 360      # number of epochs
b  = 50     # batch size
lr = 2.5e-5   # learning rate
depth = 4
prob_flag = True

# Initialize
newNN = GeneralNN(n_in_input = 4,
                    n_in_state = 6,
                    hidden_w=w,
                    n_out = 6,
                    state_idx_l=[0,1,2,3,4,5],
                    prob=prob_flag,
                    input_mode = 'Stacked Data',
                    pred_mode = 'Delta State',
                    depth=depth,
                    activation="Swish",
                    B = 1.0,
                    outIdx = [0,1,2,3,4,5],
                    dropout=0.5,
                    split_flag = True)

# Train
acc = newNN.train((X_rl, U_rl),
                    learning_rate = lr,
                    epochs=e,
                    batch_size = b,
                    optim="Adam")

plt.plot(acc)
plt.show()

# Saves NN params
dir_str = str('_models/temp_reinforced/')
date_str = str(datetime.datetime.now())[:-5]
date_str = date_str.replace(' ','--').replace(':', '-')
info_str = "||w=" + str(w) + "e=" + str(e) + "lr=" + str(lr) + "b=" + str(b) + "de=" + str(d) + "d=" + str(data_name) + "p=" + str(prob_flag)
model_name = dir_str + date_str + info_str
newNN.save_model(model_name + '.pth')

normX, normU, normdX = newNN.getNormScalers()
with open(model_name+"||normparams.pkl", 'wb') as pickle_file:
  pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
time.sleep(2)
