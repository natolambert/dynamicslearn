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
import matplotlib

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

# plot
if False:
    font = {'family' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=3)

    start_time = Time[0]
    scaled_time = (Time[:]-start_time)/1000000

    ax1 = plt.subplot(311)
    plt.title('Autonomous Flight Data')

    ax1.plot(scaled_time, X[:,3:])
    ax1.set_ylim([-60,60])
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_xlabel('Time (ms)')

    # ax2 = ax1.twinx()
    ax4 = plt.subplot(312)
    ax4.plot(scaled_time, X[:,0],color='m')
    ax4.plot(scaled_time, X[:,1],color='c')
    ax4.plot(scaled_time, X[:,2],color='y')
    ax4.set_ylabel('Angular Accel (deg/s^2)')
    ax4.set_ylim([-400,400])

    ax1.legend(['pitch', 'roll', 'yaw'],loc=2)
    ax4.legend(['omega_x','omega_y','omega_z'],loc=2)

    ax3 = plt.subplot(313)
    ax3.plot(scaled_time, Objv, color='r')
    ax3.set_ylabel('Objective Function Value')
    ax3.set_ylim([0,1000000])
    ax3.set_xlabel('Time (ms)')

    plt.show()

# Figure in paper comparing deterministc vs probablistc network
if True:

    model_prob = '_models/temp/2018-08-23--14-21-35.9||w=150e=250lr=7e-06b=32d=2018_08_22_cf1_hover_p=True.pth'
    # Load a NN model with:
    newNNprob = torch.load(model_prob)

    # Load the scaler/dataset variables from training with:
    # with open(model_name+"-normparams.pkl", 'wb') as pickle_file:
    #   pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
    # time.sleep(2)
    with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
        normXprob,normUprob,normdXprob = pickle.load(pickle_file)

    model_det = '_models/temp/2018-08-26--14-30-27.0||w=150e=80lr=7e-06b=32de=3p=Falseda=2018_08_22_cf1_hover_.pth'
    # Load a NN model with:
    newNNdet = torch.load(model_det)

    # Load the scaler/dataset variables from training with:
    # with open(model_name+"-normparams.pkl", 'wb') as pickle_file:
    #   pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
    # time.sleep(2)
    with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
        normXdet,normUdet,normdXdet = pickle.load(pickle_file)

    dxs = Seqs_X[1:,:]-Seqs_X[:-1,:]
    xs = Seqs_X[:-1,:]
    us = Seqs_U[:-1,:]

    # print(np.shape(dxs))
    # print(np.shape(xs))
    delta = True

    dxs = dxs[30000:40000, :]
    xs = xs[30000:40000, :]
    us = us[30000:40000, :]

    # Now need to iterate through all data and plot
    predictions_prob = np.empty((0,np.shape(xs)[1]))
    for (dx, x, u) in zip(dxs, xs, us):
        # grab prediction value
        # pred = model.predict(x,u)
        pred = predict_nn(newNNprob,x,u, model_dims)
        # print(np.shape(pred))
        #print('prediction: ', pred, ' x: ', x)
        if delta:
          pred = pred - x
        predictions_prob = np.append(predictions_prob, pred.reshape(1,-1),  axis=0)

    # Now need to iterate through all data and plot
    predictions_det = np.empty((0,np.shape(xs)[1]))
    for (dx, x, u) in zip(dxs, xs, us):
        # grab prediction value
        # pred = model.predict(x,u)
        pred = predict_nn(newNNdet,x,u, model_dims)
        # print(np.shape(pred))
        #print('prediction: ', pred, ' x: ', x)
        if delta:
          pred = pred - x
        predictions_det = np.append(predictions_det, pred.reshape(1,-1),  axis=0)

    dim = 3
    # Grab correction dimension data
    ground_dim = dxs[:, dim]
    pred_dim_prob = predictions_prob[:, dim]
    pred_dim_det = predictions_det[:, dim]

    # Sort with respect to ground truth
    # sort = True
    # if sort:
    #   ground_dim_sort, pred_dim_sort_prob = zip(*sorted(zip(ground_dim,pred_dim_prob)))
    #   _,               pred_dim_sort_det = zip(*sorted(zip(ground_dim,pred_dim_det)))
    # else:
    #   ground_dim_sort, pred_dim_sort = ground_dim, pred_dim

    font = {'family' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=1.5)

    ax1 = plt.subplot(211)
    plt.title('Comparison of Deterministic and Probablistic One-Step Predictions')

    x = np.size(ground_dim)
    print(x)
    time = np.linspace(0, int(x)-1, int(x))
    ts = 4**-3
    time = [t*ts for t in time]
    ax1.plot(time, ground_dim, label='Ground Truth')
    ax1.plot(time, pred_dim_prob, label='Model Prediction')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Roll (Degrees)')
    ax1.set_xlim([125,128])
    ax1.set_ylim([-.6,.6])
    ax1.legend()

    ax2 = plt.subplot(212)
    ax2.plot(time, ground_dim, label='Ground Truth')
    ax2.plot(time, pred_dim_det, label='Deterministic Prediction')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Roll (Degrees)')
    ax2.set_xlim([125,128])
    ax2.set_ylim([-.6,.6])

    print('...Plotting')
    # Plot
    plt.show()

# plot_model((Seqs_X[:,:], Seqs_U[:,:]), newNN, 0, model_dims = model_dims, delta=True, sort = False)
