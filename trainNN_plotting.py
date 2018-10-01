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
import seaborn as sns
import os
import random
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
model_name = '_models/2018-09-05--17-16-57.4||w=350e=125lr=4e-05b=200de=4d=_Sept4150zp=True.pth'
# Load a NN model with:
newNN = torch.load(model_name)

# Load the scaler/dataset variables from training with:
# with open(model_name+"-normparams.pkl", 'wb') as pickle_file:
#   pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
# time.sleep(2)
with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
    normX,normU,normdX = pickle.load(pickle_file)

# Load dataset
# print('...Loading Data')
# data_dir = '_logged_data/pink-cf1/'
# data_name = '2018_08_22_cf1_hover_'
# Seqs_X = np.loadtxt(open(data_dir + data_name + 'Seqs_X.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)
# Seqs_U = np.loadtxt(open(data_dir + data_name + 'Seqs_U.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)
#
#
#
# Seqs_dX = Seqs_X[1:,:]-Seqs_X[:-1,:]
# Seqs_X = Seqs_X[:-1]
# Seqs_U = Seqs_U[:-1]
#
# # remove repeated euler angles
# Seqs_X = Seqs_X[np.all(Seqs_dX[:,3:] !=0, axis=1)]
# Seqs_U = Seqs_U[np.all(Seqs_dX[:,3:] !=0, axis=1)]
# Seqs_dX = Seqs_dX[np.all(Seqs_dX[:,3:] !=0, axis=1)]

# print(len(Seqs_X))
# pitch = Seqs_X[:,3]
# roll = Seqs_X[:,4]
# MSE = ((pitch)**2+(roll**2)).mean()
# print(MSE)
# quit()

# Seqs_X = np.expand_dims(Seqs_X, axis=0)
# Seqs_U = np.expand_dims(Seqs_U, axis=0)
# data = sequencesXU2array(Seqs_X, Seqs_U)

# l = np.shape(Seqs_X)[0]
# model_dims = [0,1,2,3,4,5]
'''
Some good file timestamps:
sep14_150_2:
- 121857
- 122904
- 130527
- 131340
'''
fname ='_logged_data_autonomous/sep14_150_3/flight_log-20180914-182941.csv'
X, U, dX, _, Ts, _ = trim_load_delta(fname, input_stack = 4, takeoff=True)
# plot
if False:
    # U[:25,:] = 0
    # NOTE the takeoff_num in utils_data is 105
    font = {'size'   : 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    # plt.tight_layout()

    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)


    my_dpi = 96
    plt.figure(figsize=(3200/my_dpi, 4000/my_dpi), dpi=my_dpi)
    n = len(X[:,3])
    scaled_time = np.linspace(0,n,n)*6.66
    print(scaled_time)
    # ax1.set_title('Example Flight Performance')
    # plt.title('Autonomous Flight Data')

    # ax2.axvline(182.5, linestyle = ':', linewidth = 1.25, color = 'k')
    # ax1.axvline(370, linestyle = ':', linewidth = 1.25, color = 'k')
    # ax2.axvline(370, linestyle = ':', linewidth = 1.25, color = 'k')
    ax1.plot(scaled_time, U[:,0], label= 'm1', alpha =.8)
    ax1.plot(scaled_time, U[:,1], label= 'm2', alpha =.8)
    ax1.plot(scaled_time, U[:,2], label= 'm3', alpha =.8)
    ax1.plot(scaled_time, U[:,3], label= 'm4', alpha =.8)
    ax1.set_ylim([0,65000])
    ax1.set_ylabel('Motor Power (PWM)')

    ax2.set_ylim([-45,45])
    ax2.set_ylabel('Euler Angles (Deg)')
    ax2.set_xlabel('Time (ms)')

    ax2.plot(scaled_time, X[:,3], label = 'Pitch')
    ax2.plot(scaled_time, X[:,4], label = 'Roll')
    # ax2.plot(scaled_time, X[:,5]-X[0,5])
    # ax4.plot(scaled_time, X[:,5],color='y')
    # ax4.set_ylabel('Angular Accel (deg/s^2)')
    # ax4.set_ylim([-400,400])

    leg1 = ax1.legend(ncol=4, loc=4)
    leg2 = ax2.legend(loc=8, ncol=2)#, 'Yaw'])
    for line in leg1.get_lines():
        line.set_linewidth(2.5)

    for line in leg2.get_lines():
        line.set_linewidth(2.5)
    # ax1.grid(True, ls= 'dashed')
    # # ax2.grid(True, ls= 'dashed')
    # ax3.grid(True, ls= 'dashed')
    # ax4.grid(True, ls= 'dashed')

    # # Textbox to label lines
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # # arrowprops = dict(width='1')
    # textstr = 'Euler Angle Response'
    # # place a text box in upper left in axes coords
    # ax1.annotate(textstr, (385,17000), (510,17000), arrowprops=dict(arrowstyle='->'), bbox=props)
    # ax2.annotate(
    #     '', xy=(182.5, 10), xycoords='data',
    #     xytext=(370,10), textcoords='data',
    #     arrowprops={'arrowstyle': '<->'})
    # ax2.annotate(
    #     'Motor Step Response', xy=(250, 11.5), xycoords='data',
    #     xytext=(400,25), textcoords='data',
    #     arrowprops={'arrowstyle': '-'}, bbox=props)
    # ax1.annotate(textstr, (335,6000), (500,8000), arrowprops=dict(arrowstyle='->'), bbox=props)
    # ax2.text(0.2, .95, textstr, transform=ax2.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props)
    # ax2.arrow(0.2, .95, -.1, .2)

    plt.show()


if False:
    #WATERFALL PLOT
    actions = np.load('_results/waterfall/actions.npy')
    curr = np.load('_results/waterfall/curr.npy')
    best_id = np.load('_results/waterfall/best_id.npy')
    predicted = np.load('_results/waterfall/predicted.npy')
    # print(actions)
    # print(curr)
    # print(best_id)
    # print(predicted)
    print(np.shape(actions))
    print(np.shape(curr))
    print(np.shape(best_id))
    print(np.shape(predicted))

    font = {'size'   : 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    # plt.tight_layout()

    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(111)

    N = np.shape(predicted)[0]
    my_dpi = 96
    plt.figure(figsize=(3200/my_dpi, 4000/my_dpi), dpi=my_dpi)
    dim = 4
    pred = predicted[:,:,dim]
    curr_exp_dim = curr[dim]*np.ones((N,1))

    data = np.hstack((curr_exp_dim, pred))
    for traj in data:
        ax1.plot(traj, linestyle = ':', linewidth = 1.6)

    ax1.plot(data[best_id,:],linestyle = '-', linewidth=3, color='r')

    ax1.set_ylabel('Roll (deg)')
    ax1.set_xlabel('Timestep (T)')
    ax1.set_xticks(np.arange(0,5.1,1))
    ax1.set_xticklabels(["s(t)", "1", "2", "3", "4", "5"])
    plt.show()

if True:
    # plot showing 10 random flights from each rollout in 1 dimension
    font = {'size'   : 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    # plt.tight_layout()
    def update_line(hl, new_data_x, new_data_y):
        hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
        hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
        plt.draw()

    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(111)

    line1, = ax1.plot([],[])

    dim =3

    dir1 = "_logged_data_autonomous/top20/roll0/"
    dir2 = "_logged_data_autonomous/top20/roll1/"
    dir3 = "_logged_data_autonomous/top20/roll2/"
    dir4 = "_logged_data_autonomous/top20/roll3/"
    dir5 = "_logged_data_autonomous/top20/roll4/"
    dir6 = "_logged_data_autonomous/top20/roll5/"
    dirs = [dir1, dir2, dir3, dir4, dir5, dir6]
    colors = ['r', 'y', 'g', 'c', 'b', 'm']
    best_len = 0
    best_time = 2000
    for k, dir in enumerate(dirs):
        for i in range(15):
            # file = random.choice(os.listdir(dir))
            file = os.listdir(dir)[i]
            print(file)
            print('Processing File: ', file, 'Dir: ', k, 'File number: ',i)
            if dir == dir4 or dir == dir5 or dir == dir6:
                 takeoff = True
            else:
                 takeoff = False
            X, U, dX, _, Ts, time = trim_load_delta(str(dir+file), input_stack = 0, takeoff=takeoff)
            n = len(X[:,dim])
            if n > best_len:
                best_len = n
            if time > best_time:
                best_time = time
                print(best_time)
            x = np.linspace(0,time,len(Ts))
            if i == 0:
                plt.plot(x, X[:,dim], linestyle ='-', alpha =.5, linewidth = 1.2, color = colors[k], label="Rollout %d" % k)
            else:
                plt.plot(x, X[:,dim], linestyle ='-', alpha =.5, linewidth = 1.2, color = colors[k])

    # print(x)
    # scaled_time = np.round(np.arange(0, best_time, best_time/best_len),1)
    # print(scaled_time)
    # ax1.set_xticks(scaled_time[0::10])
    # ax1.set_xticklabels([str(x) for x in scaled_time[::10]])
    leg = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol = 6)#len(dirs))
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    ax1.set_xlabel("Time (ms)")
    ax1.set_xlim([0,best_time])
    ax1.set_ylim([-40,40])
    ax1.set_ylabel("Pitch (Deg)")
    plt.show()


# Framework for comparing models
if False:

    model_1 = '_models/current_best/2018-09-05--12-24-30.4||w=250e=125lr=3.5e-05b=200de=4d=_Sept4150zp=True.pth'
    # Load a NN model with:
    nn1 = torch.load(model_1)
    nn1.training = False
    nn1.eval()
    with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
        normX1,normU1,normdX1 = pickle.load(pickle_file)

    model_2 = '_models/temp_reinforced/2018-08-29--14-23-24.3||w=150e=100lr=0.0005b=150de=3d=_AUG29_125RL2p=True.pth'
    model_2 = '_models/temp/2018-08-30--15-51-50.3||w=350e=30lr=2.5e-05b=100de=3p=Falseda=2018_08_22_cf1_activeflight_.pth'
    # Load a NN model with: _models/temp/2018-08-30--15-51-50.3||w=350e=30lr=2.5e-05b=100de=3p=Falseda=2018_08_22_cf1_activeflight_.pth
    nn2 = torch.load(model_2)
    nn2.training = False
    nn2.eval()
    with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
        normX2,normU2,normdX2 = pickle.load(pickle_file)

    model_3 = '_models/temp_reinforced/2018-08-29--14-26-00.5||w=150e=100lr=0.0005b=150de=3d=_AUG29_125RL2p=True.pth'
    # Load a NN model with:
    nn3 = torch.load(model_3)
    nn3.training = False
    nn3.eval()
    with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
        normX2,normU2,normdX2 = pickle.load(pickle_file)

    # dxs = Seqs_X[1:,:]-Seqs_X[:-1,:]
    # xs = Seqs_X[:-1,:]
    # us = Seqs_U[:-1,:]
    #
    # # print(np.shape(dxs))
    # # print(np.shape(xs))
    # delta = True
    #
    # dxs = dxs[30000:40000, :]
    # xs = xs[30000:40000, :]
    # us = us[30000:40000, :]

    X_rl, U_rl, dX_rl = stack_dir("Aug_29th_125/")
    X_rl_2, U_rl_2, dX_rl_2 = stack_dir("Aug_29th_125_2/")

    delta = True

    xs = np.concatenate((X_rl,X_rl_2),axis=0)
    us = np.concatenate((U_rl,U_rl_2),axis=0)
    dxs = np.concatenate((dX_rl,dX_rl_2),axis=0)

    # Now need to iterate through all data and plot
    predictions_1 = np.empty((0,np.shape(xs)[1]))
    for (dx, x, u) in zip(dxs, xs, us):
        # grab prediction value
        # pred = model.predict(x,u)
        pred = predict_nn(nn1,x,u, model_dims)
        # print(np.shape(pred))
        #print('prediction: ', pred, ' x: ', x)
        if delta:
          pred = pred - x
        predictions_1 = np.append(predictions_1, pred.reshape(1,-1),  axis=0)

    Now need to iterate through all data and plot
    predictions_2 = np.empty((0,np.shape(xs)[1]))
    for (dx, x, u) in zip(dxs, xs, us):
        # grab prediction value
        # pred = model.predict(x,u)
        pred = predict_nn(nn2,x,u, model_dims)
        # print(np.shape(pred))
        #print('prediction: ', pred, ' x: ', x)
        if delta:
          pred = pred - x
        predictions_2 = np.append(predictions_2, pred.reshape(1,-1),  axis=0)

    # Now need to iterate through all data and plot
    predictions_3 = np.empty((0,np.shape(xs)[1]))
    for (dx, x, u) in zip(dxs, xs, us):
        # grab prediction value
        # pred = model.predict(x,u)
        pred = predict_nn(nn3,x,u, model_dims)
        # print(np.shape(pred))
        #print('prediction: ', pred, ' x: ', x)
        if delta:
          pred = pred - x
        predictions_3 = np.append(predictions_3, pred.reshape(1,-1),  axis=0)

    dim = 3
    # Grab correction dimension data
    if delta:
        ground_dim = dxs[:, dim]
    else:
        ground_dim = xs[:,dim]
    pred_dim_1 = predictions_1[:, dim]
    # pred_dim_2 = predictions_2[:, dim]
    # pred_dim_3 = predictions_3[:, dim]

    # Sort with respect to ground truth
    sort = False
    if sort:
        data = zip(ground_dim,pred_dim_1)#,pred_dim_2, pred_dim_3)
        data = sorted(data, key=lambda tup: tup[0])
        # ground_dim_sort, pred_dim_sort_1, pred_dim_sort_2, pred_dim_sort_3 = zip(*data)
        # ground_dim_sort, pred_dim_sort_1 = zip(*data)
        ground_dim_sort, pred_dim_sort_1 = zip(*sorted(zip(ground_dim,pred_dim_1)))
        # _,               pred_dim_sort_2 = zip(*sorted(zip(ground_dim,pred_dim_2)))
        # _,               pred_dim_sort_3 = zip(*sorted(zip(ground_dim,pred_dim_3)))
