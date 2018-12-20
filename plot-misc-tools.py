### REMAKING THE TRAINING CODE ###
# This file is to  plot a nueral network that has already bee ntrained to evaluate

__author__ = 'Nathan Lambert'
__version__ = '1.0'

import pickle
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
# model_name = '_models/2018-09-05--17-16-57.4||w=350e=125lr=4e-05b=200de=4d=_Sept4150zp=True.pth'
# Load a NN model with:
# newNN = torch.load(model_name)

# Load the scaler/dataset variables from training with:
# with open(model_name+"-normparams.pkl", 'wb') as pickle_file:
#   pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
# time.sleep(2)
# with open(model_name[:-4]+'||normparams.pkl', 'rb') as pickle_file:
#     normX,normU,normdX = pickle.load(pickle_file)

'''
Some good file timestamps:
sep14_150_2:
- 121857
- 122904
- 130527
- 131340
'''
fname ='_logged_data_autonomous/sep14_150_3/flight_log-20180914-182941.csv'                 # original for ICRA
fname = '_logged_data_autonomous/_newquad1/publ2/c50_roll06/flight_log-20181115-101931.csv' #file for arxiv draftt
# file for arxiv draftt
# fname = '_logged_data_autonomous/mf_ex/flight_log-20181211-184531.csv'

# finding video plot
# fname = '_logged_data_autonomous/_newquad1/fixed_samp/c50_samp300_roll3/flight_log-20181029-151853.csv'
# fname = '_logged_data_autonomous/_newquad1/fixed_samp/c100_samp275_misc/flight_log-20181031-132327.csv'
# fname = '_logged_data_autonomous/_newquad1/fixed_samp/c50_samp300_roll3/flight_log-20181029-152355.csv'
# X, U, dX, _, Ts, _ = trim_load_delta(fname, input_stack = 4, takeoff=True) flight_log-20181029-152355

load_params = {
    'delta_state': True,                # normally leave as True, prediction mode
    # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
    'include_tplus1': True,
    # trims high vbat because these points the quad is not moving
    'trim_high_vbat': 4400,
    # If not trimming data with fast log, need another way to get rid of repeated 0s
    'takeoff_points': 180,
    # if all the euler angles (floats) don't change, it is not realistic data
    'trim_0_dX': True,
    'find_move': True,
    # if the states change by a large amount, not realistic
    'trime_large_dX': True,
    # Anything out of here is erroneous anyways. Can be used to focus training
    'bound_inputs': [15000, 65500],
    # IMPORTANT ONE: stacks the past states and inputs to pass into network
    'stack_states': 3,
    # looks for sharp changes to tthrow out items post collision
    'collision_flag': False,
    # shuffle pre training, makes it hard to plot trajectories
    'shuffle_here': False,
    'timestep_flags': [],               # if you want to filter rostime stamps, do it here
    'battery': True,                   # if battery voltage is in the state data
    # adds a column to the dataframe tracking end of trajectories
    'terminals': True,
    'fastLog': False,                   # if using the software with the new fast log
    # Number of times the control freq you will be using is faster than that at data logging
    'contFreq': 1
}
X, U, dX, objv, Ts, time, terminal = trim_load_param(fname, load_params)
# plot
if False:
    # U[:25,:] = 0
    # NOTE the takeoff_num in utils_data is 105
    font = {'size'   : 22}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=5)


    with sns.axes_style("whitegrid"):
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

    # plt.tight_layout()

    # for video figure
    # my_dpi = 200
    # fig = plt.figure(figsize=(3.5*1920/my_dpi, 2*560/my_dpi), dpi=my_dpi)
    # with sns.axes_style("whitegrid"):
    #     plt.rcParams["axes.edgecolor"] = "0.15"
    #     plt.rcParams["axes.linewidth"] = 1.5
    #     ax2 = plt.subplot(111)
        # ax2 = plt.subplot(212)

    print(np.max(time))
    
    n = len(X[:,3])
    scaled_time = np.linspace(0,n,n)*20/5/1000
    # ax1.set_title('Example Flight Performance')
    # plt.title('Autonomous Flight Data')

    shorter = int(n/5)
    # ax1.plot(scaled_time, U[:,0], label= 'm1', alpha =.8)
    # ax1.plot(scaled_time, U[:,1], label= 'm2', alpha =.8)
    # ax1.plot(scaled_time, U[:,2], label= 'm3', alpha =.8)
    # ax1.plot(scaled_time, U[:,3], label= 'm4', alpha =.8)

    ax1.plot(scaled_time[:shorter], U[:shorter, 0], label='m1', 
            alpha=.8, markevery=20, marker='.', markersize='20')
    ax1.plot(scaled_time[:shorter], U[:shorter, 1], label='m2', 
             alpha=.8, markevery=20, marker='*', markersize='20')
    ax1.plot(scaled_time[:shorter], U[:shorter, 2], label='m3',
             alpha=.8, markevery=20,  marker='^', markersize='20')
    ax1.plot(scaled_time[:shorter], U[:shorter, 3], label='m4',
             alpha=.8, markevery=20, marker='1', markersize='20')
    ax1.set_ylim([20000,57000])

    ax1.set_ylabel('Motor Power (PWM)')

    ax2.set_ylim([-30,30])
    ax2.set_ylim([-25,25])
    ax2.set_ylabel('Euler Angles (Deg)')
    # ax2.set_xlabel('Time (s)')
    fig.text(.44,.00,'Time (s)')

    # ax2.plot(scaled_time, X[:, 3], label='Pitch', marker='.', markevery = 25, markersize='20')
    # ax2.plot(scaled_time, X[:, 4], label='Roll',
    #          marker='^', markevery= 25,  markersize='20')

    ax2.plot(scaled_time[:shorter], X[:shorter, 3],
             label='Pitch', markevery=20, marker='.', markersize='20')
    ax2.plot(scaled_time[:shorter], X[:shorter, 4],
             label='Roll', markevery=20,  marker='^', markersize='20')
    # ax2.plot(scaled_time, X[:,5]-X[0,5])
    # ax4.plot(scaled_time, X[:,5],color='y')
    # ax4.set_ylabel('Angular Accel (deg/s^2)')
    # ax4.set_ylim([-400,400])

    leg1 = ax1.legend(ncol=4, loc=0)
    leg2 = ax2.legend(loc=8, ncol=2, frameon=True)#, 'Yaw'])

    ax2.grid(b=True, which='major', color='k', linestyle='-', linewidth=1, alpha=.5)

    # for line in leg1.get_lines():
    #     line.set_linewidth(2.5)

    # for line in leg2.get_lines():
    #     line.set_linewidth(2.5)
    # ax1.grid(True, ls= 'dashed')
    # # ax2.grid(True, ls= 'dashed')
    # ax3.grid(True, ls= 'dashed')
    # ax4.grid(True, ls= 'dashed')

    # plt.subplots_adjust(wspace=.15, left=.07, right=1-.07)  # , hspace=.15)


    # plt.savefig('_results/poster', edgecolor='black', dpi=my_dpi, transparent=False)

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
    matplotlib.rc('lines', linewidth=3)



    fig = plt.figure()
    with sns.axes_style("whitegrid"):
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

    axes = [ax1,ax2,ax3,ax4]
    plt.tight_layout()

    line1, = ax1.plot([],[])

    dim =3

    # original plot
    dir1 = "_logged_data_autonomous/_examples/icra-top20/roll0/"
    dir2 = "_logged_data_autonomous/_examples/icra-top20/roll1/"
    dir3 = "_logged_data_autonomous/_examples/icra-top20/roll2/"
    dir4 = "_logged_data_autonomous/_examples/icra-top20/roll3/"
    dir5 = "_logged_data_autonomous/_examples/icra-top20/roll4/"
    dir6 = "_logged_data_autonomous/_examples/icra-top20/roll5/"

    #new plot
    dir1 = "_logged_data_autonomous/_newquad1/publ2/c50_rand/"
    dir2 = "_logged_data_autonomous/_newquad1/publ2/c50_roll01/"
    dir3 = "_logged_data_autonomous/_newquad1/publ2/c50_roll02/"
    dir4 = "_logged_data_autonomous/_newquad1/publ2/c50_roll03/"
    dir5 = "_logged_data_autonomous/_newquad1/publ2/c50_roll04/"
    # dir6 = "_logged_data_autonomous/_newquad1/publ2/c50_roll05/"
    # dir7 = "_logged_data_autonomous/_newquad1/publ2/c50_roll06/"


    dirs = [dir1, dir2, dir3, dir4]#, dir5]#, dir6]#, dir7]
    colors = ['r', 'y', 'g', 'c']#, 'b']#, 'm']#, 'k' ]
    colors = ['r', 'b', 'g', 'k']#, 'b']#, 'm']#, 'k' ]
    best_len = 0
    best_time = 3000

    load_params = {
        'delta_state': True,                # normally leave as True, prediction mode
        # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
        'include_tplus1': True,
        # trims high vbat because these points the quad is not moving
        'trim_high_vbat': 4200,
        # If not trimming data with fast log, need another way to get rid of repeated 0s
        'takeoff_points': 180,
        # if all the euler angles (floats) don't change, it is not realistic data
        'trim_0_dX': True,
        'find_move': True,
        # if the states change by a large amount, not realistic
        'trime_large_dX': True,
        # Anything out of here is erroneous anyways. Can be used to focus training
        'bound_inputs': [25000, 65500],
        # IMPORTANT ONE: stacks the past states and inputs to pass into network
        'stack_states': 3,
        # looks for sharp changes to tthrow out items post collision
        'collision_flag': False,
        # shuffle pre training, makes it hard to plot trajectories
        'shuffle_here': False,
        'timestep_flags': [],               # if you want to filter rostime stamps, do it here
        'battery': True,                   # if battery voltage is in the state data
        # adds a column to the dataframe tracking end of trajectories
        'terminals': True,
        'fastLog': True,                   # if using the software with the new fast log
        # Number of times the control freq you will be using is faster than that at data logging
        'contFreq': 1
    }

    # load_params ={
    #     'delta_state': True,
    #     'takeoff_points': 0,
    #     'trim_0_dX': True,
    #     'trime_large_dX': True,
    #     'bound_inputs': [20000,65500],
    #     'stack_states': 4,
    #     'collision_flag': False,
    #     'shuffle_here': False,
    #     'timestep_flags': [],
    #     'battery' : False
    # }

    for k, dir in enumerate(dirs):
        axis = axes[k]
        for i in range(10):
            # file = random.choice(os.listdir(dir))
            file = os.listdir(dir)[i]
            print(file)
            print('Processing File: ', file, 'Dir: ', k, 'File number: ',i)
            if dir == dir4 or dir == dir5 or dir == dir6:
                 takeoff = True
                 load_params['takeoff_points'] = 170
            else:
                 takeoff = False


            X, U, dX, objv, Ts, times, terminal = trim_load_param(str(dir+file), load_params)

            time = np.max(times)
            n = len(X[:,dim])
            if n > best_len:
                best_len = n
            if time > best_time:
                best_time = time
                print(best_time)
            x = np.linspace(0,time,len(Ts))
            if i == 0:
                axis.plot(x/1000, X[:,dim], linestyle ='-', alpha =.5, linewidth = 3, color = colors[k], label="Rollout %d" % k)
                # axis.plot(x, X[:, dim], color='k', linestyle='-')
            else:
                axis.plot(x/1000, X[:, dim], linestyle='-', alpha=.5,
                          linewidth=3, color=colors[k])
                # axis.plot(x, X[:, dim], color='k', linestyle='-')

    # print(x)
    # scaled_time = np.round(np.arange(0, best_time, best_time/best_len),1)
    # print(scaled_time)
    # ax1.set_xticks(scaled_time[0::10])
    # ax1.set_xticklabels([str(x) for x in scaled_time[::10]])
    # leg = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol = 6)#len(dirs))
    # # get the individual lines inside legend and set line width
    # for line in leg.get_lines():
    #     line.set_linewidth(2.5)

    # ax1.set_xlabel("Time (ms)")
    # ax1.set_xlim([0,best_time])
    # ax1.set_xlim([0,2000])
    # ax1.set_ylim([-40,40])
    # ax1.set_ylabel("Pitch (Deg)")

    #Version two of the figure
    for i, ax in enumerate(axes):
        if i >1: ax.set_xlabel("Time (s)")
        ax.set_xlim([0, best_time])
        ax.set_xlim([0, 2.500])
        ax.set_ylim([-40, 40])
        if i == 0 or i == 2:
            ax.set_ylabel("Pitch (Deg)")

        ax.grid(b=True, which='major', color='k', linestyle='-', linewidth=1, alpha=.5)
        # grid(b=True, which='minor', color='r', linestyle='--')

    plt.subplots_adjust(wspace=.15, left = .07, right = 1-.07)#, hspace=.15)
    ax1.set_title("Random Controller Flights")
    ax2.set_title("After 1 Model Iteration")
    ax3.set_title("After 2 Model Iterations")
    ax4.set_title("After 3 Model Iterations")
    # plt.suptitle("Comparision of Flight Lengths in Early Rollouts")

    # fig.set_size_inches(15, 7)


    # plt.savefig('psoter', edgecolor='black', dpi=100, transparent=True)


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

    # Now need to iterate through all data and plot
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
