# Our infrastucture files
from utils_data import *

# data packages
import pickle
import random

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
import copy

# Plotting
import matplotlib.pyplot as plt
import matplotlib

import argparse

'''
Some notes on the crazyflie PID structure. Essentially there is a trajectory planner
  that we can ignore, and a Attitude control that sents setpoints to a rate controller.
  The Attitude controller outputs a rate desired, and the rate desired updates motors

This is the code from the fimrware. You can see how the m1...m4 pwm values are set
  The motorPower.m1 is a pwm value, and limit thrust puts it in an range:
    motorPower.m1 = limitThrust(control->thrust + control->pitch +
                               control->yaw);
    motorPower.m2 = limitThrust(control->thrust - control->roll -
                               control->yaw);
    motorPower.m3 =  limitThrust(control->thrust - control->pitch +
                               control->yaw);
    motorPower.m4 =  limitThrust(control->thrust + control->roll -
                               control->yaw);

    This shows that m1 and m3 control pitch while m2 and m4 control roll.
    Yaw should account for a minor amount of this. Our setpoint will be easy,
    roll, pitch =0 ,yaw rate = 0.

Default values, for 250Hz control. Will expect our simulated values to differ:
Axis Mode: [KP, KI, KD, iLimit]

Pitch Rate: [250.0, 500.0, 2.5, 33.3]
Roll Rate: [250.0, 500.0, 2.5, 33.3]
Yaw Rate: [120.0, 16.7, 0.0, 166.7]
Pitch Attitude: [6.0, 3.0, 0.0, 20.0]
Roll Attitude: [6.0, 3.0, 0.0, 20.0]
Yaw Attitude: [6.0, 1.0, 0.35, 360.0]
'''

######################################################################
def main():
    # adding arguments to make code easier to work with
    parser = argparse.ArgumentParser(description='Engineer PID tuning off learned dynamics model.')
    parser.add_argument('dimension', type=str,
                        choices = ['all', 'pitch', 'roll'],
                        help='choose which dimension to tune PID for.')
    parser.add_argument('--log', action='store_true',
                        help='a flag for storing a training log in a txt file')
    parser.add_argument('--noprint', action='store_false',
                        help='turn off printing in the terminal window for epochs')
    parser.add_argument('--plot', action='store_true',
                        help='plots information for easy analysis')

    args = parser.parse_args()

    dim_s = args.dimension
    if dim_s == 'all':
        dim = 0
    elif dim_s == 'pitch':
        dim = 1
    elif dim_s == 'roll':
        dim = 2

    log = args.log
    noprint = args.noprint

    ######################################################################



    # Code outline to predict states given an action or control scheme

    # load model

    # load initial state or generate.
    load_params ={
        'delta_state': True,
        'takeoff_points': 180,
        'trim_0_dX': True,
        'trime_large_dX': True,
        'bound_inputs': [20000,65500],
        'stack_states': 4,
        'collision_flag': False,
        'shuffle_here': False,
        'timestep_flags': [],
        'battery' : True
    }

    dir_list = ["_newquad1/150Hz_rand/"]
    other_dirs = ["150Hz/sep13_150_2/","/150Hzsep14_150_2/","150Hz/sep14_150_3/"]
    df = load_dirs(dir_list, load_params)

    data_params = {
        'states' : [],
        'inputs' : [],
        'change_states' : [],
        'battery' : True
    }

    X, U, dX = df_to_training(df, data_params)




    # generate actions / controller

    # init some variables
    T = 150
    stack_states = 4            # declared above
    x_pred = np.zeros(9)        # a storage state vector
    x_pred_stacked = np.zeros(stack_states*9)      # need a stacked vector to pass into network
    u_stacked = np.zeros(stack_states*4)

    # some reasonable seeds:
    # 150hz: 3400, 485, 3850

    # 150Hz models`
    model_single = '_models/temp/2018-10-05--15-42-42.8--Min error-782.69296875d=_150Hz_newnet_.pth'

    model_ensemble = '_models/temp/2018-10-05--15-43-07.5--Min error-787.556328125d=_150Hz_newnet_.pth'

    model = model_ensemble

    # PID Params
    kp = 100
    ki = 500
    kd = 2.5
    ilimit = 250
    outlimit = 5000
    dt = 1/150
    PWMequil = np.array([31687.1, 37954.7, 33384.8, 36220.11]) # new quad

    for i in range(1):
        # Lets simulate some actions
        pid_roll = PID(0, kp, ki, kd, ilimit, outlimit, dt, samplingRate=-1, cutoffFreq = -1, enableDFilter = False)

        num_pts = np.shape(X)[0]
        point = random.randint(0,num_pts)
        print("`Seed` is: ", point)
        x0 = X[point,:]
        action = U[point:point+T+1,:]
        # action = U[point,:]
        # action = np.tile([45000,45000,30000,30000],4)
        # action = np.concatenate((action,[3900]))
        # print(x0)
        nn_single = torch.load(model_single)
        nn_single.eval()

        nn_ensemble = torch.load(model_ensemble)
        nn_ensemble.eval()

        x_stored_PID = pred_traj(x0, pid_roll, nn_ensemble, T)
        x_stored_past = pred_traj(x0, action, nn_ensemble, T)

        font = {'size'   : 18}

        matplotlib.rc('font', **font)
        matplotlib.rc('lines', linewidth=2.5)

        # plt.tight_layout()

        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot(111)
            # ax2 = plt.subplot(212)

        dim = 4
        plt.title("Comparing Model to Ground Truth")
        ax1.set_ylim([-35,35])
        # ax2.set_ylim([-35,35])
        ax1.plot(x_stored_PID[:,dim], linestyle = '--', color='b', label ='Predicted PID')
        ax1.plot(x_stored_past[:,dim], linestyle = '--', color='g', label ='Predicted Past')
        ax1.plot(X[point:point+T+1,dim], color = 'k', label = 'Ground Truth')
        ax1.legend()
        # ax2.plot(X[point:point+T+1,3:5])
        plt.show()


# TODO: implement a structured PID similar to the crazyflie to see if it helps performance



if __name__ == "__main__":
    print('\n---------------------------------------------------')
    print('Running file generatePID')
    print('\n')
    main()
