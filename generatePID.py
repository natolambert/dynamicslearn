# Our infrastucture files
from utils_data import *
from utils_sim import *
from pid import *

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
    # some reasonable seeds:
    # 150hz: 3400, 485, 3850
    # 50Hz  models`
    model_single_50_nobat = '_models/temp/2018-11-09--10-52-58.0_50hz_nobat_.pth'
    model_single_50 = '_models/temp/2018-11-09--10-55-12.9_50hz_withbat_stack3_.pth'

    # 100Hz models`
    model_single = '_models/temp/2018-11-09--10-48-05.2_100hz_bat_trimmed_.pth'
    model_single_nobat = '_models/temp/2018-11-09--10-46-56.7_100hz_bat_trimmed_.pth'

    # intitial trained
    model_50_truestate = '_models/temp/2018-11-20--12-32-18.5_c50_true_stack3_.pth'
    # trained on more takeoff points
    model_50_truestate2 = '_models/temp/2018-11-20--12-41-59.8_c50_true2_stack3_.pth'
    # ensemble
    model_50_true_ensemble ='_models/temp/2018-11-20--12-40-00.4_c50_trueaccel_ensem_stack3_.pth'

    #25hz true state
    model_25 = '_models/temp/2018-11-20--12-55-45.6_c25_true_stack3_.pth'

    nn = torch.load(model_single_50)
    nn.eval()

   
    '''
    RMSES
    100Hz no bat: [6.3130932  5.8711117  2.20496666 0.34913389 0.40612083 0.2069013 0.59639823 0.82310533 0.52682769]
    100Hz with bat: [6.31994656 5.92266344 2.09359516 0.35665752 0.41530902 0.2053187 0.59421848 0.80986195 0.52297999]

    50Hz no bat: [10.21272868 11.03302467  2.83940086  0.51089872  0.44949222  0.24415286 0.64122414  1.05109164  0.53779281]
    50Hz with bat: [10.2567613  11.142001    2.86625326  0.52282885  0.44307055  0.24636216 0.63476385  1.06089197  0.53843009]

    '''

    state_list, input_list, change_list = nn.get_training_lists()

    # load initial state or generate.
    load_params ={
        'delta_state': True,                # normally leave as True, prediction mode
        'include_tplus1': True,
        'find_move': True,
        'takeoff_points': 180,              # If not trimming data with fast log, need another way to get rid of repeated 0s
        'trim_high_vbat': 4050,             # trims high vbat because these points the quad is not moving
        'trim_0_dX': True,                  # if all the euler angles (floats) don't change, it is not realistic data
        'trime_large_dX': True,             # if the states change by a large amount, not realistic
        'bound_inputs': [25000,65500],      # Anything out of here is erroneous anyways. Can be used to focus training
        'stack_states': 4,                  # IMPORTANT ONE: stacks the past states and inputs to pass into network
        'collision_flag': False,            # looks for sharp changes to tthrow out items post collision
        'shuffle_here': False,              # shuffle pre training, makes it hard to plot trajectories
        'timestep_flags': [],               # if you want to filter rostime stamps, do it here
        'battery' : True,                   # if battery voltage is in the state data
        'terminals': True,                 # adds a column to the dataframe tracking end of trajectories
        'fastLog' : True,                   # if using the software with the new fast log
        'contFreq' : 1                      # Number of times the control freq you will be using is faster than that at data logging
    }

    dir_list = ["_newquad1/fixed_samp/c100_samp300_rand/","_newquad1/fixed_samp/c100_samp300_roll1/","_newquad1/fixed_samp/c100_samp300_roll2/" ]

    dir_list = ["_newquad1/publ_data/c50_samp300_rand/",
        "_newquad1/publ_data/c50_samp300_roll1/",
        "_newquad1/publ_data/c50_samp300_roll2/",
        "_newquad1/publ_data/c50_samp300_roll3/",
        "_newquad1/publ_data/c50_samp300_roll4/"]
    # dir_list = ["_newquad1/publ_data/c25_samp300_rand/",
    #     "_newquad1/publ_data/c25_samp300_roll1/",
    #     "_newquad1/publ_data/c25_samp300_roll2/",
    #     "_newquad1/publ_data/c25_samp300_roll3/",
    #     "_newquad1/publ_data/c25_samp300_roll4/"]

# other_dirs = ["150Hz/sep13_150_2/","/150Hzsep14_150_2/","150Hz/sep14_150_3/"]
    df = load_dirs(dir_list, load_params)

    # for i in range(1):
    #     df_traj, idx = get_rand_traj(df)
    #
    #     # plot_traj_model(df_traj, nn_ensemble)
    #     plot_traj_model(df_traj, nn)
    
    # # for vbat plot for updated paper
    # nn1 = torch.load(model_single)
    # nn1.eval()
    # nn2 = torch.load(model_single_nobat)
    # nn2.eval()

    # plot_voltage_context(nn1, df, model_nobat=nn2)
    # quit()

    # for vbat plot for updated paper
    # nn1 = torch.load(model_single_50)
    # nn1.eval()
    # nn2 = torch.load(model_single_50_nobat)
    # nn2.eval()

    # plot_voltage_context(nn1, df, model_nobat=nn2)
    # quit()

    data_params = {'states' : state_list, 'inputs' : input_list, 'targets' : change_list, 'battery' : True}

    X, U, dX = df_to_training(df, data_params)

    ###########################################################################
    ######################## BELOW NO IN USE NOW ##############################

    # generate actions / controller

    # init some variables
    T = 150
    stack_states = 3           # declared above
    x_pred = np.zeros(9)        # a storage state vector
    x_pred_stacked = np.zeros(stack_states*9)      # need a stacked vector to pass into network
    u_stacked = np.zeros(stack_states*4)


    # PID Params 250.0, 500.0, 2.5, 33.3
    kp = 250
    ki = 500
    kd = 2.5
    ilimit = 33.3
    outlimit = 5000
    dt = 1/100
    PWMequil = np.array([34687.1, 37954.7, 38384.8, 36220.11]) # new quad

    for i in range(7):
        # Lets simulate some actions
        pid_roll = PID(0, kp, ki, kd, ilimit, outlimit, dt, samplingRate=-1, cutoffFreq = -1, enableDFilter = False)

        df_traj, idx = get_rand_traj(df)
        X, U, dX = df_to_training(df_traj, data_params)

        # plot_traj_model(df_traj, nn_ensemble)
        # plot_traj_model(df_traj, nn)
        plot_battery_thrust(df_traj, nn)

        print("Trajectory idx is: ", idx)
        x0 = X[0,:]
        # action = U[point:point+T+1,:]
        # action = U[point,:]
        action = np.tile([45000,45000,30000,30000],4)
        action = np.concatenate((action,[3900]))
        # print(x0)


        x_stored_PID = pred_traj(x0, pid_roll, nn,T) # len(df_traj))
        # x_stored_past = pred_traj(x0, U, nn_ensemble, T)

        font = {'size'   : 18}

        matplotlib.rc('font', **font)
        matplotlib.rc('lines', linewidth=2.5)

        # plt.tight_layout()

        with sns.axes_style("darkgrid"):
            ax1 = plt.subplot(111)
            # ax2 = plt.subplot(212)

        dim = 4
        plt.title("Pid Response")
        ax1.set_ylim([-35,35])
        # ax2.set_ylim([-35,35])
        ax1.plot(x_stored_PID[:,dim], linestyle = '--', color='b', label ='Predicted PID')
        # ax1.plot(x_stored_past[:,dim], linestyle = '--', color='g', label ='Predicted Past')
        # ax1.plot(X[:,dim], color = 'k', label = 'Ground Truth')
        ax1.legend()
        # ax2.plot(X[point:point+T+1,3:5])
        plt.show()



# TODO: implement a structured PID similar to the crazyflie to see if it helps performance



if __name__ == "__main__":
    print('\n---------------------------------------------------')
    print('Running file generatePID')
    print('\n')
    main()
