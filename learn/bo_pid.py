# timing etc
import time
import datetime
import os
import sys

sys.path.append(os.getcwd())

# Our infrastucture files
from learn.utils.data import *
from learn.utils.sim import *
from learn.utils.nn import *
from learn.utils.plot import *
from learn.control.pid import *

# data packages
import pickle
import random

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss

# Plotting
import matplotlib.pyplot as plt
import matplotlib

import hydra
import logging

log = logging.getLogger(__name__)
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

"the angle PID runs on the fused IMU data to generate a desired rate of rotation. This rate of rotation feeds in to the rate PID which produces motor setpoints"
'''


def save_file(object, filename):
    path = os.path.join(os.getcwd(), filename)
    log.info(f"Saving File: {filename}")
    torch.save(object, path)


######################################################################
@hydra.main(config_path='conf/bo_pid.yaml')
def trainer(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    ######################################################################
    log.info(f"Loading model from {cfg.model_path}")
    log.info
    {f"control dimensions spec {cfg.dimension}"}
    dim_s = cfg.dimension
    if dim_s == 'all':
        dim = 0
    elif dim_s == 'pitch':
        dim = 1
    elif dim_s == 'roll':
        dim = 2



    ######################################################################

    # Code outline to predict states given an action or control scheme

    # load model
    # some reasonable seeds:
    # 150hz: 3400, 485, 3850
    # 50Hz  models`
    # model_single_50_nobat = '_models/temp/2018-11-09--10-52-58.0_50hz_nobat_.pth'
    # model_single_50 = '_models/temp/2018-11-09--10-55-12.9_50hz_withbat_stack3_.pth'

    # # 100Hz models`
    # model_single = '_models/temp/2018-11-09--10-48-05.2_100hz_bat_trimmed_.pth'
    # model_single_nobat = '_models/temp/2018-11-09--10-46-56.7_100hz_bat_trimmed_.pth'

    # intitial trained
    model_50_truestate = '_models/temp/2018-11-20--12-32-18.5_c50_true_stack3_.pth'
    # trained on more takeoff points
    model_50_truestate2 = '_models/temp/2018-11-20--12-41-59.8_c50_true2_stack3_.pth'
    # ensemble
    model_50_true_ensemble = '_models/temp/2018-11-20--12-40-00.4_c50_trueaccel_ensem_stack3_.pth'

    # 25hz true state
    model_25 = '_models/temp/2018-11-20--12-55-45.6_c25_true_stack3_.pth'

    nn = torch.load('_models/temp/2018-12-30--10-02-51.1_true_plot_50_stack3_.pth')
    nn.eval()

    '''
    RMSES
    100Hz no bat: [6.3130932  5.8711117  2.20496666 0.34913389 0.40612083 0.2069013 0.59639823 0.82310533 0.52682769]
    100Hz with bat: [6.31994656 5.92266344 2.09359516 0.35665752 0.41530902 0.2053187 0.59421848 0.80986195 0.52297999]

    50Hz no bat: [10.21272868 11.03302467  2.83940086  0.51089872  0.44949222  0.24415286 0.64122414  1.05109164  0.53779281]
    50Hz with bat: [10.2567613  11.142001    2.86625326  0.52282885  0.44307055  0.24636216 0.63476385  1.06089197  0.53843009]

    '''

    state_list, input_list, change_list = nn.get_training_lists()

    ###########################################################################

    # PID Params 250.0, 500.0, 2.5, 33.3
    kp = 250
    ki = 100
    kd = 2.5
    ilimit = 33.3
    outlimit = 5000
    dt = 1 / 50
    PWMequil = np.array([34687.1, 37954.7, 38384.8, 36220.11])  # new quad

    for i in range(15):
        # Lets simulate some actions
        pid_roll = PID(0, kp, ki, kd, ilimit, outlimit, dt, samplingRate=-1, cutoffFreq=-1, enableDFilter=False)


if __name__ == '__main__':
    sys.exit(trainer())
