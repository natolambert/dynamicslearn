# Our infrastucture files
from utils_data import *
from utils_sim import *

# data packages
import pickle
import random

# neural nets
from model_general_nn import *
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

# basic PID class to perform arithmetic around the setpoint
class PID():
    def __init__(self, desired,
                    kp, ki, kd,
                    ilimit, outlimit,
                    dt, samplingRate = 0, cutoffFreq = -1,
                    enableDFilter = False):

        # internal variables
        self.error = 0
        self.error_prev = 0
        self.integral = 0
        self.deriv = 0

        # constants
        self.desired = desired
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # limits integral growth
        self.ilimit = ilimit

        # limits ridiculous actions. Should set to variance
        self.outlimit = outlimit

        # timee steps for changing step size of PID response
        self.dt = dt
        self.samplingRate = samplingRate    # sample rate is for filtering

        self.cutoffFreq = cutoffFreq
        self.enableDFilter = enableDFilter

        if cutoffFreq != -1 or enableDFilter:
            raise NotImplementedError('Have not implemnted filtering yet')

    def update(self, measured):

        # init
        out = 0.

        # update error
        self.error_prev = self.error

        # calc new error
        self.error = self.desired - measured

        # proportional gain is easy
        out += self.kp*self.error

        # calculate deriv term
        self.deriv = (self.error-self.error_prev) / self.dt

        # filtter if needed (DT function_)
        if self.enableDFilter:
            print('Do Filter')
            self.deriv = self.deriv

        # calcualte error value added
        out += self.deriv*self.kd

        # accumualte normalized eerror
        self.integral = self.error*self.dt

        # limitt the integral term
        if self.ilimit !=0:
            self.integral = np.clip(self.integral,-self.ilimit, self.ilimit)

        out += self.ki*self.integral

        # limitt the total output
        if self.outlimit !=0:
            out = np.clip(out,-self.outlimit, self.outlimit)

        return out

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
# class to mimic the PID structure onboard the crazyflie
class crazyPID(PID):
    def __init__(self, equil, dt, out_lim = 5000,
                att_pitch = [], att_roll = [], att_yaw = [],
                rate_pitch = [], rate_roll = [], rate_yaw = []):

        self.dt = dt

        # PIDs
        self.PID_att_pitch = []
        self.PID_att_roll = []
        self.PID_att_yaw = []
        self.PID_rate_pitch = []
        self.PID_rate_roll = []
        self.PID_rate_yaw = []

        # Above, all of the last six inputs being att_pitch etc are lists of length 5
        # Axis Mode: [KP, KI, KD, iLimit]
        if att_pitch != []:
            self.PID_att_pitch = PID(0, att_pitch[0],
                                        att_pitch[1],
                                        att_pitch[2],
                                        att_pitch[3], dt)

        if PID_att_roll != []:
            self.PID_att_roll = PID(0, PID_att_roll[0],
                                        PID_att_roll[1],
                                        PID_att_roll[2],
                                        PID_att_roll[3], dt)

        if att_pitch != []:
            self.PID_att_yaw = PID(0, att_yaw[0],
                                        att_yaw[1],
                                        att_yaw[2],
                                        att_yaw[3], dt)

        if PID_att_roll != []:
            self.PID_rate_pitch = PID(0, rate_pitch[0],
                                        rate_pitch[1],
                                        rate_pitch[2],
                                        rate_pitch[3], dt)

        if att_pitch != []:
            self.PID_rate_roll = PID(0, rate_roll[0],
                                        rate_roll[1],
                                        rate_roll[2],
                                        rate_roll[3], dt)

        if PID_att_roll != []:
            self.PID_rate_yaw = PID(0, rate_yaw[0],
                                        rate_yaw[1],
                                        rate_yaw[2],
                                        rate_yaw[3], dt)

        # create list of 'active' PIDs
        self.PIDs = []
        if self.PID_att_pitch != []: self.PIDs.append(self.PID_att_pitch)
        if self.PID_att_roll != []: self.PIDs.append(self.PID_att_roll)
        if self.PID_att_yaw != []: self.PIDs.append(self.PID_att_yaw)
        if self.PID_rate_pitch != []: self.PIDs.append(self.PID_rate_pitch)
        if self.PID_rate_roll != []: self.PIDs.append(self.PID_rate_roll)
        if self.PID_rate_yaw != []: self.PIDs.append(self.PID_rate_yaw)

        # def update(self, x):
        #     # this function will take in the current state, and based on
