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

def get_action(cur_state, model, method = 'Random'):
    # Returns an action for the robot given the current state and the model

class PID():
    def __init__(self, desired,
                    kp, ki, kd,
                    ilimit, outlimit,
                    dt, samplingRate, cutoffFreq = -1,
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

def pred_traj(x0, action, model, T):
    # get dims
    stack = int((len(x0))/9)
    xdim = 9
    udim = 4

    # figure out if given an action or a controller
    if not isinstance(action, np.ndarray):
        # given PID controller. Generate actions as it goes
        mode = 1

        PID = copy.deepcopy(action) # for easier naming and resuing code

        # create initial action
        action_eq = np.array([31687.1, 37954.7, 33384.8, 36220.11])
        action = np.array([31687.1, 37954.7, 33384.8, 36220.11])
        if stack > 1:
            action = np.tile(action, stack)
        action = np.concatenate((action,[3900]))

        # step 0 PID response
        action[:udim] += PID.update(x0[4])
    else:
        mode = 0

    # function to generate trajectories
    x_stored = np.zeros((T+1,len(x0)))
    x_stored[0,:] = x0
    x_shift = np.zeros(len(x0))

    for t in range(T):
        if mode == 1:
            # predict with actions coming from controller
            if stack > 1:       # if passed array of actions, iterate
                x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action)

                # slide action here
                action[udim:-1] = action[:-udim-1]
            else:
                x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action)

            # update action
            PIDout = PID.update(x_pred[4])
            action[:udim] = action_eq+np.array([1,1,-1,-1])*PIDout
            print(x_pred[4])
            print(PIDout)
            print(action[:udim])

        # else give action array
        elif mode == 0:
            # predict
            if stack > 1:       # if passed array of actions, iterate
                x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action[t,:])
            else:
                x_pred = x_stored[t,:9]+ model.predict(x_stored[t,:], action)

        # shift values
        x_shift[:9] = x_pred
        x_shift[9:-1] = x_stored[t,:-10]

        # store values
        x_stored[t+1,:] = x_shift

    x_stored[:,-1] = x0[-1]     # store battery for all (assume doesnt change on this time horizon)

    return x_stored
