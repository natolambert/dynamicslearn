import torch
from torch.autograd import Variable
from torch.nn import MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

from GenNN import GeneralNN
from Parse import load_dirs, df_to_training
from utils.nn import Swish
from EnsembleNN import EnsembleNN
from kMeansData import kClusters
from collections import OrderedDict
import xlrd
import ctypes

###############################################################################
'''Loading and parsing data for model training. Setting training/nn parameters'''


dir_list = ["data/c25_rand/",
            "data/c25_roll1/",
            "data/c25_roll2/",
            "data/c25_roll3/",
            "data/c25_roll4/",
            "data/c25_roll5/",
            "data/c25_roll6/",
            "data/c25_roll7/",
            "data/c25_roll8/",
            "data/c25_roll9/",
            "data/c25_roll10/",
            "data/c25_roll11/",
            "data/c25_roll12/",
            "data/c25_roll13/",
            "data/c25_roll14/",
            "data/c25_roll15/",]

load_params ={
    'delta_state': True,                # normally leave as True, prediction mode
    'include_tplus1': True,             # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
    'trim_high_vbat': 4050,             # trims high vbat because these points the quad is not moving
    'takeoff_points': 180,              # If not trimming data with fast log, need another way to get rid of repeated 0s
    'trim_0_dX': True,                  # if all the euler angles (floats) don't change, it is not realistic data
    'find_move': True,
    'trime_large_dX': True,             # if the states change by a large amount, not realistic
    'bound_inputs': [20000,65500],      # Anything out of here is erroneous anyways. Can be used to focus training
    'stack_states': 3,                  # IMPORTANT ONE: stacks the past states and inputs to pass into network
    'collision_flag': False,            # looks for sharp changes to tthrow out items post collision
    'shuffle_here': False,              # shuffle pre training, makes it hard to plot trajectories
    'timestep_flags': [],               # if you want to filter rostime stamps, do it here
    'battery' : True,                   # if battery voltage is in the state data
    'terminals': True,                 # adds a column to the dataframe tracking end of trajectories
    'fastLog' : True,                   # if using the software with the new fast log
    'contFreq' : 1,                      # Number of times the control freq you will be using is faster than that at data logging
    'iono_data': True,
    'zero_yaw': True,
    'moving_avg': 7
}

df = load_dirs(dir_list, load_params)

data_params = {
    # Note the order of these matters. that is the order your array will be in
    'states' : ['omega_x0', 'omega_y0', 'omega_z0',
                'pitch0',   'roll0',    'yaw0',
                'lina_x0',  'lina_y0',  'lina_z0',
                'omega_x1', 'omega_y1', 'omega_z1',
                'pitch1',   'roll1',    'yaw1',
                'lina_x1',  'lina_y1',  'lina_z1',
                'omega_x2', 'omega_y2', 'omega_z2',
                'pitch2',   'roll2',    'yaw2',
                'lina_x2',  'lina_y2',  'lina_z2'],

    'inputs' : ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],

    'targets' : ['t1_omega_x', 't1_omega_y', 't1_omega_z',
                        'd_pitch', 'd_roll', 'd_yaw',
                        't1_lina_x', 't1_lina_y', 't1_lina_z'],

    'battery' : True                    # Need to include battery here too
}

X, U, dX = df_to_training(df, data_params)

def getNewNNParams(nState, nInput, stack):
    return {                           # all should be pretty self-explanatory
        'dx' : nState * stack,
        'du' : nInput * stack,
        'dt' : nState,
        'hid_width' : 250,
        'hid_depth' : 3,
        'bayesian_flag' : True,
        'activation': Swish(),
        'dropout' : 0.2,
        'split_flag' : False,
        'stack': stack,
        'ensemble' : 5,
        'epsilon': 2.2e-100
    }

def getNewTrainParams():
    return {
        'epochs' : 300,
        'batch_size' : 18,
        'optim' : 'Adam',
        'split' : 0.8,
        'lr': .002, # bayesian .00175, mse:  .0001
        'lr_schedule' : [50,.85],
        'test_loss_fnc' : [],
        'preprocess' : True,
        'noprint' : False,
        'momentum': .9
        }
###############################################################################
'''TRAINING A MODEL AND SAVING IT TO A TXT FILE'''
'''ALSO CONTAINS LINE TO TEST INERTIA WITH RESPECT TO # CLUSTERS'''
'''
ENSEMBLE = True
data = (X,U,dX)

if ENSEMBLE:
    nn_params, train_params = getNNTrainParams(False)

    ensembleNN = EnsembleNN(nn_params)
    ensembleNN.init_weights_orth()
    print("Training networks in ensemble...")
    ensembleNN.train_cust(data, train_params)
    path = "EnsembleModelOptimized FullSplitComplexFavoritism2.txt"
    ensembleNN.save_model(path)
else:
    nn_params = [getNewNNParams()]
    train_params = [getNewTrainParams()]
    newNN = GeneralNN(nn_params[0])
    newNN.init_weights_orth()
    testLoss, trainLoss = newNN.train_cust(data, train_params[0])
    path = "TrainedDModel.txt"
    newNN.save_model(path)
'''


##############################################################################
'''GRID SEARCH FOR FINDING OPTIMAL MODEL PARAMETERS'''
'''Parameters tested: dropout, ensembles, step size, decay, batch size'''
'''km = kClusters(9) #dummy number of clusters
km.cluster((X,U,dX))
training, testing = km.sample()
data = np.vstack((training, testing))
data = (data[:, :27], data[:, 27: 39], data[:, 39:])

learnrate = [.002, .0021, .0022]
dropout = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
step = [50, 60, 75]
decay =[.5, .8, .85]
batch = [8, 16, 32]

dict = {"Dropout": [0], "LR": [0], "Step": [0], "Decay": [0], "Batch": [0], "Epoch Mini": [0], "Minimum Loss": [0]} #dummies
train_params = getNewTrainParams()
nn_params = getNewNNParams()
train_params["split"] = len(training) / (len(training) + len(testing))
print("")
print("Running Grid Search...")
print("")
count = 1
for lr in learnrate:
    train_params["lr"] = lr
    for drop in dropout:
        nn_params["dropout"] = drop
        for s in step:
            for d in decay:
                train_params["lr_schedule"] = [s, d]
                for b in batch:
                    train_params["batch_size"] = b
                    print("")
                    print("TRAINING ENSEMBLE NUMBER: ", count)
                    print("")
                    gen = GeneralNN(nn_params)
                    gen.init_weights_orth()
                    minitest, minitrain = gen.train_cust(data, train_params)
                    epochs = np.argmin(np.array(minitest))
                    minitest = min(minitest)
                    minitrain = min(minitrain)
                    dict["LR"] += [lr]
                    dict["Dropout"] += [drop]
                    dict["Step"] += [s]
                    dict["Decay"] += [d]
                    dict["Batch"] += [b]
                    dict["Epoch Mini"] += [epochs]
                    dict["Minimum Loss"] += [minitest]
                    count += 1

                    print("Minimum Loss: ", minitest, " Epoch found: ", epochs)

df = pd.DataFrame(OrderedDict(dict.items(), key = lambda t: len(t[0])))
print("Saving data into excel...")
writer = ExcelWriter('ParameterSweep4_Favoritism.xlsx')
df.to_excel(writer, 'Sheet1', index = False)
writer.save()
print("Excel saved")
'''
###############################################################################
##############################################################################
'''PROVIDE INITIAL CONDITIONS/INPUT FOR BAYESIAN OPTIMIZATION'''

def getInput(model):

    df = load_dirs(dir_list, load_params)
    X, U, dX = df_to_training(df, data_params)
    return X, U
