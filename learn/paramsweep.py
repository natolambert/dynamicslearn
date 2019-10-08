# Our infrastucture files
from utils.data import *
from utils.sim import *
from utils.nn import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN
from model_split_nn import SplitModel
from _activation_swish import Swish

# Torch Packages
import torch
import torch.nn as nn
from torch.nn import MSELoss

# timing etc
import time
import datetime
import os

# Plotting
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import KFold

if __name__ == '__main__':
    date_str = str(datetime.datetime.now())[:-5]
    date_str = date_str.replace(' ','--').replace(':', '-')
    print('Running... paramsweep.py' + date_str +'\n')


    data_params_stack1 = {
        'states': ['pitch0', 'roll0', 'yaw0', 'lina_x0', 'lina_y0', 'lina_z0'],                      # most of these are to be implented for easily training specific states etc
        'inputs': ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0'],
        'targets': ['d_pitch', 'd_roll', 'd_yaw',
                    't1_linax', 't1_linay', 't1_linaz'],
        'battery': True                    # Need to include battery here too
    }

    data_params_stack2 = {
        'states': ['pitch0', 'roll0', 'yaw0', 'lina_x0', 'lina_y0', 'lina_z0',
                'pitch1', 'roll1', 'yaw1', 'lina_x1', 'lina_y1', 'lina_z1'],                      # most of these are to be implented for easily training specific states etc
        'inputs': ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1'],
        'targets': ['d_pitch', 'd_roll', 'd_yaw',
                    't1_linax', 't1_linay', 't1_linaz'],
        'battery': True                    # Need to include battery here too
    }

    data_params_stack3 = {
        'states': ['pitch0', 'roll0', 'yaw0', 'lina_x0', 'lina_y0', 'lina_z0',
                'pitch1', 'roll1', 'yaw1', 'lina_x1', 'lina_y1', 'lina_z1',
                'pitch2', 'roll2', 'yaw2', 'lina_x2', 'lina_y2', 'lina_z2'],                      # most of these are to be implented for easily training specific states etc
        'inputs': ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0', 
                'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],
        'targets': ['d_pitch', 'd_roll', 'd_yaw',
                    't1_linax', 't1_linay', 't1_linaz'],
        'battery': True                    # Need to include battery here too
    }

    data_params_stack4 = {
        'states': ['pitch0', 'roll0', 'yaw0', 'lina_x0', 'lina_y0', 'lina_z0',
                'pitch1', 'roll1', 'yaw1', 'lina_x1', 'lina_y1', 'lina_z1',
                'pitch2', 'roll2', 'yaw2', 'lina_x2', 'lina_y2', 'lina_z2',
                'pitch3', 'roll3', 'yaw3', 'lina_x3', 'lina_y3', 'lina_z3'],                      # most of these are to be implented for easily training specific states etc
        'inputs': ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2',
                'm1_pwm_3', 'm2_pwm_3', 'm3_pwm_3', 'm4_pwm_3'],
        'targets': ['d_pitch', 'd_roll', 'd_yaw',
                    't1_linax', 't1_linay', 't1_linaz'],
        'battery': True                    # Need to include battery here too
    }

    load_params = {
        'delta_state': True,                # normally leave as True, prediction mode
        # when true, will include the time plus one in the dataframe (for trying predictions of true state vs delta)
        'include_tplus1': True,
        # trims high vbat because these points the quad is not moving
        'trim_high_vbat': 4050,
        # If not trimming data with fast log, need another way to get rid of repeated 0s
        'takeoff_points': 180,
        # if all the euler angles (floats) don't change, it is not realistic data
        'trim_0_dX': True,
        'find_move': True,
        # if the states change by a large amount, not realistic
        'trime_large_dX': True,
        # Anything out of here is erroneous anyways. Can be used to focus training
        'bound_inputs': [20000, 65500],
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
        'contFreq': 1,
        'iono_data': True,
        'zero_yaw': True,
        'moving_avg': 7
    }


    #########################   ######################################################

    # Setup values
    # learns = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 5e-7, 1e-7]
    # batches = [15, 25, 32, 50, 100, 200]

    # df = load_dirs(dir_list, load_params)

    data_params = {
        'states' : [],                      # most of these are to be implented for easily training specific states etc
        'inputs' : [],
        'change_states' : [],
        'battery' : True                    # Need to include battery here too
    }


    # Setup DF
    d = {"MinTrainLoss" : [],
        "MinTestLoss" : [],
        "LR": [],
        "Depth": [],
        "Batch Size": [],
        "Activation": [],
        "ProbFlag": [],
        "Opt": []
    }

    K_fold = 5
    results = pd.DataFrame(d)

    learns = [.002]
    batches = [18] #32, 45]
    # Iteration Loop
    i = 0
    for l in learns:
        for b in batches:
            for act in [Swish()]:
                for de in [2,3]:
                    for opt in ["Adam"]:
                        for stack in [2,3,4]:
                            for mavg in [1,2,3,5]:
                                load_params['moving_avg'] = mavg
                                load_params['stack_states'] = stack

                                df = stack_dir_pd_iono('broken/', load_params)

                                # choose params
                                if stack == 1:
                                    data_params = data_params_stack1
                                elif stack == 2:
                                    data_params = data_params_stack2
                                elif stack == 3:
                                    data_params = data_params_stack3
                                elif stack == 4:
                                    data_params = data_params_stack4

                                X, U, dX = df_to_training(df, data_params)
        

                                print('-----------------------------------------------------')
                                print('ITERATION: ', i)


                                nn_params = {                           # all should be pretty self-explanatory
                                    'dx' : np.shape(X)[1],
                                    'du' : np.shape(U)[1],
                                    'dt' : np.shape(dX)[1],
                                    'hid_width' : 250,
                                    'hid_depth' : int(de),
                                    'bayesian_flag' : True,
                                    'activation': act,
                                    'dropout' : 0.0,
                                    'split_flag' : False,
                                    'pred_mode' : 'Delta State',
                                    'ensemble' : False
                                }

                                train_params = {
                                    'epochs': 50,
                                    'batch_size': 18,
                                    'optim': 'Adam',
                                    'split': 1/K_fold,
                                    'lr': .002,  # bayesian .00175, mse:  .0001
                                    'lr_schedule': [30, .6],
                                    'test_loss_fnc': [],
                                    'preprocess': True,
                                    'noprint': True
                                }

                                

                                # Cross validate each network design over the dataset used
                                kf = KFold(n_splits=K_fold)
                                kf.get_n_splits(X)

                                cross_val_err_test = []
                                cross_val_err_train = []
                                # iterate through the validation sets
                                for train_index, test_index in kf.split(X):
                                    # print(train_index)
                                    # print(test_index)   # train = data[0]
                                    X_train, X_test = X[train_index], X[test_index]
                                    U_train, U_test = U[train_index], U[test_index]
                                    dX_train, dX_test = dX[train_index], dX[test_index]
                                    X_k = np.append(X_train, X_test, axis=0)
                                    U_k = np.append(U_train, U_test, axis=0)
                                    dX_k = np.append(dX_train, dX_test, axis=0)

                                    # Initialize
                                    newNN = GeneralNN(nn_params)
                                    newNN.init_weights_orth()
                                    newNN.init_loss_fnc(dX, l_mean=1, l_cov=1)  # data for std,
                                    acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

                                    cross_val_err_test.append(min(acctest))
                                    cross_val_err_train.append(min(acctrain))

                                loss_train = np.mean(cross_val_err_train)
                                loss_test = np.mean(cross_val_err_test)

                                # # Train
                                # try:
                                #     acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

                                # except:
                                #     acctest = [np.inf]
                                #     acctrain = [np.inf]

                                # best_loss_test = min(acctest)
                                # best_loss_train = min(acctrain)

                                d = {"MinTrainLoss" : loss_train,
                                    "MinTestLoss" : loss_test,
                                    "LR" : l,
                                    "Depth" : d,
                                    "Batch Size" : b,
                                    "Activation" : act,
                                    "ProbFlag" : True,
                                    "Opt" : opt,
                                    "Stack" : stack,
                                    "Moving Avg" : mavg
                                }
                                print(d)
                                results = results.append(d, ignore_index=True)
                                i+=1


    print(results)
    results.to_csv('PARAMSWEEP-2.csv')
    print('Saved and Done')
