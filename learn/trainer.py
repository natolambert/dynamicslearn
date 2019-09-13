import os
import sys
sys.path.append(os.getcwd())

# Our infrastucture files
# from utils_data import * 
# from utils_nn import *
from learn.utils.data import *
from learn.utils.sim import *
from learn.utils.nn import *


# data packages
import pickle

# neural nets
from learn.model_general_nn import GeneralNN
# from model_split_nn import SplitModel
# from model_ensemble_nn import EnsembleNN

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

import argparse
from omegaconf import OmegaConf

######################################################################

if __name__ == '__main__':
    # adding arguments to make code easier to work with
    parser = argparse.ArgumentParser(description='Train a Neural Netowrk off Autonomous Data')
    parser.add_argument('model_name', type=str, help='Give this string to give your model a memorable name')
    parser.add_argument('--log', action='store_true',
                        help='a flag for storing a training log in a txt file')
    parser.add_argument('--noprint', action='store_false',
                        help='turn off printing in the terminal window for epochs')
    parser.add_argument('--ensemble', action='store_true',
                        help='trains an ensemble of models instead of one network')
    parser.add_argument('--nosave', action='store_false',
                        help='if you want to test code and not save the model')

    args = parser.parse_args()

    log = args.log
    noprint = args.noprint
    ensemble = args.ensemble
    model_name = args.model_name

    ######################################################################

    print('\n')
    date_str = str(datetime.datetime.now())[:-5]
    date_str = date_str.replace(' ','--').replace(':', '-')
    print('Running... trainNN_RL.py' + date_str +'\n')

    c = OmegaConf.load('conf/trainer.yaml')
    
    data_dir = c.load.base_dir
    df = stack_dir_pd_iono(data_dir, c.load)
    '''
    ['d_omega_x' 'd_omega_y' 'd_omega_z' 'd_pitch' 'd_roll' 'd_yaw' 'd_lina_x'
    'd_lina_y' 'd_liny_z' 'timesteps' 'objective vals' 'flight times'
    'omega_x0' 'omega_y0' 'omega_z0' 'pitch0' 'roll0' 'yaw0' 'lina_x0'
    'lina_y0' 'lina_z0' 'omega_x1' 'omega_y1' 'omega_z1' 'pitch1' 'roll1'
    'yaw1' 'lina_x1' 'lina_y1' 'lina_z1' 'omega_x2' 'omega_y2' 'omega_z2'
    'pitch2' 'roll2' 'yaw2' 'lina_x2' 'lina_y2' 'liny_z2' 'm1_pwm_0'
    'm2_pwm_0' 'm3_pwm_0' 'm4_pwm_0' 'm1_pwm_1' 'm2_pwm_1' 'm3_pwm_1'
    'm4_pwm_1' 'm1_pwm_2' 'm2_pwm_2' 'm3_pwm_2' 'm4_pwm_2' 'vbat']
    '''
    # explore_pwm_equil(df)
    # quit()

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
                    # 'omega_x3', 'omega_y3', 'omega_z3',
                    # 'pitch3',   'roll3',    'yaw3',
                    # 'lina_x3',  'lina_y3',  'lina_z3'],

        'inputs' : ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                    'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                    'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],# 'vbat'],
                    # 'm1_pwm_3', 'm2_pwm_3', 'm3_pwm_3', 'm4_pwm_3', 'vbat'],

        'targets' : ['t1_omega_x', 't1_omega_y', 't1_omega_z',
                            'd_pitch', 'd_roll', 'd_yaw',
                            't1_lina_x', 't1_lina_y', 't1_lina_z'],

        'battery' : False                    # Need to include battery here too
    }

    def create_model_params(model_cfg):

        return 
    data_params_iono = {
        # Note the order of these matters. that is the order your array will be in
        'states': ['omega_x0', 'omega_y0', 'omega_z0',
                'pitch0',   'roll0',    'yaw0',
                'lina_x0',  'lina_y0',  'lina_z0',
                'omega_x1', 'omega_y1', 'omega_z1',
                'pitch1',   'roll1',    'yaw1',
                'lina_x1',  'lina_y1',  'lina_z1',
                'omega_x2', 'omega_y2', 'omega_z2',
                'pitch2',   'roll2',    'yaw2',
                'lina_x2',  'lina_y2',  'lina_z2'],
        # 'omega_x3', 'omega_y3', 'omega_z3',
        # 'pitch3',   'roll3',    'yaw3',
        # 'lina_x3',  'lina_y3',  'lina_z3'],

        'inputs': ['m1_pwm_0', 'm2_pwm_0', 'm3_pwm_0', 'm4_pwm_0',
                'm1_pwm_1', 'm2_pwm_1', 'm3_pwm_1', 'm4_pwm_1',
                'm1_pwm_2', 'm2_pwm_2', 'm3_pwm_2', 'm4_pwm_2'],  # 'vbat'],
        # 'm1_pwm_3', 'm2_pwm_3', 'm3_pwm_3', 'm4_pwm_3', 'vbat'],

        'targets': ['t1_omega_x', 't1_omega_y', 't1_omega_z',
                    'd_pitch', 'd_roll', 'd_yaw',
                    't1_lina_x', 't1_lina_y', 't1_lina_z'],

        'battery': False                    # Need to include battery here too
    }

    st = ['d_omega_x', 'd_omega_y', 'd_omega_z',
                        'd_pitch', 'd_omega_z', 'd_pitch',
                        'd_lina_x', 'd_lina_y', 'd_liny_z']

    X, U, dX = df_to_training(df, data_params)

    print('---')
    print("X has shape: ", np.shape(X))
    print("U has shape: ", np.shape(U))
    print("dX has shape: ", np.shape(dX))
    print('---')

    # nn_params = {                           # all should be pretty self-explanatory
    #     'dx' : np.shape(X)[1],
    #     'du' : np.shape(U)[1],
    #     'dt' : np.shape(dX)[1],
    #     'hid_width' : 250,
    #     'hid_depth' : 2,
    #     'bayesian_flag' : True,
    #     'activation': Swish(),
    #     'dropout' : 0.0,
    #     'split_flag' : False,
    #     'pred_mode' : 'Delta State',
    #     'ensemble' : ensemble
    # }

    # train_params = {
    #     'epochs' : 20,
    #     'batch_size' : 18,
    #     'optim' : 'Adam',
    #     'split' : 0.8,
    #     'lr': .00155, # bayesian .00175, mse:  .0001
    #     'lr_schedule' : [30,.6],
    #     'test_loss_fnc' : [],
    #     'preprocess' : True,
    #     'noprint' : noprint
    # }


    nn_params = {                           # all should be pretty self-explanatory
        'dx': np.shape(X)[1],
        'du': np.shape(U)[1],
        'dt': np.shape(dX)[1],
        'hid_width': 250,
        'hid_depth': 2,
        'bayesian_flag': True,
        'activation': Swish(),
        'dropout': 0.0,
        'split_flag': False,
        'pred_mode': 'Delta State',
        'ensemble': ensemble
    }

    train_params = {
        'epochs': 33,
        'batch_size': 18,
        'optim': 'Adam',
        'split': 0.8,
        'lr': .00255,  # bayesian .00175, mse:  .0001
        'lr_schedule': [30, .6],
        'test_loss_fnc': [],
        'preprocess': True,
        'noprint': noprint
    }


    # log file
    if log:
        with open('_training_logs/'+'logfile' + date_str + '.txt', 'w') as my_file:
            my_file.write("Logfile for training run: " + date_str +"\n")
            my_file.write("Net Name: " + str(model_name) + "\n")
            my_file.write("============================================="+"\n")
            my_file.write("Data Load Params:"+"\n")
            for k, v in load_params.items():
                my_file.write(str(k) + ' >>> '+ str(v) + '\n')
            my_file.write("\n")

            my_file.write("NN Structure Params:"+"\n")
            for k, v in nn_params.items():
                my_file.write(str(k) + ' >>> '+ str(v) + '\n')
            my_file.write("\n")

            my_file.write("NN Train Params:"+"\n")
            for k, v in train_params.items():
                my_file.write(str(k) + ' >>> '+ str(v) + '\n')
            my_file.write("\n")


    if ensemble:
        newNN = EnsembleNN(nn_params,7)
        acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

        print(acctest)

    else:
        newNN = GeneralNN(nn_params)
        newNN.init_weights_orth()
        if nn_params['bayesian_flag']: newNN.init_loss_fnc(dX,l_mean = 1,l_cov = 1) # data for std,
        acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

    newNN.store_training_lists(data_params['states'],data_params['inputs'],data_params['targets'])

    # plot
    if ensemble:
        min_err = np.min(acctrain,0)
        min_err_test = np.min(acctest,0)
    else:
        min_err = np.min(acctrain)
        min_err_test = np.min(acctest)

    if log:
        with open('_training_logs/'+'logfile' + date_str + '.txt', 'a') as my_file:
            my_file.write("Prediction List" + str(data_params['targets'])+"\n")
            my_file.write("Min test error: " +str(min_err_test)+ "\n")
            my_file.write("Mean Min test error: " + str(np.mean(min_err_test)) + "\n")
            my_file.write("Min train error: " +str(min_err)+ "\n")

    ax1 = plt.subplot(211)
    # ax1.set_yscale('log')
    ax1.plot(acctest, label = 'Test Loss')
    plt.title('Test Loss')
    ax2 = plt.subplot(212)
    # ax2.set_yscale('log')
    ax2.plot(acctrain, label = 'Train Loss')
    plt.title('Training Loss')
    ax1.legend()
    plt.show()

    # Saves NN params
    if args.nosave:
        dir_str = str('_models/temp/')
        data_name = '_100Hz_'
        # info_str = "_" + model_name + "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
        info_str = "_" + model_name +"_" + "stack" + str(load_params['stack_states']) + "_" #+ "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
        model_name = dir_str + date_str + info_str
        newNN.save_model(model_name + '.pth')
        print('Saving model to', model_name)

        normX, normU, normdX = newNN.getNormScalers()
        with open(model_name+"--normparams.pkl", 'wb') as pickle_file:
            pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
            time.sleep(2)

        # Saves data file
        with open(model_name+"--data.pkl", 'wb') as pickle_file:
            pickle.dump(df, pickle_file, protocol=2)
            time.sleep(2)
