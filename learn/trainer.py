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
import hydra

# Plotting
import matplotlib.pyplot as plt
import matplotlib

import argparse
from omegaconf import OmegaConf

import logging

log = logging.getLogger(__name__)


######################################################################
@hydra.main(config_path='conf/trainer.yaml')
def trainer(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    ensemble = cfg.ensemble
    model_name = cfg.model_name

    ######################################################################

    date_str = str(datetime.datetime.now())[:-5]
    date_str = date_str.replace(' ', '--').replace(':', '-')
    log.info('Training a new model')

    data_dir = cfg.load.base_dir
    df, log_load = preprocess_iono(data_dir, cfg.load)
    msg = f"Loading Data"
    if 'dir' in log_load is not None:
        msg += f", dir={log_load['dir']}"
    if 'num_files' in log_load is not None:
        msg += f", num_files={log_load['num_files']}"
    if 'datapoints' in log_load:
        msg += f", datapoints={log_load['datapoints']}"
    log.info(msg)

    '''
    ['d_omegax' 'd_omegay' 'd_omegaz' 'd_pitch' 'd_roll' 'd_yaw' 'd_linax'
    'd_linay' 'd_linyz' 'timesteps' 'objective vals' 'flight times'
    'omega_x0' 'omega_y0' 'omega_z0' 'pitch0' 'roll0' 'yaw0' 'lina_x0'
    'lina_y0' 'lina_z0' 'omega_x1' 'omega_y1' 'omega_z1' 'pitch1' 'roll1'
    'yaw1' 'lina_x1' 'lina_y1' 'lina_z1' 'omega_x2' 'omega_y2' 'omega_z2'
    'pitch2' 'roll2' 'yaw2' 'lina_x2' 'lina_y2' 'liny_z2' 'm1pwm_0'
    'm2pwm_0' 'm3pwm_0' 'm4pwm_0' 'm1pwm_1' 'm2pwm_1' 'm3pwm_1'
    'm4pwm_1' 'm1pwm_2' 'm2pwm_2' 'm3pwm_2' 'm4pwm_2' 'vbat']
    '''
    # explorepwm_equil(df)
    # quit()

    data_params = {
        # Note the order of these matters. that is the order your array will be in
        'states': ['omega_x0', 'omega_y0', 'omega_z0',
                   'pitch0', 'roll0', 'yaw0',
                   'lina_x0', 'lina_y0', 'lina_z0',
                   'omega_x1', 'omega_y1', 'omega_z1',
                   'pitch1', 'roll1', 'yaw1',
                   'lina_x1', 'lina_y1', 'lina_z1',
                   'omega_x2', 'omega_y2', 'omega_z2',
                   'pitch2', 'roll2', 'yaw2',
                   'lina_x2', 'lina_y2', 'lina_z2'],
        # 'omega_x3', 'omega_y3', 'omega_z3',
        # 'pitch3',   'roll3',    'yaw3',
        # 'lina_x3',  'lina_y3',  'lina_z3'],

        'inputs': ['m1pwm_0', 'm2pwm_0', 'm3pwm_0', 'm4pwm_0',
                   'm1pwm_1', 'm2pwm_1', 'm3pwm_1', 'm4pwm_1',
                   'm1pwm_2', 'm2pwm_2', 'm3pwm_2', 'm4pwm_2'],  # 'vbat'],
        # 'm1pwm_3', 'm2pwm_3', 'm3pwm_3', 'm4pwm_3', 'vbat'],

        'targets': ['t1_omegax', 't1_omegay', 't1_omegaz',
                    'd_pitch', 'd_roll', 'd_yaw',
                    't1_linax', 't1_linay', 't1_linaz'],

        'battery': False  # Need to include battery here too
    }

    def create_model_params(model_cfg):
        params = dict()
        # For delta state, prepend with d_omega_y
        # for true state, prepend with t1_lina_x
        # for history of said item, append the number directly  m2pwm2
        base_list = ['omegax', 'omegay', 'omegaz', 'pitch', 'roll', 'yaw', 'linax',
                     'linay', 'linyz', 'timesteps', 'objective vals', 'flight times',
                     'm1pwm', 'm2pwm', 'm3pwm', 'm4pwm', 'vbat']
        always_ignore = ['timesteps', 'objective vals', 'flight times', ]
        for a in always_ignore: base_list.remove(a)
        base_list = np.array(base_list)
        states = base_list[['pwm' not in b for b in base_list]]
        targets = base_list
        inputs = base_list[['pwm' in b for b in base_list]].tolist()
        if model_cfg.history > 0:
            expanded_s = []
            expanded_i = []
            for h in range(model_cfg.history):
                for s in states:
                    if s in model_cfg.ignore_in: continue
                    s += str(h)
                    expanded_s.append(s)
                for i in inputs:
                    i += str(h)
                    expanded_i.append(i)

            print("append historic elements for inputs")
            states = expanded_s
            inputs = expanded_i

        params['targets'] = targets
        params['states'] = expanded_s
        params['inputs'] = expanded_i
        params['battery'] = False

        def check_in(string, set):
            out = False
            for s in set:
                if s in string:
                    out = True
            return out

        print('trim')
        return

    create_model_params(cfg.model)
    data_params_iono = {
        # Note the order of these matters. that is the order your array will be in
        'states': ['omega_x0', 'omega_y0', 'omega_z0',
                   'pitch0', 'roll0', 'yaw0',
                   'lina_x0', 'lina_y0', 'lina_z0',
                   'omega_x1', 'omega_y1', 'omega_z1',
                   'pitch1', 'roll1', 'yaw1',
                   'lina_x1', 'lina_y1', 'lina_z1',
                   'omega_x2', 'omega_y2', 'omega_z2',
                   'pitch2', 'roll2', 'yaw2',
                   'lina_x2', 'lina_y2', 'lina_z2'],
        # 'omega_x3', 'omega_y3', 'omega_z3',
        # 'pitch3',   'roll3',    'yaw3',
        # 'lina_x3',  'lina_y3',  'lina_z3'],

        'inputs': ['m1pwm_0', 'm2pwm_0', 'm3pwm_0', 'm4pwm_0',
                   'm1pwm_1', 'm2pwm_1', 'm3pwm_1', 'm4pwm_1',
                   'm1pwm_2', 'm2pwm_2', 'm3pwm_2', 'm4pwm_2'],  # 'vbat'],
        # 'm1pwm_3', 'm2pwm_3', 'm3pwm_3', 'm4pwm_3', 'vbat'],

        'targets': ['t1_omegax', 't1_omegay', 't1_omegaz',
                    'd_pitch', 'd_roll', 'd_yaw',
                    't1_linax', 't1_linay', 't1_linaz'],

        'battery': False  # Need to include battery here too
    }

    st = ['d_omegax', 'd_omegay', 'd_omegaz',
          'd_pitch', 'd_omegaz', 'd_pitch',
          'd_linax', 'd_linay', 'd_linyz']

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

    nn_params = {  # all should be pretty self-explanatory
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
    }

    # log file
    if log:
        with open('_training_logs/' + 'logfile' + date_str + '.txt', 'w') as my_file:
            my_file.write("Logfile for training run: " + date_str + "\n")
            my_file.write("Net Name: " + str(model_name) + "\n")
            my_file.write("=============================================" + "\n")
            my_file.write("Data Load Params:" + "\n")
            for k, v in load_params.items():
                my_file.write(str(k) + ' >>> ' + str(v) + '\n')
            my_file.write("\n")

            my_file.write("NN Structure Params:" + "\n")
            for k, v in nn_params.items():
                my_file.write(str(k) + ' >>> ' + str(v) + '\n')
            my_file.write("\n")

            my_file.write("NN Train Params:" + "\n")
            for k, v in train_params.items():
                my_file.write(str(k) + ' >>> ' + str(v) + '\n')
            my_file.write("\n")

    if ensemble:
        newNN = EnsembleNN(nn_params, 7)
        acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

        print(acctest)

    else:
        newNN = GeneralNN(nn_params)
        newNN.init_weights_orth()
        if nn_params['bayesian_flag']: newNN.init_loss_fnc(dX, l_mean=1, l_cov=1)  # data for std,
        acctest, acctrain = newNN.train_cust((X, U, dX), train_params)

    newNN.store_training_lists(data_params['states'], data_params['inputs'], data_params['targets'])

    # plot
    if ensemble:
        min_err = np.min(acctrain, 0)
        min_err_test = np.min(acctest, 0)
    else:
        min_err = np.min(acctrain)
        min_err_test = np.min(acctest)

    if log:
        with open('_training_logs/' + 'logfile' + date_str + '.txt', 'a') as my_file:
            my_file.write("Prediction List" + str(data_params['targets']) + "\n")
            my_file.write("Min test error: " + str(min_err_test) + "\n")
            my_file.write("Mean Min test error: " + str(np.mean(min_err_test)) + "\n")
            my_file.write("Min train error: " + str(min_err) + "\n")

    ax1 = plt.subplot(211)
    # ax1.set_yscale('log')
    ax1.plot(acctest, label='Test Loss')
    plt.title('Test Loss')
    ax2 = plt.subplot(212)
    # ax2.set_yscale('log')
    ax2.plot(acctrain, label='Train Loss')
    plt.title('Training Loss')
    ax1.legend()
    plt.show()

    # Saves NN params
    if args.nosave:
        dir_str = str('_models/temp/')
        data_name = '_100Hz_'
        # info_str = "_" + model_name + "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
        info_str = "_" + model_name + "_" + "stack" + str(
            load_params['stack_states']) + "_"  # + "--Min error"+ str(min_err_test)+ "d=" + str(data_name)
        model_name = dir_str + date_str + info_str
        newNN.save_model(model_name + '.pth')
        print('Saving model to', model_name)

        normX, normU, normdX = newNN.getNormScalers()
        with open(model_name + "--normparams.pkl", 'wb') as pickle_file:
            pickle.dump((normX, normU, normdX), pickle_file, protocol=2)
            time.sleep(2)

        # Saves data file
        with open(model_name + "--data.pkl", 'wb') as pickle_file:
            pickle.dump(df, pickle_file, protocol=2)
            time.sleep(2)


if __name__ == '__main__':
    sys.exit(trainer())
