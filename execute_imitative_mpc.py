from utils.data import *
from utils.sim import *
from utils.nn import *
from utils.rl import *

# data packages
import pickle

# neural nets
from model_general_nn import GeneralNN
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

# for importing the rlkit model
from rlkit.rlkit.torch.networks import Mlp
from rlkit.rlkit.torch.sac.policies import TanhGaussianPolicy
save_rlkit_policy(
    'data/tsac-cf/tsac-cf_2019_03_24_15_12_56_0000--s-0/params.pkl')

# model = nn.Module()
# model.load_state_dict(torch.load('_policies/test.pth'))
# model.eval()

model = TanhGaussianPolicy(
    hidden_sizes=[300, 300],
    obs_dim=27,
    action_dim=12,
)
model.load_state_dict(torch.load('_policies/test.pth'))
model.eval()
print(model.forward(torch.zeros([1,27])))
quit()

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


# dir_list = ["_newquad1/publ2/c50_rand/",
#             "_newquad1/publ2/c50_roll01/",
#             "_newquad1/publ2/c50_roll02/",
#             "_newquad1/publ2/c50_roll03/",
#             "_newquad1/publ2/c50_roll04/",
#             "_newquad1/publ2/c50_roll05/",
#             "_newquad1/publ2/c50_roll06/",
#             "_newquad1/publ2/c50_roll07/",
#             "_newquad1/publ2/c50_roll08/",
#             "_newquad1/publ2/c50_roll09/",
#             "_newquad1/publ2/c50_roll10/",
#             "_newquad1/publ2/c50_roll11/",
#             "_newquad1/publ2/c50_roll12/"]

dir_list = ["_newquad1/publ2/c25_roll08/",
            "_newquad1/publ2/c25_roll09/",
            "_newquad1/publ2/c25_roll10/",
            "_newquad1/publ2/c25_roll11/",
            "_newquad1/publ2/c25_roll12/"]


# quit()
df = load_dirs(dir_list, load_params)


data_params = {
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

# the true state target values
# 't1_omega_x', 't1_omega_y', 't1_omega_z', 't1_pitch', 't1_roll', 't1_yaw', 't1_lina_x', 't1_lina_y' 't1_lina_z'

st = ['d_omega_x', 'd_omega_y', 'd_omega_z',
      'd_pitch', 'd_omega_z', 'd_pitch',
      'd_lina_x', 'd_lina_y', 'd_liny_z']

X, U, dX = df_to_training(df, data_params)


nn_params = {                           # all should be pretty self-explanatory
    'dx': np.shape(X)[1],
    'du': np.shape(U)[1],
    'dt': np.shape(dX)[1],
    'hid_width': 50,
    'hid_depth': 2,
    'bayesian_flag': True,
    'activation': Swish(),
    'dropout': 0.1,
    'split_flag': False,
    'pred_mode': 'Delta State',
    'ensemble': False
}

train_params = {
    'epochs': 45,
    'batch_size': 18,
    'optim': 'Adam',
    'split': 0.8,
    'lr': .0003,  # bayesian .00175, mse:  .0001
    'lr_schedule': [30, .6],
    'test_loss_fnc': [],
    'preprocess': True,
    'noprint': True
}

policy = generate_mpc_imitate((X,U,dX), data_params, nn_params, train_params)

