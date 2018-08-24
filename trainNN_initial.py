### REMAKING THE TRAINING CODE ###

__author__ = 'Nathan Lambert'
__version__ = '1.0'

from dynamics import *
import pickle
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
from dynamics_crazyflie_linearized import CrazyFlie
from utils_plot import *
from utils_data import *
from model_general_nn import GeneralNN, predict_nn
import torch
from torch.nn import MSELoss
import time
import datetime
from model_split_nn import SplitModel

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from enum import Enum



################################ INITIALIZATION ################################
print('\n')
print('---begin--------------------------------------------------------------')
start_time = time.time()
# initialize some variables
dt_x = .0002        # 5khz dynam update
dt_m = .002         # 500 hz measurement update
dt_u = .004         # ~200 hz control update
samp = int(dt_m/dt_x)     # effective sample rate of simulated data

print('Simulation update step is: ', dt_x, ' and control update is: ', dt_u, 'the ratio is: ', dt_u/dt_x)
print('...Initializing Dynamics Object')
# dynamics object for passing variables. Will be depreciated in full release.
crazy1 = CrazyFlie(dt_x, x_noise = .000)
# initial state is origin
x0 = np.zeros(crazy1.x_dim)
u0 = crazy1.u_e

# good for unit testin dynamics
x1 = crazy1.simulate(x0,u0)

############################## LOADING DATA ###############################
print('...Loading Data')
data_dir = '_logged_data/pink-cf1/'
data_name = '2018_08_22_cf1_hover_'
Seqs_X = np.loadtxt(open(data_dir + data_name + 'Seqs_X.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)
Seqs_U = np.loadtxt(open(data_dir + data_name + 'Seqs_U.csv', 'r', encoding='utf-8'), delimiter=",", skiprows=1)
Seqs_dX = Seqs_X[1:,:]-Seqs_X[:-1,:]
Seqs_X = Seqs_X[:-1]
Seqs_U = Seqs_U[:-1]

# remove repeated euler angles
Seqs_X = Seqs_X[np.all(Seqs_dX[:,3:] !=0, axis=1)]
Seqs_U = Seqs_U[np.all(Seqs_dX[:,3:] !=0, axis=1)]
Seqs_dX = Seqs_dX[np.all(Seqs_dX[:,3:] !=0, axis=1)]

# plt.plot(Seqs_dX[:,:])
# plt.legend(['omeg_x', 'omeg_y','omeg_z', 'pitch', 'roll', 'yaw'])
# plt.show()
# quit()

print('  State data of shape: ', np.shape(Seqs_X))
print('  Input data of shape: ', np.shape(Seqs_U))

# handles functions for processing raw logged data
raw_data = False
if raw_data:
    # will need to use the following functions:
    # unpack_cf_imu()
    # unpack_cf_pwm()
    # may need to make one for only accelerations
    print('')
    # NOTE this will be one function in future version

plotting = False
if plotting:
    # currently plotting needs a different data structure
    # NOTE This will be depreciated in future versions
    data = sequencesXU2array(Seqs_X, Seqs_U)

############################## Training NN ###############################
print('...Training NN')
# Some nn Parameters
w = 150     # Network width
e = 300      # number of epochs
b  = 32     # batch size
lr = 7e-6   # learning rate
prob_flag = True

# Initialize
newNN = GeneralNN(n_in_input = 4,
                    n_in_state = 6,
                    hidden_w=w,
                    n_out = 6,
                    state_idx_l=[0,1,2,3,4,5],
                    prob=prob_flag,
                    input_mode = 'Stacked Data',
                    pred_mode = 'Delta State',
                    depth=3,
                    activation="Swish",
                    B = 1.0,
                    outIdx = [0,1,2,3,4,5],
                    dropout=0.5,
                    split_flag = True)

# Train
acc = newNN.train((Seqs_X, Seqs_U),
                    learning_rate = lr,
                    epochs=e,
                    batch_size = b,
                    optim="Adam")

plt.plot(acc)
plt.show()

# Saves NN params
dir_str = str('_models/temp/')
date_str = str(datetime.datetime.now())[:-5]
date_str = date_str.replace(' ','--').replace(':', '-')
info_str = "||w=" + str(w) + "e=" + str(e) + "lr=" + str(lr) + "b=" + str(b) + "d=" + str(data_name) + "p=" + str(prob_flag)
model_name = dir_str + date_str + info_str
newNN.save_model(model_name + '.pth')

normX, normU, normdX = newNN.getNormScalers()
with open(model_name+"||normparams.pkl", 'wb') as pickle_file:
  pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
time.sleep(2)

# NOTE: On saving and loading model_split_nn
'''
Save a NN model with:
    dir_str = str('_models/')
    date_str = str(datetime.datetime.now())
    info_str = "w-" + str(width) + "e-" + str(epochs) + "lr-" + str(learning_rate) + "b-" + str(batch) + "d-" + str(data_name) + "p-" + str(prob)
    #model_name = str('_general_temp')
    model_name = dir_str + date_str + info_str
    newNN.save_model(model_name + '.pth')
    normX, normU, normdX = newNN.getNormScalers()

Load a NN model with:
    newNN = torch.load('_models/2018-08-08 13:07:38.973867w-150e-50lr-7e-06b-32d-pink_long_hover_cleanp-True.pth')

Save the scaler/dataset variables from training with:
    with open(model_name+"-normparams.pkl", 'wb') as pickle_file:
      pickle.dump((normX,normU,normdX), pickle_file, protocol=2)
    time.sleep(2)
'''
