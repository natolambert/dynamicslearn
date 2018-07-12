from dynamics import *
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
# from dynamics_ionocraft_imu import IonoCraft_IMU
# from dynamics_ionocraft_threeinput import IonoCraft_3u
from dynamics_crazyflie_linearized import CrazyFlie
from utils_plot import *
from utils_data import *
from models import LeastSquares
#from model_pnn import PNeuralNet
from model_general_nn import GeneralNN, predict_nn
import torch
from torch.nn import MSELoss
# import torch.nn as nn
import time
import datetime

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from enum import Enum

class RunType(Enum):
  CF = 1
  IONO =2

runType = RunType.CF



### EXAMPLE EXECUTION

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

# dynamics object
iono1 = IonoCraft(dt_x, threeinput = True, x_noise = 1e-9)
# crazy = CrazyFlie(dt_x, x_noise = .000)
print('...Initializing Dynamics Object')

# initial state is origin
x0 = np.zeros(iono1.x_dim)
u0 = iono1.u_e

# good for unit testin dynamics
x1 = iono1.simulate(x0,u0)
printState(x1)

# quit()

################################ DATA ################################
# Simulate data for training
N = 150     # num sequneces


# generate training data
print('...Generating Training Data')
# Seqs_X, Seqs_U = generate_data(iono1, dt_m, dt_control = dt_u, sequence_len=200, num_iter = N, variance = .007)

if runType is RunType.IONO:
  Seqs_X, Seqs_U = loadcsv('_logged_data/iono/tt_log_4.csv')
elif runType is RunType.CF:
  #Seqs_X, Seqs_U = loadcsv('_logged_data/crazyflie/stateandaction-20180711T22-44-47.csv')
  #Seqs_X, Seqs_U = loadcsv('_logged_data/crazyflie/clean_fly_and_hover_long_data.csv')
  Seqs_X, Seqs_U = loadcsv('_logged_data/crazyflie/hopping.csv')
  data = np.concatenate([Seqs_X,Seqs_U],1)
  data = data[data[:,3]!=0,:]
  #states = np.concatenate([data[:,1:3],data[:,4:6],data[:,7:8]], 1)
  states = np.concatenate([data[:,4:6],data[:,7:8]], 1)
  #imu = unpack_cf_imu(states[:,1], states[:,0])

  #Seqs_X = np.concatenate([imu, states[:,2:]], 1)
  Seqs_X = states
  Seqs_U = unpack_cf_pwm(data[:,3])

if len(Seqs_X.shape) < 3:
  print("added padding dimension to Seqs_X")
  Seqs_X = np.expand_dims(Seqs_X, axis=0)
if len(Seqs_U.shape) < 3:
  print("added padding dimension to Seqs_U")
  Seqs_U = np.expand_dims(Seqs_U, axis=0)

#
#
np.savez('_simmed_data/testingfile_generalnn.npz', Seqs_X, Seqs_U)

print('.... loading training data')
npzfile = np.load('_simmed_data/testingfile_generalnn.npz')
Seqs_X = npzfile['arr_0']
Seqs_U = npzfile['arr_1']


# converts data from list of trajectories of [next_states, states, inputs] to a large array of [next_states, states, inputs]
# downsamples by a factor of samp. This is normalizing the differnece betweem dt_x and dt_measure
data = sequencesXU2array(Seqs_X, Seqs_U)


################################ LEARNING ################################


# #creating neural network with 2 layers of 100 linearly connected ReLU units
print('...Training Model')
layer_sizes = [12, 400, 400, 9]
layer_types = ['nn.Linear()','nn.ReLU()', 'nn.Linear()', 'nn.Linear()']
states_learn = ['yaw', 'pitch', 'roll', 'ax', 'ay', 'az'] #,'ax', 'ay', 'az'] #['yaw', 'pitch', 'roll', 'ax', 'ay', 'az']
# ['X', 'Y', 'Z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll', 'w_z', 'w_x', 'w_y']
forces_learn = ['Thrust', 'taux', 'tauy']

if runType is RunType.CF:
  newNN = GeneralNN(n_in_input = 4, n_in_state = 3, n_out = 3, state_idx_l=[6,7,8], prob=False, pred_mode = 'Next State')#, ang_trans_idx =[0,1,2])
elif runType is RunType.IONO:
  newNN = GeneralNN(n_in_input = 4, n_in_state = 6, n_out = 6, state_idx_l=[0,1,2,3,4,5], prob=False, pred_mode = 'Next State')#, ang_trans_idx =[0,1,2])
else:
  newNN = GeneralNN(n_in_input = 3, n_in_state = 3, n_out = 3, state_idx_l=[6,7,8], prob=True, pred_mode = 'Next State')#, ang_trans_idx =[0,1,2])
ypraccel = [6,7,8,12,13,14]
ypr = [6,7,8]
print(np.shape(Seqs_U))

if runType is RunType.CF:
  acc = newNN.train((Seqs_X, Seqs_U), learning_rate=2.5e-5, epochs=50, batch_size = 124, optim="Adam")
elif runType is RunType.IONO:
  acc = newNN.train((Seqs_X, Seqs_U), learning_rate=2.5e-5, epochs=50, batch_size = 100, optim="Adam")
else:
  acc = newNN.train((Seqs_X[:,::samp,ypr], Seqs_U[:,::samp,:]), learning_rate=2.5e-5, epochs=25, batch_size = 100, optim="Adam")

# Saves model with date string for sorting
dir_str = str('_models/')
date_str = str(datetime.date.today())
model_name = str('_general_temp')
newNN.save_model(dir_str+date_str+model_name+'.pth')
time.sleep(2)

print('Loading as new model')
newNN2 = torch.load(dir_str+date_str+model_name+'.pth')

#if runType is RunType.CF:
  #acc2 = newNN2.train((Seqs_X, Seqs_U), learning_rate=2.5e-5, epochs=25, batch_size = 100, optim="Adam")
#else:
#  acc = newNN.train((Seqs_X[:,::samp,ypr], Seqs_U[:,::samp,:]), learning_rate=2.5e-5, epochs=25, batch_size = 100, optim="Adam")

# Plot accuracy #
#plt.plot(np.transpose(acc2))
plt.plot(np.transpose(acc))
plt.show()
#quit()

# quit()
#
# pnn = PNeuralNet()
# pnn_ypr = PNeuralNet_ypr()
# dnn_ypr = NeuralNet_ypr()
# # ['F1', 'F2', 'F3', 'F4']
# # nn = NeuralNet(layer_sizes, layer_types, iono1, states_learn, forces_learn)
#
#
# ypraccel = [6,7,8,12,13,14]
# ypr = [6,7,8]
# # Create New model
# # acc = nn_ens.train_ens((Seqs_X[:,::samp,:], Seqs_U[:,::samp,:]), learning_rate=2.5e-5, epochs=15, batch_size = 1500, optim="Adam")
# # acc = pnn.train((Seqs_X[:,::samp,ypraccel], Seqs_U[:,::samp,:]), learning_rate=7.5e-6, epochs=240, batch_size = 100, optim="Adam")
#
# acc = pnn_ypr.train((Seqs_X[:,::samp,ypr], Seqs_U[:,::samp,:]), learning_rate=2.5e-5, epochs=200, batch_size = 100, optim="Adam")
#
# # Plot accuracy #
# plt.plot(np.transpose(acc))
# plt.show()

# Saves model with date string for sorting
# dir_str = str('_models/')
# date_str = str(datetime.date.today())
# model_name = str('_MODEL_absdet')
# pnn.save_model(dir_str+date_str+model_name+'.pth')

# Or load model
# pnn = torch.load('pnn_moredata.pth')
# nn = torch.load('testingnn_new.pth')
print(np.shape(data))

ypr = [0,1,2]
plot_model(data, newNN, 0, model_dims = [0,1,2], delta=False)
plot_model(data, newNN, 1, model_dims = ypr, delta=False)
plot_model(data, newNN, 2, model_dims = ypr, delta=False)

plot_trajectories_state(Seqs_X, 2)

# quit()

################################ Obj Fnc ################################
origin_minimizer = Objective(np.linalg.norm, 'min', 6, dim_to_eval=[6,7,8])
print('...Objective Function Initialized')

################################ MPC ################################

# initialize MPC object with objective function above
mpc1 = MPController(newNN, iono1, dt_x, dt_u, origin_minimizer, N=50, T=5, variance = .00003)
print('...MPC Running')

new_len = 500
x_controlled, u_seq = sim_sequence(iono1, dt_m, dt_u, controller = mpc1, sequence_len = new_len, to_print = False)


# x0 = np.zeros(12)
# new_seq, Us = sim_sequence(iono1, dt_u, sequence_len = 150, x0=x0, controller = mpc1)
#
# compareTraj(Us, x0, iono1, nn, show=True)
################################ Sim Controlled ################################

# Sim sequence off the trained controller
new_len = 5000
x_controlled, u_seq = sim_sequence(iono1, dt_m, dt_u, controller = mpc1, sequence_len = new_len, to_print = False)
print(np.mean(x_controlled[:,[12,13,14]]))
print(x_controlled[:,[6,7,8]])
print('Simulated Learned.')
################################ PLot ################################
print('...Plotting')
# plot states and inputs of the trajectory if wanted
T = np.linspace(0,new_len*dt_x,new_len)
plot12(x_controlled, T)
# plotInputs(u_seq, T)
fig_inputs = plt.figure()
plt.title('Three Inputs')
plt.plot(T, u_seq[:,0],label='Thrust')
plt.plot(T, u_seq[:,1],label='taux')
plt.plot(T, u_seq[:,2],label='tauy')
plt.legend()
# plt.show()
# # Plots animation, change save to false to not save .gif
plotter1 = PlotFlight(x_controlled[::15,:],.5)
plotter1.show(save=False)
# print('Saved Gif')


print('---------------------------------------------------------end run-----')
