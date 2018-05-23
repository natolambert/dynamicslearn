from dynamics import *
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
# from dynamics_ionocraft_imu import IonoCraft_IMU
# from dynamics_ionocraft_threeinput import IonoCraft_3u
from dynamics_crazyflie_linearized import CrazyFlie
from utils_plot import *
from utils_data import *
from models import LeastSquares

import torch.nn as nn


# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


### EXAMPLE EXECUTION

################################ INITIALIZATION ################################
print('\n')
print('---begin--------------------------------------------------------------')

# initialize some variables
dt_x = .0002        # 5khz dynam update
dt_m = .002         # 500 hz measurement update
dt_u = .004         # ~200 hz control update
samp = int(dt_m/dt_x)     # effective sample rate of simulated data

print('Simulation update step is: ', dt_x, ' and control update is: ', dt_u, 'the ratio is: ', dt_u/dt_x)

# dynamics object
iono1 = IonoCraft(dt_x, x_noise = 1e-9)
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
N = 200     # num sequneces


# generate training data
print('...Generating Training Data')
Seqs_X, Seqs_U = generate_data(iono1, dt_m, dt_control = dt_u, sequence_len=500, num_iter = N, variance = .00010)

# for redundancy
Seqs_X = np.array(Seqs_X)
Seqs_U = np.array(Seqs_U)

np.savez('testingfile', Seqs_X, Seqs_U)
#
# print('.... loading training data')
# npzfile = np.load('testingfile.npz')
# Seqs_X = npzfile['arr_0']
# Seqs_U = npzfile['arr_1']

# converts data from list of trajectories of [next_states, states, inputs]
#       to a large array of [next_states, states, inputs]
# downsamples by a factor of samp. This is normalizing the differnece betweem dt_x and dt_measure
# data = sequencesXU2array(Seqs_X[:,::samp,:], Seqs_U[:,::samp,:])

# Check shape of data
# print(np.shape(data))
# quit()

################################ LEARNING ################################


# #creating neural network with 2 layers of 100 linearly connected ReLU units
print('...Training Model')
layer_sizes = [12, 100, 100, 9]
layer_types = ['nn.Linear()','nn.ReLU()', 'nn.ReLU()', 'nn.Linear()']
states_learn = ['yaw', 'pitch', 'roll', 'ax', 'ay', 'az']
# ['X', 'Y', 'Z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll', 'w_z', 'w_x', 'w_y']
forces_learn = ['Thrust', 'taux', 'tauy']
# ['F1', 'F2', 'F3', 'F4']
nn = NeuralNet(layer_sizes, layer_types, iono1, states_learn, forces_learn)

# Create New model
# acc = nn.train(list(zip(inputs, outputs)), learning_rate=1e-4, epochs=100)
acc = nn.train((Seqs_X[:,::samp,:], Seqs_U[:,::samp,:]), learning_rate=2.5e-5, epochs=300, batch_size = 1500, optim="Adam")
# plt.plot(acc)
# plt.show()

# nn.save_model('testingnn.pth')

# Or load model
# nn.load_model('testingnn.pth')


#
# # create a learning model
# lin1 = LeastSquares(dt_x, u_dim = 3)
#
# # train it like this
# lin1.train(l2array(data[:,0]),l2array(data[:,1]),l2array(data[:,2]))

################################ Obj Fnc ################################
origin_minimizer = Objective(np.linalg.norm, 'min', 6, dim_to_eval=[6,7,8,12,13,14])
print('...Objective Function Initialized')

################################ MPC ################################

# initialize MPC object with objective function above
mpc1 = MPController(nn, iono1, dt_x, dt_u, origin_minimizer, N=75, T=3, variance = .00003)
print('...MPC Running')
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
