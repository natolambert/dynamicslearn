from dynamics import *
from controllers import randController, MPController
from dynamics_ionocraft import IonoCraft
from dynamics_ionocraft_imu import IonoCraft_IMU
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
dt_x = .001
dt_u = .01
print('Simulation update step is: ', dt_x, ' and control update is: ', dt_u, 'the ratio is: ', dt_u/dt_x)

# dynamics object
iono1 = IonoCraft(dt_x, x_noise = .0001)
print('...Initializing Dynamics Object')

mgo4 = iono1.m*iono1.g/4

mgo4 = iono1.m*iono1.g/4

# initial state is origin
x0 = np.zeros(12)
u0 = np.array([mgo4+.0001,mgo4,mgo4,mgo4]) #np.zeros(4)

# good for unit testin dynamics
x1 = iono1.simulate(x0,u0)
# x1[x1 < .00001] = 0
x2 = iono1.simulate(x1,u0)
# x2[x2 < .00001] = 0
x3 = iono1.simulate(x2,u0)

# prints state in readible form
# printState(x3)

# Simulate data for training
N = 250     # num sequneces

# generate training data
print('...Generating Training Data')
Seqs_X, Seqs_U = generate_data(iono1, dt_control = dt_u, sequence_len=100, num_iter = N)

# converts data from list of trajectories of [next_states, states, inputs]
#       to a large array of [next_states, states, inputs]
data = sequencesXU2array(Seqs_X, Seqs_U)

# Check shape of data
# print(np.shape(data))

################################ LEARNING ################################

# #creating neural network with 2 layers of 100 linearly connected ReLU units
print('...Training Model')
layer_sizes = [16, 100, 100, 12]
layer_types = ['nn.Linear()', 'nn.ReLU()', 'nn.ReLU()', 'nn.Linear()']
states_learn = ['X', 'Y', 'Z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'roll', 'w_z', 'w_x', 'w_y']
forces_learn = ['F1', 'F2', 'F3', 'F4']
nn = NeuralNet(layer_sizes, layer_types, iono1, states_learn, forces_learn)

# acc = nn.train(list(zip(inputs, outputs)), learning_rate=1e-4, epochs=100)
Seqs_X = np.array(Seqs_X)
Seqs_U = np.array(Seqs_U)
acc = nn.train((Seqs_X, Seqs_U), learning_rate=1e-4, epochs=250)
# create a learning model
# lin1 = LeastSquares()
#
# # train it like this
# lin1.train(l2array(data[:,0]),l2array(data[:,1]),l2array(data[:,2]))

################################ Obj Fnc ################################
origin_minimizer = Objective(np.linalg.norm, 'min', 6, dim_to_eval=[0,1,2,3,4,5])
print('...Objective Function Initialized')

################################ MPC ################################

# initialize MPC object with objective function above
mpc1 = MPController(nn, iono1, dt_u, origin_minimizer)
print('...MPC Running')
x0 = np.zeros(12)
new_seq, Us = sim_sequence(iono1, dt_u, sequence_len = 150, x0=x0, controller = mpc1)
#
# compareTraj(Us, x0, iono1, nn, show=True)
################################ Sim Controlled ################################

# Sim sequence off the trained controller
new_len = 500
x_controlled, u_seq = sim_sequence(iono1, dt_u, controller = mpc1, sequence_len = new_len, to_print = False)
print(u_seq)
print('Simulated Learned.')
################################ PLot ################################
print('...Plotting')
# plot states and inputs of the trajectory if wanted
T = np.linspace(0,new_len*dt_x,new_len)
plot12(x_controlled, T)
plotInputs(u_seq, T)

# Plots animation, change save to false to not save .gif
plotter1 = PlotFlight(x_controlled,.5)
plotter1.show(save=True)
print('Saved Gif')


print('---------------------------------------------------------end run-----')
