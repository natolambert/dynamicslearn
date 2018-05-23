# File containing the base class dynamics
import numpy as np
import math
from math import cos
from math import sin
from math import pi
from controllers import *


__author__ = 'Nathan Lambert'
__version__ = '0.1'

class Dynamics:
    # Primary basis of this class is to check variable dimensions and time steps for dynamics.

    # init class
    def __init__(self, state_dict, input_dict, dt=.01, x_dim=12, u_dim = 4, x_noise = .0001, u_noise=0):
        self.dt = dt                # time update step
        self.x_dim = x_dim          # x dimension, can be derived from the state_dict
        self.u_dim = u_dim          # u dimension
        self.x_noise = x_noise
        self.u_noise = u_noise
        self.x_dict = state_dict
        self.u_dict = input_dict

    @property
    def get_dims(self):
        return self.x_dim, self.u_dim

    @property
    def get_dt(self):
        return self.dt

    # dimension check raises error if incorrect
    def _enforce_dimension(self, x, u):
        if np.size(x) != self.x_dim:
            raise ValueError('x dimension passed into dynamics does not align with initiated value - given: ' + str(np.size(x)) + ', desired: ' + str(self.x_dim) )
        if np.size(u) != self.u_dim:
            raise ValueError('u dimension passed into dynamics does not align with initiated value - given: ' + str(np.size(u)) + ', desired: ' + str(self.u_dim) )

def Q_BI(ypr):
    # returns the Q body to inertial frame transformation matrix. Used int dynamics simulations
    psi = ypr[0] % (2*pi)       # yaw
    theta = ypr[1] % (2*pi)     # pitch
    phi = ypr[2]  % (2*pi)      # roll
    Q = np.array([  [cos(theta)*cos(psi), cos(psi)*sin(theta)*sin(phi)-cos(phi)*sin(psi), sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta)],
                    [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(theta)*sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi)-cos(psi)*sin(phi)],
                    [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)] ])
    return Q

def Q_IB(ypr):
    # returns the Q inertial to body frame transformation matrix. Used in dynamics simulations
    psi = ypr[0] % (2*pi)       # yaw
    theta = ypr[1] % (2*pi)     # pitch
    phi = ypr[2] % (2*pi)       # roll
    Q = np.array([  [cos(psi)*cos(theta), cos(theta)*sin(psi), -sin(theta)],
                    [cos(psi)*sin(phi)*sin(theta)-cos(phi)*sin(psi), cos(phi)*cos(psi)+sin(theta)*sin(phi)*sin(psi), cos(theta)*sin(phi)],
                    [sin(phi)*sin(psi)+cos(psi)*cos(phi)*sin(theta), cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi), cos(phi)*cos(theta)] ])
    return Q

def W_inv(ypr):
    # transformation from angular velocities w_BI to Euler angle rates dot(E)
    psi = ypr[0] % (2*pi)       # yaw
    theta = ypr[1] % (2*pi)    # pitch
    phi = ypr[2] % (2*pi)       # roll
    W_inv = (1./cos(theta))*np.array([  [0, sin(phi), cos(phi)],
            [0, cos(phi)*cos(theta), -sin(phi)*cos(theta)],
            [cos(theta), sin(phi)*sin(theta), cos(phi)*sin(theta)] ])

    return W_inv

def generate_data(dynam, dt_m, dt_control, sequence_len=10, num_iter=100, variance = .000001, controller = 'random'):
    # generates a batch of data sequences for learning. Will be an array of (sequence_len x 2) sequences with state and inputs
    if controller == 'random':
        controller = randController(dynam, dynam.dt, dt_control, variance = variance)

    Seqs_X = []
    Seqs_U = []
    for i in range(num_iter):
        X, U = sim_sequence(dynam, dynam.dt, dt_control, sequence_len, x0= [], variance = variance, controller=controller)
        Seqs_X.append(X)
        Seqs_U.append(U)

    return Seqs_X, Seqs_U

def sim_sequence(dynam, dt_m, dt_u, sequence_len=10, x0=[], variance = .0001, controller = 'random', to_print = False):
    # Simulates a squence following the control sequence provided
    # returns the list of states and inputs as a large array
    # print('new seq')
    if controller == 'random':
        controller = randController(dynam, dynam.dt, dt_u, variance=.00005)
        print('Running Random control for designated dynamics...')
    if (x0 == []):
        x0 = np.zeros(dynam.get_dims[0])

    # inititialize initial control to be equilibrium amount
    u = controller.control

    # intitialize arrays to append the sequences
    U = np.array([u])
    X = np.array([x0])
    for n in range(sequence_len-1):

        # update state
        x_prev = X[-1]
        x = dynam.simulate(x_prev,u)

        # generate new u
        u = controller.update(x_prev)
        # print(u)
        # contstruct array
        X = np.append(X, [x], axis=0)
        U = np.append(U, [u], axis=0)

        if to_print:
            print('State is: ', x_prev)
            print('Control is: ', u)


    return X, U
