# File containing the base class dynamics
import numpy as np
import math
from math import cos
from math import sin


class Dynamics:
    # init class
    def __init__(self,dt, x_dim=12, u_dim = 4):
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim

    @property
    def dims(self):
        return self.x_dim, self.u_dim

    # dimension check raises error if incorrect
    def _enforce_dimension(self, x, u):
        if np.size(x) != self.x_dim:
            raise ValueError('x dimension passed into dynamics does not align with initiated value - given: ' + str(np.size(x)) + ', desired: ' + str(self.x_dim) )
        if np.size(u) != self.u_dim:
            raise ValueError('u dimension passed into dynamics does not align with initiated value - given: ' + str(np.size(u)) + ', desired: ' + str(self.u_dim) )

def Q_BI(ypr):
    # returns the Q body to inertial frame transformation matrix
    psi = ypr[0]       # yaw
    theta = ypr[1]     # pitch
    phi = ypr[2]       # roll
    Q = np.array([  [cos(theta)*cos(psi), cos(psi)*sin(theta)*sin(phi)-cos(phi)*sin(psi), sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta)],
                    [cos(theta)*sin(psi), cos(phi)*cos(psi)+sin(theta)*sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi)-cos(psi)*sin(phi)],
                    [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)] ])
    return Q

def Q_IB(ypr):
    # returns the Q inertial to body frame transformation matrix
    psi = ypr[0]       # yaw
    theta = ypr[1]     # pitch
    phi = ypr[2]       # roll
    Q = np.array([  [cos(psi)*cos(theta), cos(theta)*sin(psi), -sin(theta)],
                    [cos(psi)*sin(phi)*sin(theta)-cos(phi)*sin(psi), cos(phi)*cos(psi)+sin(theta)*sin(phi)*sin(psi), cos(theta)*sin(phi)],
                    [sin(phi)*sin(psi)+cos(psi)*cos(phi)*sin(theta), cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi), cos(phi)*cos(theta)] ])
    return Q

def W_inv(ypr):
    # transformation from angular velocities w_BI to Euler angle rates dot(E)
    psi = ypr[0]       # yaw
    theta = ypr[1]     # pitch
    phi = ypr[2]       # roll
    W_inv = (1./cos(theta))*np.array([  [0, sin(phi), cos(phi)],
            [0, cos(phi)*cos(theta), -sin(phi)*cos(theta)],
            [cos(theta), sin(phi)*sin(theta), cos(phi)*sin(theta)] ])

    return W_inv

def generate_data(dynam, sequence_len, num_iter=1, controller = 'random'):
    # generates a batch of data sequences for learning. Will be an array of (sequence_len x 2) sequences with state and inputs
    return []

def sim_sequence(dynam, sequence_len, control = 'random'):
    # Simulates a squence following the control sequence provided
    return []
