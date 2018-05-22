# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
# from future.builtins import range
from builtins import range, super
# ----------------------------------------------------------------------------------------------------------------------

# Start importing packages
import numpy as np
import math
from dynamics import *

# Original version inhereted from Somil Bansal - Tomlin Group
__author__ = 'Somil Bansal'
__version__ = '0.1'

class CrazyFlie(Dynamics):
    def __init__(self, dt, m=.035, L=.065, Ixx = 2.3951e-5, Iyy = 2.3951e-5, Izz = 3.2347e-5, x_noise = .0001, u_noise=0):
        _state_dict = {
                    'X': [0, 'pos'],
                    'Y': [1, 'pos'],
                    'Z': [2, 'pos'],
                    'vx': [3, 'vel'],
                    'vy': [4, 'vel'],
                    'vz': [5, 'vel'],
                    'yaw': [6, 'angle'],
                    'pitch': [7, 'angle'],
                    'roll': [8, 'angle'],
                    'w_x': [9, 'omega'],
                    'w_y': [10, 'omega'],
                    'w_z': [11, 'omega']
        }
        # user can pass a list of items they want to train on in the neural net, eg learn_list = ['vx', 'vy', 'vz', 'yaw'] and iterate through with this dictionary to easily stack data

        # input dictionary less likely to be used because one will not likely do control without a type of acutation. Could be interesting though
        _input_dict = {
                    'Thrust': [0, 'force'],
                    'taux': [1, 'torque'],
                    'tauy': [2, 'torque'],
                    'tauz': [3, 'torque']
        }
        super().__init__(_state_dict, _input_dict, dt, x_dim=12, u_dim=4, x_noise = x_noise, u_noise=u_noise)

        # Setup the state indices
        self.idx_xyz = [0, 1, 2]
        self.idx_xyz_dot = [3, 4, 5]
        self.idx_ptp = [6, 7, 8]
        self.idx_ptp_dot = [9, 10, 11]

        # Setup the parameters
        self.m = m
        self.L = L
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = 9.81

        # Define equilibrium input for quadrotor around hover
        self.u_e = np.array([m*self.g, 0, 0, 0])

        # Hover control matrices
        self._hover_mats = [np.array([1, 0, 0, 0]),      # z
                            np.array([0, 1, 0, 0]),   # pitch
                            np.array([0, 0, 1, 0])]   # roll

    def pqr2rpy(self, x0, pqr):
        rotn_matrix = np.array([[1., math.sin(x0[0]) * math.tan(x0[1]), math.cos(x0[0]) * math.tan(x0[1])],
                                [0., math.cos(x0[0]),                   -math.sin(x0[0])],
                                [0., math.sin(x0[0]) / math.cos(x0[1]), math.cos(x0[0]) / math.cos(x0[1])]])
        return rotn_matrix.dot(pqr)

    def simulate(self, x, u, t=None):
        # Input structure:
        # u1 = thrust
        # u2 = torque-wx
        # u3 = torque-wy
        # u4 = torque-wz
        self._enforce_dimension(x, u)
        dt = self.dt
        u0 = u
        x0 = x
        idx_xyz = self.idx_xyz
        idx_xyz_dot = self.idx_xyz_dot
        idx_ptp = self.idx_ptp
        idx_ptp_dot = self.idx_ptp_dot

        m = self.m
        L = self.L
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        g = self.g

        Tx = np.array([Iyy / Ixx - Izz / Ixx, L / Ixx])
        Ty = np.array([Izz / Iyy - Ixx / Iyy, L / Iyy])
        Tz = np.array([Ixx / Izz - Iyy / Izz, 1. / Izz])

        # Add noise to input
        u_noise_vec = np.random.normal(loc=0, scale = self.u_noise, size=(self.u_dim))
        u = u+u_noise_vec

        # Array containing the forces
        Fxyz = np.zeros(3)
        Fxyz[0] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.cos(x0[idx_ptp[2]]) + math.sin(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[1] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.sin(x0[idx_ptp[2]]) - math.sin(x0[idx_ptp[0]]) * math.cos(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[2] = g - 1 * (math.cos(x0[idx_ptp[0]]) * math.cos(x0[idx_ptp[1]])) * u0[0] / m

        # Compute the torques
        t0 = np.array([x0[idx_ptp_dot[1]] * x0[idx_ptp_dot[2]], u0[1]])
        t1 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[2]], u0[2]])
        t2 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[1]], u0[3]])
        Txyz = np.array([Tx.dot(t0), Ty.dot(t1), Tz.dot(t2)])

        x1 = np.zeros(12)
        x1[idx_xyz_dot] = x0[idx_xyz_dot] + dt * Fxyz
        x1[idx_ptp_dot] = x0[idx_ptp_dot] + dt * Txyz
        x1[idx_xyz] = x0[idx_xyz] + dt * x0[idx_xyz_dot]
        x1[idx_ptp] = x0[idx_ptp] + dt * self.pqr2rpy(x0[idx_ptp], x0[idx_ptp_dot])

        # Add noise component
        x_noise_vec = np.random.normal(loc=0, scale = self.x_noise, size=(self.x_dim))
        return x1+x_noise_vec

    @property
    def non_linear(self):
        return True
