# ----------------------------------------------------------------------------------------------------------------------

'''
UNDER CONSTRUCTION: linearized dynamics model around the hover point for the ionocraft

Test cases passed:
- gravity fall only
- hover condition

'''

# Start importing packages
import numpy as np
import math
from dynamics import *

# Original version inhereted from Somil Bansal - Tomlin Group
__author__ = 'Nathan Lambert'
__version__ = '0.1'

class IonoCraft(Dynamics):
    def __init__(self, dt, m=67e-6, L=.01, Ixx = 5.5833e-10, Iyy = 5.5833e-10, Izz = 1.1167e-09, angle = 0, linear = False):
        super().__init__(dt, x_dim=12, u_dim=4)

        # Setup the state indices
        self.idx_xyz = [0, 1, 2]
        self.idx_xyz_dot = [3, 4, 5]
        self.idx_ptp = [6, 7, 8]
        self.idx_ptp_dot = [9, 10, 11]

        # Setup the parameters
        self.m = m
        self.L = L
        self.angle = angle
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = 9.81
        self.Ib = np.array([
            [Ixx, 0, 0],
            [0, Iyy, 0],
            [0, 0, Izz]
        ])

    def force2thrust_torque(self, angle):
        # transformation matrix for ionocraft with XY thrusts
        # [Thrustx; Thrusty; Thrustz; Tauz; Tauy; Taux;] = M * [F4; F3; F2; F1]
        M =  np.array([[0.,         math.sin(angle),           0.,                     -math.sin(angle)],
                    [-math.sin(angle),         0.,                        math.sin(angle),           0.],
                    [math.cos(angle),          math.cos(angle),           math.cos(angle),           math.cos(angle)],
                    [-self.L*math.sin(angle),  self.L*math.sin(angle),    -self.L*math.sin(angle),   self.L*math.sin(angle)],
                    [-self.L*math.cos(angle),  -self.L*math.cos(angle),   self.L*math.cos(angle),    self.L*math.cos(angle)],
                    [self.L*math.cos(angle),   -self.L*math.cos(angle),   -self.L*math.cos(angle),   self.L*math.cos(angle)]])
        return M

    def pqr2rpy(self, x0, pqr):
        rotn_matrix = np.array([[1., math.sin(x0[0]) * math.tan(x0[1]), math.cos(x0[0]) * math.tan(x0[1])],
                                [0., math.cos(x0[0]),                   -math.sin(x0[0])],
                                [0., math.sin(x0[0]) / math.cos(x0[1]), math.cos(x0[0]) / math.cos(x0[1])]])

        return rotn_matrix.dot(pqr)

    def simulate(self, x, u, t=None):
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
        angle = self.angle
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        Ib = self.Ib
        g = self.g

        # Easy access to yawpitchroll vector
        ypr = x0[idx_ptp]

        # Transform the input into body frame forces
        T_tau_thrusters = np.zeros(6)
        T_tau_thrusters = np.matmul(self.force2thrust_torque(angle),u)
        Tx = T_tau_thrusters[0]
        Ty = T_tau_thrusters[1]
        Tz = T_tau_thrusters[2]
        Tauz = T_tau_thrusters[3]
        Tauy = T_tau_thrusters[4]
        Taux = T_tau_thrusters[5]
        Tau = np.array([Taux, Tauy, Tauz])

        # External forces acting on robot
        F_ext = np.zeros(3)
        F_ext = np.matmul(Q_IB(ypr),np.array([0, 0, m*g])) - np.array([Tx,Ty,Tz])

        # Implement free body dynamics
        x1 = np.zeros(12)
        xdot = np.zeros(12)

        # global position derivative
        xdot[idx_xyz] = np.matmul(Q_BI(ypr),x0[idx_xyz_dot])

        # Euler angle derivative
        xdot[idx_ptp] = np.matmul(W_inv(ypr), x0[idx_ptp_dot])

        # body velocity derivative
        omega = x0[idx_ptp_dot]
        omega_mat = np.array([  [0, -omega[2], omega[1]],
                                [omega[2], 0, -omega[0]],
                                [-omega[2], omega[0], 0]
                                ])

        xdot[idx_xyz_dot] = (1/m)*F_ext - np.matmul(omega_mat, x0[idx_xyz_dot])

        # angular velocity derivative
        xdot[idx_ptp_dot] = np.matmul(np.linalg.inv(Ib),Tau) - np.matmul(np.matmul(np.linalg.inv(Ib),omega_mat),np.matmul(Ib,x0[idx_ptp_dot]))
        x1 = x0+dt*xdot

        return x1
