# ----------------------------------------------------------------------------------------------------------------------

'''
UNDER CONSTRUCTION: linearized dynamics model around the hover point for the ionocraft

'''

# Start importing packages
import numpy as np
import math
import dynamics as dynamics

# Original version inhereted from Somil Bansal - Tomlin Group
__author__ = 'Nathan Lambert'
__version__ = '0.1'

class IonoCraft(dynamics):
    def __init__(self, dt, m=67e-6, L=.01, Ixx = 5.5833e-10, Iyy = 5.5833e-10, Izz = 1.1167e-09, linear = False):
        super().__init__(dt, x_dim=12, u_dim=4)

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

    def force2thrust_torque(self,angle):
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
        dt = self._dt
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

        # Implement free body dynamics

        # if self.Linear:
        #
        # else:


        return x1
