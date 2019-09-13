import numpy as np
import torch
import torch.optim as optim
import joblib
import math
import gym
from gym import spaces, logger
from gym.utils import seeding

class CrazyflieRigidEnv(gym.Env):
    """
    Description:
       A flying robot with 4 thrusters moves through space
    Source:
        This file is created by Nathan Lambert, adapted from a model from Somil Bansal
    Observation: 
        Type: Box(12)
        Num	Observation                 Min         Max
        0	x-pos                       -10         10      (meters)
        1	y-pos                       -10         10      (meters)
        2	z-pos                       -10         10      (meters)
        3	x-vel                       -Inf        Inf     (meters/sec)
        4   y-vel                       -Inf        Inf     (meters/sec)
        5   z-vel                       -Inf        Inf     (meters/sec)
        6   yaw                         -180        180     (degrees)
        7   pitch                       -90         90      (degrees)
        8   roll                        -180        180     (degrees)
        9   omega_x                     -Inf        Inf     (rad/s^2)
        10  omega_y                     -Inf        Inf     (rad/s^2)
        11  omega_z                     -Inf        Inf     (rad/s^2)
        
    Actions:
        Type: box([-1,1])
        Num	Action 
        -1	Push cart to the left max force
        1	Push cart to the right max force
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    def __init__(self, dt=.01, m=.035, L=.065, Ixx=2.3951e-5, Iyy=2.3951e-5, Izz=3.2347e-5, x_noise=.0001, u_noise=0):
        self.x_dim = 12
        self.u_dim = 4
        self.dt = dt

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
        # This is not the case for PWM inputs
        self.u_e = np.array([m*self.g, 0, 0, 0])
        # Four PWM inputs around hover, extracted from mean of clean_hover_data.csv
        # self.u_e = np.array([42646, 40844, 47351, 40116])

        # Hover control matrices
        self._hover_mats = [np.array([1, 0, 0, 0]),      # z
                            np.array([0, 1, 0, 0]),   # pitch
                            np.array([0, 0, 1, 0])]   # roll

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        # self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        state = self.state

        # TODO: Fill in dynamics for rigid quadrotor dynamics
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

        # # Add noise to input
        # u_noise_vec = np.random.normal(
        #     loc=0, scale=self.u_noise, size=(self.u_dim))
        # u = u+u_noise_vec

        # Array containing the forces
        Fxyz = np.zeros(3)
        Fxyz[0] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.cos(
            x0[idx_ptp[2]]) + math.sin(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[1] = -1 * (math.cos(x0[idx_ptp[0]]) * math.sin(x0[idx_ptp[1]]) * math.sin(
            x0[idx_ptp[2]]) - math.sin(x0[idx_ptp[0]]) * math.cos(x0[idx_ptp[2]])) * u0[0] / m
        Fxyz[2] = g - 1 * (math.cos(x0[idx_ptp[0]]) *
                           math.cos(x0[idx_ptp[1]])) * u0[0] / m

        # Compute the torques
        t0 = np.array([x0[idx_ptp_dot[1]] * x0[idx_ptp_dot[2]], u0[1]])
        t1 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[2]], u0[2]])
        t2 = np.array([x0[idx_ptp_dot[0]] * x0[idx_ptp_dot[1]], u0[3]])
        Txyz = np.array([Tx.dot(t0), Ty.dot(t1), Tz.dot(t2)])

        x1 = np.zeros(12)
        x1[idx_xyz_dot] = x0[idx_xyz_dot] + dt * Fxyz
        x1[idx_ptp_dot] = x0[idx_ptp_dot] + dt * Txyz
        x1[idx_xyz] = x0[idx_xyz] + dt * x0[idx_xyz_dot]
        x1[idx_ptp] = x0[idx_ptp] + dt * \
            self.pqr2rpy(x0[idx_ptp], x0[idx_ptp_dot])

        # Add noise component
        # x_noise_vec = np.random.normal(
        #     loc=0, scale=self.x_noise, size=(self.x_dim))

        # makes states less than 1e-12 = 0
        x1[x1 < 1e-12] = 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = 0
        self.steps_beyond_done = None
        return np.array(self.state)
