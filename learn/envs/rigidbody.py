import numpy as np
import torch
import torch.optim as optim
import math
import gym
from gym import spaces, logger
from gym.utils import seeding


class RigidEnv(gym.Env):
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
        # Quad force - note different robots have different IMU orientation
        Type: box([0,maxpwm])
        Num	Action
        1   m1 motor voltage
        2   m2 motor voltage
        3   m3 motor voltage
        4   m4 motor voltage

    """

    def __init__(self, dt=.001, x_noise=.0005, u_noise=0):
        self.x_dim = 12
        self.u_dim = 4
        self.dt = dt
        self.x_noise = x_noise

        # Setup the state indices
        self.idx_xyz = [0, 1, 2]
        self.idx_xyz_dot = [3, 4, 5]
        self.idx_ptp = [6, 7, 8]
        self.idx_ptp_dot = [9, 10, 11]

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, pwm):
        # assert self.action_space.contains(u), "%r (%s) invalid" % (u, type(u))

        # We need to convert from upright orientation to N-E-Down that the simulator runs in
        # For reference, a negative thrust of -mg/4 will keep the robot stable
        u = self.pwm_thrust_torque(pwm)
        state = self.state

        dt = self.dt
        u0 = u
        x0 = state
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
        x1[idx_ptp] = x0[idx_ptp] + dt * self.pqr2rpy(x0[idx_ptp], x0[idx_ptp_dot])

        # Add noise component
        x_noise_vec = np.random.normal(
            loc=0, scale=self.x_noise, size=(self.x_dim))

        x1 += x_noise_vec
        # makes states less than 1e-12 = 0
        x1[abs(x1) < 1e-12] = 0
        self.state = x1

        obs = self.get_obs()
        reward = self.get_reward(obs, u)
        done = self.get_done(obs)

        return obs, reward, done, {}

    def get_obs(self):
        raise NotImplementedError("Subclass must implement this function")

    def set_state(self, x):
        self.state = x

    def reset(self):
        x0 = np.array([0, 0, 0])
        v0 = self.np_random.uniform(low=-0.01, high=0.01, size=(3,))
        # ypr0 = self.np_random.uniform(low=-0.0, high=0.0, size=(3,))
        ypr0 = self.np_random.uniform(low=-np.pi/16., high=np.pi/16, size=(3,))
        ypr0[-1] = 0 # 0 out yaw
        w0 = self.np_random.uniform(low=-0.01, high=0.01, size=(3,))

        self.state = np.concatenate([x0, v0, ypr0, w0])
        self.steps_beyond_done = None
        return self.get_obs()

    def get_reward(self, next_ob, action):
        raise NotImplementedError("Subclass must implement this function")

    def get_reward_torch(self, next_ob, action):
        raise NotImplementedError("Subclass must implement this function")

    def get_done(self, state):
        # Done is pitch or roll > 35 deg
        max_a = np.deg2rad(45)
        d = (abs(state[1]) > max_a) or (abs(state[0]) > max_a)
        return d

    def pqr2rpy(self, x0, pqr):
        rotn_matrix = np.array([[1., math.sin(x0[0]) * math.tan(x0[1]), math.cos(x0[0]) * math.tan(x0[1])],
                                [0., math.cos(x0[0]), -math.sin(x0[0])],
                                [0., math.sin(x0[0]) / math.cos(x0[1]), math.cos(x0[0]) / math.cos(x0[1])]])
        return rotn_matrix.dot(pqr)

    def pwm_thrust_torque(self, PWM):
        raise NotImplementedError("Subclass must implement this function")
