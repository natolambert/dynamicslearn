import numpy as np
import torch
import torch.optim as optim
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
        # BELOW NOT UPDATED TODO
        Type: box([-1,1])
        Num	Action 
        -1	Push cart to the left max force
        1	Push cart to the right max force

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
        self.u_e = np.array([m * self.g, 0, 0, 0])
        # Four PWM inputs around hover, extracted from mean of clean_hover_data.csv
        # self.u_e = np.array([42646, 40844, 47351, 40116])

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                       high=np.array([65535, 65535, 65535, 65535]),
                                       dtype=np.int32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, pwm):
        # assert self.action_space.contains(u), "%r (%s) invalid" % (u, type(u))
        u = self.pwm_thrust_torque(pwm)
        state = self.state

        # TODO: Fill in dynamics for rigid quadrotor dynamics
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
        # x_noise_vec = np.random.normal(
        #     loc=0, scale=self.x_noise, size=(self.x_dim))

        # makes states less than 1e-12 = 0
        x1[abs(x1) < 1e-12] = 0
        self.state = x1

        obs = self.get_obs()
        reward = self.get_reward(obs, u)
        done = self.get_done(obs)

        return obs, reward, done, {}

    def get_obs(self):
        return np.array(self.state[6:])

    def set_state(self, x):
        self.state = x

    def reset(self):
        x0 = np.array([0, 0, 0])
        v0 = self.np_random.uniform(low=-0.01, high=0.01, size=(3,))
        ypr0 = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        w0 = self.np_random.uniform(low=-0.01, high=0.01, size=(3,))

        self.state = np.concatenate([x0, v0, ypr0, w0])
        self.steps_beyond_done = None
        return self.get_obs()

    def get_reward(self, next_ob, action):
        # Going to make the reward -c(x) where x is the attitude based cost
        assert isinstance(next_ob, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert next_ob.ndim in (1, 2)

        was1d = next_ob.ndim == 1
        if was1d:
            next_ob = np.expand_dims(next_ob, 0)
            action = np.expand_dims(action, 0)

        assert next_ob.ndim == 2
        cost_pr = np.power(next_ob[:, 1], 2) + np.power(next_ob[:, 2], 2)
        cost_rates = np.power(next_ob[:, 3], 2) + np.power(next_ob[:, 4], 2) + np.power(next_ob[:, 5], 2)
        lambda_omega = .0000
        cost = cost_pr + lambda_omega * cost_rates
        return -cost

    def get_reward_torch(self, next_ob, action):
        assert torch.is_tensor(next_ob)
        assert torch.is_tensor(action)
        assert next_ob.dim() in (1, 2)

        was1d = len(next_ob.shape) == 1
        if was1d:
            next_ob = next_ob.unsqueeze(0)
            action = action.unsqueeze(0)

        cost_pr = next_ob[:, 1].pow(2) + next_ob[:, 2].pow(2)
        cost_rates = next_ob[:, 3].pow(2) + next_ob[:, 4].pow(2) + next_ob[:, 5].pow(2)
        lambda_omega = .0000
        cost = cost_pr + lambda_omega * cost_rates
        return -cost

    def get_done(self, state):
        # Done is pitch or roll > 35 deg
        # pitch is state 7, roll is state 8
        max_a = 35
        d = (abs(state[1]) > max_a) or (abs(state[2]) > max_a)
        return d

    def pqr2rpy(self, x0, pqr):
        rotn_matrix = np.array([[1., math.sin(x0[0]) * math.tan(x0[1]), math.cos(x0[0]) * math.tan(x0[1])],
                                [0., math.cos(x0[0]), -math.sin(x0[0])],
                                [0., math.sin(x0[0]) / math.cos(x0[1]), math.cos(x0[0]) / math.cos(x0[1])]])
        return rotn_matrix.dot(pqr)

    def pwm_thrust_torque(self, PWM):
        # Takes in the a 4 dimensional PWM vector and returns a vector of
        # [Thrust, Taux, Tauy, Tauz] which is used for simulating rigid body dynam
        # Sources of the fit: https://wiki.bitcraze.io/misc:investigations:thrust,
        #   http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8905295&fileOId=8905299

        # The quadrotor is 92x92x29 mm (motor to motor, square along with the built in prongs). The the distance from the centerline,

        # Thrust T = .35*d + .26*d^2 kg m/s^2 (d = PWM/65535 - normalized PWM)
        # T = (.409e-3*pwm^2 + 140.5e-3*pwm - .099)*9.81/1000 (pwm in 0,255)

        def pwm_to_thrust(PWM):
            # returns thrust from PWM
            pwm_n = PWM / 65535.0
            thrust = .35 * pwm_n + .26 * pwm_n ** 2
            return thrust

        l = 35.527e-3 / np.sqrt(2)  # length to motors / axis of rotation for xy
        lz = 46e-3  # axis for tauz
        c = .05  # coupling coefficient for yaw torque

        # Torques are slightly more tricky
        # x = m2+m3-m1-m4
        # y =m1+m2-m3-m4

        # Estimates forces
        m1 = pwm_to_thrust(PWM[0])
        m2 = pwm_to_thrust(PWM[1])
        m3 = pwm_to_thrust(PWM[2])
        m4 = pwm_to_thrust(PWM[3])

        Thrust = (m1 + m2 + m3 + m4)  # pwm_to_thrust(np.sum(PWM) / (4 * 65535.0))
        # taux = -l * (m2 + m3 - m4 - m1)
        # tauy = -l * (m1 + m2 - m3 - m4)
        # tauz = -lz * c * (m1 + m3 - m2 - m4)
        taux = l * (m1 + m3 - m4 - m2)
        tauy = l * (m2 + m3 - m4 - m1)
        tauz = lz * c * (m2 + m4 - m1 - m3)
        return -np.array([Thrust, taux, tauy, tauz])
