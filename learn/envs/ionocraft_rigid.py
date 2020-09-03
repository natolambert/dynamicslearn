import numpy as np
import torch
import torch.optim as optim
import math
import gym
from gym import spaces
from gym.utils import seeding
from .rigidbody import RigidEnv


class IonocraftRigidEnv(RigidEnv):
    def __init__(self, dt=.01, m=.00005, L=.01, Ixx=1.967 * 10 ** -9, Iyy=1.967 * 10 ** -9, Izz=3.775 * 10 ** -9):
        super(IonocraftRigidEnv, self).__init__(dt=dt)

        # Setup the parameters
        self.m = m
        self.L = L
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = 9.81

        self.inv_huber = False

        # Define equilibrium input for quadrotor around hover
        # This is not the case for PWM inputs
        # self.u_e = np.array([m * self.g, 0, 0, 0])

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]),
                                       high=np.array([3000, 3000, 3000, 3000]),
                                       dtype=np.int32)

    def get_obs(self):
        return np.array(self.state[6:])

    def set_state(self, x):
        self.state = x

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

        if not self.inv_huber:
            pitch = next_ob[:, 0]
            roll = next_ob[:, 1]
            # cost_pr = np.power(pitch, 2) + np.power(roll, 2)
            # cost_rates = np.power(next_ob[:, 3], 2) + np.power(next_ob[:, 4], 2) + np.power(next_ob[:, 5], 2)
            # lambda_omega = .0001
            # cost = cost_pr + lambda_omega * cost_rates
            flag1 = np.abs(pitch) < 5
            flag2 = np.abs(roll) < 5
            rew = int(flag1) + int(flag2)
            return rew
        else:
            pitch = np.divide(next_ob[:, 0], 180)
            roll = np.divide(next_ob[:, 1], 180)

            def invhuber(input):
                input = np.abs(input)
                if input.ndim == 1:
                    if np.abs(input) > 5:
                        return input ** 2
                    else:
                        return input
                else:
                    flag = np.abs(input) > 5
                    sqr = np.power(input, 2)
                    cost = input[np.logical_not(flag)] + sqr[flag]
                    return cost

            p = invhuber(pitch)
            r = invhuber(roll)
            cost = p + r
        return -cost

    def get_reward_torch(self, next_ob, action):
        assert torch.is_tensor(next_ob)
        assert torch.is_tensor(action)
        assert next_ob.dim() in (1, 2)

        was1d = len(next_ob.shape) == 1
        if was1d:
            next_ob = next_ob.unsqueeze(0)
            action = action.unsqueeze(0)

        if not self.inv_huber:
            # cost_pr = next_ob[:, 0].pow(2) + next_ob[:, 1].pow(2)
            # cost_rates = next_ob[:, 3].pow(2) + next_ob[:, 4].pow(2) + next_ob[:, 5].pow(2)
            # lambda_omega = .0001
            # cost = cost_pr + lambda_omega * cost_rates
            flag1 = torch.abs(next_ob[:, 0]) < 5
            flag2 = torch.abs(next_ob[:, 1]) < 5
            rew = (flag1).double() + (flag2).double()
            return rew
        else:
            def invhuber(input):
                input = torch.abs(input)
                if len(input) == 1:
                    if torch.abs(input) > 5:
                        return input.pow(2)
                    else:
                        return input
                else:
                    flag = torch.abs(input) > 5
                    sqr = input.pow(2)
                    cost = (~flag).double() * input + flag.double() * sqr
                    return cost

            p = invhuber(next_ob[:, 0])
            r = invhuber(next_ob[:, 1])
            cost = p + r

        return -cost

    def pwm_thrust_torque(self, PWM):
        # Takes in the a 4 dimensional PWM vector and returns a vector of
        # [Thrust, Taux, Tauy, Tauz] which is used for simulating rigid body dynam
        # u1 u2 u3 u4
        # u1 is the thrust along the zaxis in B, and u2, u3 and u4 are rolling, pitching and
        # yawing moments respectively

        def pwm_to_thrust(PWM, beta):
            # returns thrust from PWM
            mu = 2*10**-4
            I = (PWM/3000)*.5*10**-3    # fit PWM range to 0 to .5mA (rough estimate)
            F = (beta*I*(500*10**-6))/mu
            return F

        l = self.L / np.sqrt(2)  # length to motors / axis of rotation for xy
        lz = 0  # axis for tauz
        c = 0  # coupling coefficient for yaw torque
        beta = .6

        # Estimates forces
        m1 = pwm_to_thrust(PWM[0], beta)
        m2 = pwm_to_thrust(PWM[1], beta)
        m3 = pwm_to_thrust(PWM[2], beta)
        m4 = pwm_to_thrust(PWM[3], beta)

        Thrust = (-m1 - m2 - m3 - m4)  # pwm_to_thrust(np.sum(PWM) / (4 * 65535.0))
        taux = l * (-m1 - m2 + m3 + m4)
        tauy = l * (m1 - m2 - m3 + m4)
        tauz = -lz * c * (-m1 + m2 - m3 + m4)
        return np.array([Thrust, taux, tauy, tauz])
