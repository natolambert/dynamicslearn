import torch
import gym
from gym.spaces import Box
import pickle
import numpy as np
import random


class ModelEnv(gym.Env):
    """
    Gym environment for our custom system (Crazyflie quad).

    """

    def __init__(self, env, cfg, model, metric):
        gym.Env.__init__(self)
        self.cfg = cfg
        self.model = model
        self.env = env
        self.reward_fnc = metric

    def set_state(self, state):
        self.state = state

    def reset(self):
        obs = self.reset()
        self.state = obs
        # self.dynam.reset()
        return obs

    def get_obs(self):
        return self.state

    def reset(self):
        # ypr0 = self.np_random.uniform(low=-0.0, high=0.0, size=(3,))
        ypr0 = self.np_random.uniform(low=-np.pi / 16., high=np.pi / 16., size=(3,))
        ypr0[-1] = 0  # 0 out yaw
        w0 = self.np_random.uniform(low=-0.01, high=0.01, size=(3,))

        self.state = np.concatenate([ypr0, w0])
        self.steps_beyond_done = None
        return self.get_obs()

    def get_done(self, state):
        # Done is pitch or roll > 35 deg
        max_a = np.deg2rad(45)
        if torch.is_tensor(state):
            d = (torch.abs(state[:, 1]) > 40) | (torch.abs(state[:, 0]) > 40)
        else:
            d = (abs(state[1]) > max_a) or (abs(state[0]) > max_a)
        return d

    def step(self, action):
        if self.state_failed(new_state):
            done = True

        reward = self.reward_fnc(self.state, action)

        return self.state, reward, done, {}

    def step_from(self, state, action):
        # Does not work with history mode on
        output, logvars = self.model.predict(state, action, ret_var=True)
        next_state = state + output

        obs = next_state
        reward = self.reward_fnc(obs, action)
        done = self.get_done(obs)
        return obs, reward, done, {}


def push_history(new, orig):
    """
    Takes in the new data and makes it the first elements of a vector.
    - For using dynamics models with history.
    :param new: New data
    :param orig: old data (with some form of history)
    :return: [new, orig] cut at old length
    """
    assert len(orig) / len(new) % 1.0 == 0
    hist = int(len(orig) / len(new))
    l = len(new)
    data = np.copy(orig)
    data[l:] = orig[:-l]
    data[:l] = new
    return data
