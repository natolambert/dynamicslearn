import os
import sys
from dotmap import DotMap

import pandas as pd
import numpy as np
import torch
import math

from learn.control.random import RandomController
from learn.control.mpc import MPController
from learn import envs
from learn.trainer import train_model
import gym
import logging
import hydra

log = logging.getLogger(__name__)


######################################################################
@hydra.main(config_path='conf/mpc.yaml')
def mpc(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    env = gym.make('CrazyflieRigid-v0')
    env.reset()

    for s in range(cfg.experiment.seeds):
        log.info(f"Random Seed: {s}")
        data = rollout(env, RandomController(env, cfg.policy), cfg.experiment)
        X, dX, U = to_XUdX(data)

        # dx = np.shape(X)[1]
        # du = np.shape(U)[1]
        # dt = np.shape(dX)[1]
        #
        # # if set dimensions, double check them here
        # if model_cfg.params.dx != -1:
        #     assert model_cfg.params.dx == dx, "model dimensions in cfg do not match data given"
        # if model_cfg.params.du != -1:
        #     assert model_cfg.params.du == du, "model dimensions in cfg do not match data given"
        # if model_cfg.params.dt != -1:
        #     assert model_cfg.params.dt == dt, "model dimensions in cfg do not match data given"
        min_reward = -np.inf
        model, train_log = train_model(X, U, dX, cfg.model)

        for i in range(cfg.experiment.num_r):
            controller = MPController(env, model, cfg.policy)
            data_new = rollout(env, controller, cfg.experiment)

            X, dX, U = combine_data(data_new, (X, dX, U))
            msg = "Rollout completed of "
            msg += f"Cumulative reward {np.sum(np.stack(data_new[2]))}, "
            msg += f"Flight length {len(np.stack(data_new[2]))}"
            log.info(msg)
            model, train_log = train_model(X, U, dX, cfg.model)

            if np.sum(np.stack(data_new[2])) > min_reward:
                min_reward = np.sum(np.stack(data_new[2]))

        log.info(f"Max Reward Achieved: {min_reward}")



def to_XUdX(data):
    states = np.stack(data[0])
    X = states[:-1, :]
    dX = states[1:, :] - states[:-1, :]
    U = np.stack(data[1])[:-1, :]
    return X, dX, U


def combine_data(new_data, full_data):
    X_new, dX_new, U_new = to_XUdX(new_data)
    X = np.concatenate((full_data[0], X_new), axis=0)
    U = np.concatenate((full_data[2], U_new), axis=0)
    dX = np.concatenate((full_data[1], dX_new), axis=0)
    return X, dX, U


def rollout(env, controller, exp_cfg):
    done = False
    states = []
    actions = []
    rews = []
    state = env.reset()
    for t in range(exp_cfg.r_len):
        if done:
            break
        action = controller.get_action(state)
        states.append(state)
        actions.append(action)

        state, rew, done, _ = env.step(action)
        rews.append(rew)

    return states, actions, rews


if __name__ == '__main__':
    sys.exit(mpc())
