import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import torch
import math
import time
from learn.control.random import RandomController
from learn.control.mpc import MPController
from learn import envs
from learn.trainer import train_model
import gym
import logging
import hydra
from learn.utils.plotly import plot_rewards_over_trials, plot_rollout, plot_lie, plot_results_yaw
from learn.utils.sim import *

log = logging.getLogger(__name__)


def save_log(cfg, trial_num, trial_log):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)


######################################################################
@hydra.main(config_path='conf/mpc.yaml', strict=False)
def mpc(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    # plot_results_yaw(pts=cfg.data)
    # quit()

    env_name = cfg.env.params.name
    env = gym.make(env_name)
    env.reset()

    env.seed(cfg.random_seed, inertial=cfg.experiment.inertial)

    standard_model = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/outputs/2020-11-02/13-11-28/'
    traj_model = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/outputs/2020-11-02/13-14-09/'

    transitions = []
    actions = []
    rewards = []
    import glob
    for f in os.listdir(standard_model):
        if f[-4:] == '.dat':
            data = torch.load(standard_model+f)
            transitions.append(data['raw_data'][0])
            actions.append(data['raw_data'][1])

    t = [np.stack(t).squeeze() for t in transitions]
    a = [[np.stack(a).squeeze() for a in actions]]

def to_XUdX(data):
    states = np.stack(data[0])
    X = states[:-1, :]
    dX = states[1:, :] - states[:-1, :]
    U = np.stack(data[1])[:-1, :]
    return X, dX, U


def combine_data(data_rs, full_data):
    X = full_data[0]
    U = full_data[2]
    dX = full_data[1]
    for data in data_rs:
        X_new, dX_new, U_new = to_XUdX(data)
        X = np.concatenate((X, X_new), axis=0)
        U = np.concatenate((U, U_new), axis=0)
        dX = np.concatenate((dX, dX_new), axis=0)
    return X, dX, U


if __name__ == '__main__':
    sys.exit(mpc())
