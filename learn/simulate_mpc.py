import os
import sys
from dotmap import DotMap

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

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


def save_log(cfg, trial_num, trial_log):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)


######################################################################
@hydra.main(config_path='conf/mpc.yaml')
def mpc(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    env = gym.make('CrazyflieRigid-v0')
    env.reset()
    trial_rewards = []

    # log.info(f"Random Seed: {s}")
    data = rollout(env, RandomController(env, cfg.policy), cfg.experiment)
    X, dX, U = to_XUdX(data)

    model, train_log = train_model(X, U, dX, cfg.model)

    for i in range(cfg.experiment.num_r):
        controller = MPController(env, model, cfg.policy)
        data_new = rollout(env, controller, cfg.experiment)
        rew = np.stack(data_new[2])

        X, dX, U = combine_data(data_new, (X, dX, U))
        msg = "Rollout completed of "
        msg += f"Cumulative reward {np.sum(np.stack(data_new[2]))}, "
        msg += f"Flight length {len(np.stack(data_new[2]))}"
        log.info(msg)

        plot_rollout(data_new[0])

        reward = np.sum(rew)
        trial_rewards.append(reward)

        trial_log = dict(
            env_name=cfg.env.params.name,
            seed=cfg.random_seed,
            trial_num=i,
            rewards=trial_rewards,
            nll=train_log,
        )
        save_log(cfg, i, trial_log)

        model, train_log = train_model(X, U, dX, cfg.model)


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


def plot_rollout(states):
    import plotly.graph_objects as go
    import numpy as np
    ar = np.stack(states)
    l = np.shape(ar)[0]
    xs = np.arange(l)
    yaw = ar[:,0]
    pitch = ar[:,1]
    roll = ar[:,2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=yaw, name='Yaw',
                             line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=xs, y=pitch, name='Pitch',
                             line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=xs, y=roll, name='Roll',
                             line=dict(color='green', width=4)))

    fig.update_layout(title='Euler Angles from MPC Rollout',
                      xaxis_title='Timestep',
                      yaxis_title='Angle (Degrees)',
                      plot_bgcolor='white',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True,),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True,),
                      )
    fig.show()


if __name__ == '__main__':
    sys.exit(mpc())
