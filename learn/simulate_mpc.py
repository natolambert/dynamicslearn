import os
import sys
from dotmap import DotMap

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
from learn.utils.plotly import plot_rewards_over_trials, plot_rollout

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

    env_name = 'CrazyflieRigid-v0'
    env = gym.make(env_name)
    env.reset()
    full_rewards = []

    for s in range(cfg.experiment.seeds):
        trial_rewards = []
        log.info(f"Random Seed: {s}")
        rand_costs = []
        data_rand = []
        r = 0
        while r < cfg.experiment.repeat:
            data_r = rollout(env, RandomController(env, cfg), cfg.experiment)
            rews = data_r[-2]
            sim_error = data_r[-1]
            if sim_error:
                print("Repeating strange simulation")
                continue
            rand_costs.append(np.sum(rews))  # for minimization
            log.info(f" - Cost {np.sum(rews)}")
            r += 1
            data_rand.append(data_r)

            plot_rollout(data_r[0], data_r[1], pry=cfg.pid.params.pry, save=True, loc=f"/R_{r}")

        X, dX, U = to_XUdX(data_r)
        X, dX, U = combine_data(data_rand[:-1], (X, dX, U))
        msg = "Random Rollouts completed of "
        msg += f"Mean Cumulative reward {np.mean(rand_costs)}, "
        msg += f"Mean Flight length {cfg.policy.params.period * np.mean([np.shape(d[0])[0] for d in data_rand])}"
        log.info(msg)

        model, train_log = train_model(X, U, dX, cfg.model)

        for i in range(cfg.experiment.num_r):
            controller = MPController(env, model, cfg)

            r = 0
            cum_costs = []
            data_rs = []
            while r < cfg.experiment.repeat:
                data_r = rollout(env, controller, cfg.experiment)
                rews = data_r[-2]
                sim_error = data_r[-1]
                if sim_error:
                    print("Repeating strange simulation")
                    continue
                cum_costs.append(np.sum(rews))  # for minimization
                log.info(f" - Cost {np.sum(rews)}")
                r += 1
                data_rs.append(data_r)

                plot_rollout(data_r[0], data_r[1], pry=cfg.pid.params.pry, save=True, loc=f"/{str(i)}_{r}")

            X, dX, U = combine_data(data_rs, (X, dX, U))
            msg = "Rollouts completed of "
            msg += f"Mean Cumulative reward {np.mean(cum_costs)}, "
            msg += f"Mean Flight length {cfg.policy.params.period * np.mean([np.shape(d[0])[0] for d in data_rs])}"
            log.info(msg)

            reward = np.sum(rews)
            # reward = max(-10000, reward)
            trial_rewards.append(reward)

            trial_log = dict(
                env_name=cfg.env.params.name,
                model=model,
                seed=cfg.random_seed,
                raw_data=data_rs,
                trial_num=i,
                rewards=cum_costs,
                nll=train_log,
            )
            save_log(cfg, i, trial_log)

            model, train_log = train_model(X, U, dX, cfg.model)
            full_rewards.append(cum_costs)

        plot_rewards_over_trials(np.transpose(np.stack(full_rewards)), env_name)


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


def rollout(env, controller, exp_cfg):
    start = time.time()
    done = False
    states = []
    actions = []
    rews = []
    state = env.reset()
    for t in range(exp_cfg.r_len + 1):
        last_state = state
        if done:
            break
        action, update = controller.get_action(state)
        if update:
            states.append(state)
            actions.append(action)

        state, rew, done, _ = env.step(action)
        sim_error = euler_numer(last_state, state)
        done = done or sim_error
        if update:
            rews.append(rew)
    end = time.time()
    log.info(f"Rollout in {end - start} s, logged {len(rews)} (subsampled by control period)")
    return states, actions, rews, sim_error


def euler_numer(last_state, state):
    flag = False
    if abs(state[3] - last_state[3]) > 5:
        flag = True
    elif abs(state[4] - last_state[4]) > 5:
        flag = True
    elif abs(state[5] - last_state[5]) > 5:
        flag = True
    if flag:
        print("Stopping - Large euler angle step detected, likely non-physical")
    return flag


if __name__ == '__main__':
    sys.exit(mpc())
