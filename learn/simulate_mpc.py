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
@hydra.main(config_path='conf/mpc.yaml')
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
    log.info(f"Running experiment with interial prop x:{env.Ixx}, y:{env.Iyy}")

    # full_rewards = []
    # temp = hydra.utils.get_original_cwd() + '/outputs/2020-07-11/17-17-05/trial_3.dat'
    # dat = torch.load(temp)
    # actions = dat['raw_data'][0][1]
    # l = []
    #
    # yaw_actions = np.array([
    #     [1500, 1500, 1500, 1500],
    #     [2000, 1000, 1000, 2000],
    #     [1000, 2000, 2000, 1000],
    #     [2000, 2000, 1000, 1000],
    #     [1000, 1000, 2000, 2000],
    # ])
    #
    # def find_ind(arr):
    #     if np.all(np.equal(arr, [1500, 1500, 1500, 1500])):
    #         return 0
    #     elif np.all(np.equal(arr, [2000, 1000, 1000, 2000])):
    #         return 1
    #     elif np.all(np.equal(arr, [1000, 2000, 2000, 1000])):
    #         return 3
    #     elif np.all(np.equal(arr, [2000, 2000, 1000, 1000])):
    #         return 2
    #     else: # [1000, 1000, 2000, 2000]
    #         return 4
    #
    # for act in actions:
    #     act = act.numpy()
    #     id = find_ind(act)
    #     l.append(id)
    #
    # initial = l[:24]
    # states = dat['raw_data'][0][0][:25]
    # yaw_value = np.rad2deg(states[-1][0])-np.rad2deg(states[0][0])
    # print(f"Yaw after 25 steps{yaw_value}")
    # plot_lie(initial)
    # # plot_rollout(np.stack(dat['raw_data'][0][0])[:500,:3], dat['raw_data'][0][1], loc="/yaw_plt", save=True, only_x=True, legend=False)
    # quit()

    if cfg.metric.name == 'Living':
        metric = living_reward
        log.info(f"Using metric living reward")
    elif cfg.metric.name == 'Rotation':
        metric = rotation_mat
        log.info(f"Using metric rotation matrix")
    elif cfg.metric.name == 'Square':
        metric = squ_cost
        log.info(f"Using metric square cost")
    elif cfg.metric.name == 'Yaw':
        metric = yaw_r
        log.info(f"Using metric yaw sliding mode")
    elif cfg.metric.name == 'Yaw2':
        metric = yaw_r2
        log.info(f"Using metric yaw base")
    elif cfg.metric.name == 'Yaw3':
        metric = yaw_r3
        log.info(f"Using metric yaw rate")
    else:
        raise ValueError("Improper metric name passed")

    for s in range(cfg.experiment.seeds):
        log.info(f"Random Seed: {s}")
        total_costs = []
        data_rand = []
        total_steps = []
        r = 0
        while r < cfg.experiment.random:
            data_r = rollout(env, RandomController(env, cfg), cfg.experiment, metric=metric)
            plot_rollout(data_r[0], data_r[1], pry=cfg.pid.params.pry, save=cfg.save, loc=f"/R_{r}")
            rews = data_r[-2]
            sim_error = data_r[-1]
            if sim_error:
                print("Repeating strange simulation")
                continue
            # rand_costs.append(np.sum(rews) / len(rews))  # for minimization
            total_costs.append(np.sum(rews))  # for minimization
            # log.info(f" - Cost {np.sum(rews) / cfg.experiment.r_len}")
            r += 1

            # data_sample = subsample(data_r, cfg.policy.params.period)
            data_rand.append(data_r)
            total_steps.append(0)

        X, dX, U = to_XUdX(data_r)
        X, dX, U = combine_data(data_rand[:-1], (X, dX, U))
        msg = "Random Rollouts completed of "
        msg += f"Mean Cumulative reward {np.mean(total_costs)}, "
        msg += f"Mean Flight length {cfg.policy.params.period * np.mean([np.shape(d[0])[0] for d in data_rand])}"
        log.info(msg)
        last_yaw = np.max(np.abs(np.stack(data_r[0])[:,2])) #data_r[0][-1][2]

        trial_log = dict(
            env_name=cfg.env.params.name,
            # model=model,
            seed=cfg.random_seed,
            # raw_data=data_rs,
            yaw_num=last_yaw,
            trial_num=-1,
            rewards=total_costs,
            steps=total_steps,
            # nll=train_log,
        )
        save_log(cfg, -1, trial_log)

        model, train_log = train_model(X, U, dX, cfg.model)

        for i in range(cfg.experiment.num_roll - cfg.experiment.random):
            controller = MPController(env, model, cfg)

            r = 0
            cum_costs = []
            data_rs = []
            while r < cfg.experiment.repeat:
                data_r = rollout(env, controller, cfg.experiment, metric=metric)
                plot_rollout(data_r[0], data_r[1], pry=cfg.pid.params.pry, save=cfg.save, loc=f"/{str(i)}_{r}")
                rews = data_r[-2]
                sim_error = data_r[-1]

                if sim_error:
                    print("Repeating strange simulation")
                    continue
                # cum_costs.append(np.sum(rews) / len(rews))  # for minimization
                total_costs.append(np.sum(rews))  # for minimization
                # log.info(f" - Cost {np.sum(rews) / cfg.experiment.r_len}")
                r += 1

                # data_sample = subsample(data_r, cfg.policy.params.period)
                data_rs.append(data_r)
                total_steps.append(np.shape(X)[0])

            X, dX, U = combine_data(data_rs, (X, dX, U))
            msg = "Rollouts completed of "
            msg += f"Mean Cumulative reward {np.mean(total_costs)}, "  # / cfg.experiment.r_len
            msg += f"Mean Flight length {cfg.policy.params.period * np.mean([np.shape(d[0])[0] for d in data_rs])}"
            log.info(f"Final yaw {180*np.array(data_r[0][-1][2])/np.pi}")
            log.info(msg)
            last_yaw = np.max(np.abs(np.stack(data_r[0])[:,2])) #data_r[0][-1][2]

            trial_log = dict(
                env_name=cfg.env.params.name,
                # model=model,
                seed=cfg.random_seed,
                # raw_data=data_rs,
                yaw_num=last_yaw,
                trial_num=i,
                rewards=total_costs,
                steps=total_steps,
                nll=train_log,
            )
            save_log(cfg, i, trial_log)

            model, train_log = train_model(X, U, dX, cfg.model)

        fig = plot_rewards_over_trials(np.transpose(np.stack([total_costs])), env_name, save=True)
        fig.write_image(os.getcwd() + "/learning-curve.pdf")


# def subsample(rollout, period):
#     """
#     Subsamples the rollout data for training a dynamics model with
#     :param rollout: data from the rollout function
#     :param period: from the robot cfg, how frequent control updates vs sim
#     :return:
#     """
#     l = [r[::period] for i, r in enumerate(rollout) if i < 3]
#     # l = l.append([rollout[-1]])
#     return l


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
