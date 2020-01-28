import os
import sys
from dotmap import DotMap

# For BO optimization
import opto
import opto.data as rdata
from opto.opto.classes.OptTask import OptTask
from opto.opto.classes import StopCriteria, Logs
from opto.utils import bounds
from opto.opto.acq_func import EI, UCB
from opto import regression

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import torch
import math

from learn.control.random import RandomController
from learn.control.mpc import MPController
from learn.control.pid import PidPolicy
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


class simple_bo():
    def __init__(self, bo_cfg, policy_cfg, opt_function):
        # self.Objective = SimulationOptimizer(bo_cfg, policy_cfg)
        self.b_cfg = bo_cfg
        self.p_cfg = policy_cfg

        self.PIDMODE = policy_cfg.mode
        self.policy = PidPolicy(policy_cfg)
        evals = bo_cfg.iterations
        param_min = list(policy_cfg.pid.params.min_values)
        param_max = list(policy_cfg.pid.params.max_values)
        self.n_parameters = self.policy.numParameters
        self.n_pids = self.policy.numpids
        params_per_pid = self.n_parameters / self.n_pids
        assert params_per_pid % 1 == 0
        params_per_pid = int(params_per_pid)

        self.task = OptTask(f=opt_function, n_parameters=self.n_parameters, n_objectives=1,
                            bounds=bounds(min=param_min * params_per_pid,
                                          max=param_max * params_per_pid), task={'minimize'},
                            vectorized=False)
        # labels_param = ['KP_pitch','KI_pitch','KD_pitch', 'KP_roll' 'KI_roll', 'KD_roll', 'KP_yaw', 'KI_yaw', 'KD_yaw', 'KP_pitchRate', 'KI_pitchRate', 'KD_pitchRate', 'KP_rollRate',
        # 'KI_rollRate', 'KD_rollRate', "KP_yawRate", "KI_yawRate", "KD_yawRate"])
        self.Stop = StopCriteria(maxEvals=evals)

    def optimize(self):
        p = DotMap()
        p.verbosity = 1
        p.acq_func = EI(model=None, logs=None)
        # p.acq_func = UCB(model=None, logs=None)
        p.model = regression.GP
        self.opt = opto.BO(parameters=p, task=self.task, stopCriteria=self.Stop)
        self.opt.optimize()

    def getParameters(self):
        log = self.opt.get_logs()
        losses = log.get_objectives()
        best = log.get_best_parameters()
        bestLoss = log.get_best_objectives()
        nEvals = log.get_n_evals()
        best = [matrix.tolist() for matrix in best]  # can be a buggy line

        print("Best PID parameters found with loss of: ", np.amin(bestLoss), " in ", nEvals, " evaluations.")
        print("Pitch:   Prop: ", best[0], " Int: ", 0, " Deriv: ", best[1])
        print("Roll:    Prop: ", best[2], " Int: ", 0, " Deriv: ", best[3])

        return log

    def basic_rollout(self, s0, i_model):
        # todo need to accound for history automatically
        max_len = self.b_cfg.max_length
        cur_action = self.policy.get_action(s0)
        next_state, logvars = smart_model_step(i_model, s0, cur_action)
        state = push_history(next_state, s0)
        cost = 0
        for k in range(max_len):
            # print(f"Itr {k}")
            # print(f"Action {cur_action.tolist()}")
            # print(f"State {next_state.tolist()}")
            cur_action = self.policy.get_action(next_state)
            next_state, logvars = smart_model_step(i_model, state, cur_action)
            state = push_history(next_state, state)
            # print(f"logvars {logvars}")
            # weight = 0 if k < 5 else 1
            weight = .9 ** (max_len - k)
            cost += weight * get_reward_iono(next_state, cur_action)

        return cost / max_len  # cost


######################################################################
@hydra.main(config_path='conf/mpc.yaml')
def pid(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    env_name = 'CrazyflieRigid-v0'
    env = gym.make(env_name)
    env.reset()
    full_rewards = []
    exp_cfg = cfg.experiment
    controller = PidPolicy(cfg.control)

    def bo_rollout_wrapper(params):  # env, controller, exp_cfg):
        print(f"PID Params: {params}")
        params = np.asarray(params)[0,:]
        pid_sets = [[params[0], 0, params[1]],
                    [params[2], 0, params[3]]]
        controller.set_params(pid_sets)
        states, actions, rews = rollout(env, controller, exp_cfg)
        cum_cost = -1 * np.sum(rews)  # for minimization
        print(f"Cum. Cost {cum_cost}")
        plot_rollout(states, actions)
        return cum_cost.reshape(1, 1)

    sim = simple_bo(cfg.bo, cfg.control, bo_rollout_wrapper)
    msg = "Initialized BO Objective of PID Control"
    log.info(msg)
    sim.optimize()

    for s in range(cfg.experiment.seeds):
        trial_rewards = []
        log.info(f"Random Seed: {s}")
        data = rollout(env, RandomController(env, cfg.policy), cfg.experiment)
        X, dX, U = to_XUdX(data)

        model, train_log = train_model(X, U, dX, cfg.model)

        for i in range(cfg.experiment.num_r):
            # controller = MPController(env, model, cfg.policy)
            controller.reset_params()
            data_new = rollout(env, controller, cfg.experiment)
            rew = np.stack(data_new[2])

            X, dX, U = combine_data(data_new, (X, dX, U))
            msg = "Rollout completed of "
            msg += f"Cumulative reward {np.sum(np.stack(data_new[2]))}, "
            msg += f"Flight length {len(np.stack(data_new[2]))}"
            log.info(msg)

            reward = np.sum(rew)
            reward = max(-10000, reward)
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
        full_rewards.append(trial_rewards)

    plot_rewards_over_trials(full_rewards, env_name)


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
        last_state = state
        if done:
            break
        action, update = controller.get_action(state)
        # if update:
        states.append(state)
        actions.append(action)

        state, rew, done, _ = env.step(action)
        sim_error = euler_numer(last_state, state)
        done = done or sim_error
        # if update:
        rews.append(rew)

    return states, actions, rews

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
    sys.exit(pid())
