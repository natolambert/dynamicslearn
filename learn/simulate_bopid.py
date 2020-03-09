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
from learn.utils.bo import get_reward_euler, plot_cost_itr, plot_parameters, PID_scalar

log = logging.getLogger(__name__)


def save_log(cfg, trial_num, trial_log):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)


class simple_bo():
    def __init__(self, cfg, opt_function):
        # self.Objective = SimulationOptimizer(bo_cfg, policy_cfg)
        self.b_cfg = cfg.bo
        self.p_cfg = cfg.policy
        self.cfg = cfg

        self.t_c = self.p_cfg.pid.params.terminal_cost
        self.l_c = self.p_cfg.pid.params.living_cost

        self.norm_cost = 1

        self.PIDMODE = cfg.policy.mode
        self.policy = PidPolicy(cfg)
        evals = cfg.bo.iterations
        param_min = [0] * len(list(cfg.pid.params.min_values))
        param_max = [1] * len(list(cfg.pid.params.max_values))
        self.n_parameters = self.policy.numParameters
        self.n_pids = self.policy.numpids
        params_per_pid = self.n_parameters / self.n_pids
        assert params_per_pid % 1 == 0
        params_per_pid = int(params_per_pid)

        self.task = OptTask(f=opt_function, n_parameters=self.n_parameters, n_objectives=1,
                            bounds=bounds(min=param_min * self.n_pids,
                                          max=param_max * self.n_pids), task={'minimize'},
                            vectorized=False)
        # labels_param = ['KP_pitch','KI_pitch','KD_pitch', 'KP_roll' 'KI_roll', 'KD_roll', 'KP_yaw', 'KI_yaw', 'KD_yaw', 'KP_pitchRate', 'KI_pitchRate', 'KD_pitchRate', 'KP_rollRate',
        # 'KI_rollRate', 'KD_rollRate', "KP_yawRate", "KI_yawRate", "KD_yawRate"])
        self.Stop = StopCriteria(maxEvals=evals)
        self.sim = cfg.bo.sim

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

    # def basic_rollout(self, s0, i_model):
    #     # todo need to accound for history automatically
    #     max_len = self.b_cfg.max_length
    #     cur_action = self.policy.get_action(s0)
    #     next_state, logvars = smart_model_step(i_model, s0, cur_action)
    #     state = push_history(next_state, s0)
    #     cost = 0
    #     for k in range(max_len):
    #         # print(f"Itr {k}")
    #         # print(f"Action {cur_action.tolist()}")
    #         # print(f"State {next_state.tolist()}")
    #         cur_action = self.policy.get_action(next_state)
    #         next_state, logvars = smart_model_step(i_model, state, cur_action)
    #         state = push_history(next_state, state)
    #         # print(f"logvars {logvars}")
    #         # weight = 0 if k < 5 else 1
    #         weight = .9 ** (max_len - k)
    #         cost += weight * get_reward_iono(next_state, cur_action)
    #
    #     return cost / max_len  # cost


######################################################################
@hydra.main(config_path='conf/mpc.yaml')
def pid(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    env_name = cfg.env.params.name
    env = gym.make(env_name)
    env.reset()
    full_rewards = []
    exp_cfg = cfg.experiment

    pid_s = PID_scalar(cfg)

    def investigate_env_orientation(env, action, k):
        for i in range(k):
            state, rew, done, _ = env.step(action)
            state = np.round(state, 2)
            if i % 5 == 0: print(f" Euler Angles Pitch {state[1]}, Roll {state[0]}, Yaw {state[2]}")

    # global max_cost
    # max_cost = 0.


    def bo_rollout_wrapper(params):  # env, controller, exp_cfg):
        pid_1 = pid_s.transform(np.array(params)[0, :3])
        pid_2 = pid_s.transform(np.array(params)[0, 3:])
        print(f"Optimizing Parameters {np.round(pid_1, 3)},{np.round(pid_2, 3)}")
        pid_params = [[pid_1[0], pid_1[1], pid_1[2]], [pid_2[0], pid_2[1], pid_2[2]]]
        sim.policy.set_params(pid_params)

        cum_cost = []
        r = 0
        while r < cfg.experiment.repeat:
            sim.policy.reset()
            states, actions, rews, sim_error = rollout(env, sim.policy, exp_cfg)
            # plot_rollout(states, actions, pry=[1, 0, 2])
            # if sim_error:
            #     print("Repeating strange simulation")
            #     continue
            rollout_cost = -1 * np.sum(rews) / cfg.experiment.repeat #/ len(rews)
            # if rollout_cost > max_cost:
            #      max_cost = rollout_cost
            # rollout_cost += get_reward_euler(states[-1], actions[-1])
            cum_cost.append(rollout_cost)
            r += 1

        cum_cost = np.mean(cum_cost)
        # print(f"Cum. Cost {cum_cost / max_cost}")
        print(f"Cum. Cost {cum_cost}")

        return cum_cost.reshape(1, 1)

    sim = simple_bo(cfg, bo_rollout_wrapper)

    log.info(f"Running random rollout for cost baseline")

    msg = "Initialized BO Objective of PID Control"
    log.info(msg)
    sim.optimize()

    logs = sim.getParameters()
    plot_cost_itr(logs, cfg)
    plot_parameters(logs, cfg, pid_s)

    print("\n Other items tried")
    for vals, fx in zip(np.array(logs.data.x.T), np.array(logs.data.fx.T)):
        vals = np.round(pid_s.transform(vals), 1)
        print(f"Cost - {np.round(fx, 3)}: Pp: ", vals[0], " Ip: ", vals[1], " Dp: ", vals[2],
              "| Pr: ", vals[3], " Ir: ", vals[4], " Dr: ", vals[5])


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

    # large terminal cost
    # rews[-1] = rews[-1] * exp_cfg.r_len

    return states, actions, rews, sim_error


def euler_numer(last_state, state, mag=10):
    flag = False
    if abs(state[0] - last_state[0]) > mag:
        flag = True
    elif abs(state[1] - last_state[1]) > mag:
        flag = True
    elif abs(state[2] - last_state[2]) > mag:
        flag = True
    # if flag:
        # print("Stopping - Large euler angle step detected, likely non-physical")
    return False #flag


if __name__ == '__main__':
    sys.exit(pid())
