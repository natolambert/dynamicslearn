import os
import sys

#
# # For BO optimization
# import opto
# import opto.data as rdata
# from opto.opto.classes.OptTask import OptTask
# from opto.opto.classes import StopCriteria, Logs
# from opto.utils import bounds
# from opto.opto.acq_func import EI, UCB
# from opto import regression

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

from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax import ParameterType, FixedParameter, Arm, Metric, Runner, OptimizationConfig, Objective, Data


def save_log(cfg, trial_num, trial_log):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)


def squ_cost(state, action):
    pitch = state[0]
    roll = state[1]
    cost = pitch**2 + roll**2
    return cost

def living_reward(state, action):
    pitch = state[0]
    roll = state[1]
    flag1 = np.abs(pitch) < 5
    flag2 = np.abs(roll) < 5
    rew = int(flag1) + int(flag2)
    return rew

def rotation_mat(state, action):
    x0 = state
    rotn_matrix = np.array([[1., math.sin(x0[0]) * math.tan(x0[1]), math.cos(x0[0]) * math.tan(x0[1])],
                            [0., math.cos(x0[0]), -math.sin(x0[0])],
                            [0., math.sin(x0[0]) / math.cos(x0[1]), math.cos(x0[0]) / math.cos(x0[1])]])
    return np.linalg.det(rotn_matrix)

def get_rewards(states, actions, fncs=[]):
    rews = [[] for _ in range(len(fncs))]
    for i, (s, a) in enumerate(zip(states, actions)):
        for f in fncs:
            rews[i].append(f(s,a))

    return rews

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

    # global max_cost
    # max_cost = 0.
    def bo_rollout_wrapper(params, weights=None):  # env, controller, exp_cfg):
        # pid_1 = pid_s.transform(np.array(params)[0, :3])
        # pid_2 = pid_s.transform(np.array(params)[0, 3:])
        pid_1 = [params["roll-p"], params["roll-i"],
                 params["roll-d"]]  # [params["pitch-p"], params["pitch-i"], params["pitch-d"]]
        pid_2 = [params["roll-p"], params["roll-i"], params["roll-d"]]
        print(f"Optimizing Parameters {np.round(pid_1, 3)},{np.round(pid_2, 3)}")
        pid_params = [[pid_1[0], pid_1[1], pid_1[2]], [pid_2[0], pid_2[1], pid_2[2]]]
        # pid_params = [[1000, 0, 0], [1000, 0, 0]]
        pid = PidPolicy(cfg)

        pid.set_params(pid_params)

        cum_cost = []
        r = 0
        fncs = [squ_cost, living_reward, rotation_mat]
        mult_rewards = [[] for _ in range(len(fncs))]
        while r < cfg.experiment.repeat:
            pid.reset()
            states, actions, rews, sim_error = rollout(env, pid, exp_cfg)
            # plot_rollout(states, actions, pry=[1, 0, 2])
            rewards_full = get_rewards(states, actions, fncs=fncs)
            if sim_error:
                print("Repeating strange simulation")
                continue
            if len(rews) < 400:
                cum_cost.append(-(cfg.experiment.r_len - len(rews)) / cfg.experiment.r_len)
            else:
                rollout_cost = np.sum(rews) / cfg.experiment.r_len  # / len(rews)
                # if rollout_cost > max_cost:
                #      max_cost = rollout_cost
                # rollout_cost += get_reward_euler(states[-1], actions[-1])
                cum_cost.append(rollout_cost)
            r += 1

        std = np.std(cum_cost)
        cum_cost = np.mean(cum_cost)
        # print(f"Cum. Cost {cum_cost / max_cost}")
        print(f"- Mean Cum. Cost / Rew: {cum_cost}, std dev: {std}")

        return cum_cost.reshape(1, 1), std

    from ax import (
        ComparisonOp,
        ParameterType,
        RangeParameter,
        SearchSpace,
        SimpleExperiment,
        OutcomeConstraint,
    )

    exp = SimpleExperiment(
        name="PID Control Robot",
        search_space=SearchSpace([
            RangeParameter(
                name=f"roll-p", parameter_type=ParameterType.FLOAT, lower=10, upper=500.0, log_scale=True,
            ),
            # FixedParameter(name="roll-i", value=0.0, parameter_type=ParameterType.FLOAT),
            RangeParameter(
                name=f"roll-i", parameter_type=ParameterType.FLOAT, lower=0, upper=10.0, log_scale=False,
            ),
            RangeParameter(
                name=f"roll-d", parameter_type=ParameterType.FLOAT, lower=0, upper=10.0, log_scale=False,
            ),

            # RangeParameter(
            #     name=f"pitch-p", parameter_type=ParameterType.FLOAT, lower=1.0, upper=500.0, log_scale=True,
            # ),
            # RangeParameter(
            #     name=f"pitch-d", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            # ),
            # RangeParameter(
            #     name=f"pitch-i", parameter_type=ParameterType.FLOAT, lower=0.0, upper=500.0
            # ),
            # FixedParameter(name="pitch-i", value=0.0, parameter_type=ParameterType.FLOAT),

        ]),
        evaluation_function=bo_rollout_wrapper,
        objective_name="Reward",
        minimize=False,
        outcome_constraints=[],
    )

    class PIDMetric(Metric):
        def fetch_trial_data(self, trial):
            records = []
            for arm_name, arm in trial.arms_by_name.items():
                params = arm.parameters
                mean, sem = bo_rollout_wrapper(params)
                records.append({
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "mean": mean,
                    "sem": sem,
                    "trial_index": trial.index,
                })
            return Data(df=pd.DataFrame.from_records(records))

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=PIDMetric(name="base"),
            minimize=False,
        ),
    )

    class MyRunner(Runner):
        def run(self, trial):
            return {"name": str(trial.index)}

    exp.runner = MyRunner()
    exp.optimization_config = optimization_config

    from ax.plot.trace import optimization_trace_single_method
    from ax.utils.notebook.plotting import render, init_notebook_plotting
    from ax.plot.contour import plot_contour

    print(f"Running Sobol initialization trials...")
    sobol = Models.SOBOL(exp.search_space)
    num_search = 20
    for i in range(num_search):
        exp.new_trial(generator_run=sobol.gen(1))
        exp.trials[len(exp.trials) - 1].run()

    # data = exp.fetch_data()
    gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
    plot = plot_contour(model=gpei,
                        param_x="roll-p",
                        param_y="roll-d",
                        metric_name="base", )
    data = plot[0]['data']
    lay = plot[0]['layout']

    render(plot)

    num_opt = 50
    for i in range(num_opt):
        print(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
        # Reinitialize GP+EI model at each step with updated data.
        batch = exp.new_trial(generator_run=gpei.gen(1))
        gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

        if (i % 5) == 0:
            plot = plot_contour(model=gpei,
                                param_x="roll-p",
                                param_y="roll-d",
                                metric_name="base", )
            data = plot[0]['data']
            lay = plot[0]['layout']

            render(plot)

    objective_means = np.array([[exp.trials[trial].objective_mean] for trial in exp.trials])
    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(objective_means.T, axis=1),
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )
    render(best_objective_plot)

    plot = plot_contour(model=gpei,
                        param_x="roll-p",
                        param_y="pitch-p",
                        metric_name="base", )
    data = plot[0]['data']
    lay = plot[0]['layout']

    import plotly.graph_objects as go
    fig = {
        "data": data,
        "layout": lay,
    }
    # go.Figure(fig).write_image("test.pdf")

    render(plot)

    plot2 = plot_contour(model=gpei,
                         param_x="roll-d",
                         param_y="pitch-d",
                         metric_name="base", )
    data = plot2[0]['data']
    lay = plot2[0]['layout']

    import plotly.graph_objects as go
    # fig2 = {
    #     "data": data,
    #     "layout": lay,
    # }
    # go.Figure(fig).write_image("test.pdf")

    render(plot2)

    plot2 = plot_contour(model=gpei,
                         param_x="roll-i",
                         param_y="pitch-i",
                         metric_name="base", )
    data = plot2[0]['data']
    lay = plot2[0]['layout']

    import plotly.graph_objects as go
    # fig2 = {
    #     "data": data,
    #     "layout": lay,
    # }
    # go.Figure(fig).write_image("test.pdf")

    render(plot2)

    log.info(f"Running random rollout for cost baseline")

    msg = "Initialized BO Objective of PID Control"
    log.info(msg)


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
        #     print(action)
        # if update:
        states.append(state)
        actions.append(action)

        state, rew, done, _ = env.step(action)
        sim_error = euler_numer(last_state, state)
        done = done or sim_error
        # if update:
        rews.append(rew)

    return states, actions, rews, sim_error


def euler_numer(last_state, state, mag=5):
    flag = False
    if abs(state[0] - last_state[0]) > mag:
        flag = True
    elif abs(state[1] - last_state[1]) > mag:
        flag = True
    elif abs(state[2] - last_state[2]) > mag:
        flag = True
    # if flag:
    #     print("Stopping - Large  euler angle step detected, likely non-physical")
    return False  #


if __name__ == '__main__':
    sys.exit(pid())
