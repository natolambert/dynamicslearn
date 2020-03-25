import os
import sys

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
from learn.utils.sim import *

log = logging.getLogger(__name__)


def save_log(cfg, exp, trial_log, ax=False):
    trial_num = 0
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)
    save(exp, os.path.join(os.getcwd(), "exp.json"))


from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax import save, ParameterType, FixedParameter, Arm, Metric, Runner, OptimizationConfig, Objective, Data

from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour


def plot_learning(exp, cfg):
    objective_means = np.array([[exp.trials[trial].objective_mean] for trial in exp.trials])
    cumulative = optimization_trace_single_method(
        y=np.maximum.accumulate(objective_means.T, axis=1), ylabel=cfg.metric.name,
        trace_color=(83, 78, 194),
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )
    all = optimization_trace_single_method(
        y=objective_means.T, ylabel=cfg.metric.name,
        model_transitions=[cfg.bo.random], trace_color=(114, 110, 180),
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )
    layout_learn = cumulative[0]['layout']
    layout_learn['paper_bgcolor'] = 'rgba(0,0,0,0)'
    layout_learn['plot_bgcolor'] = 'rgba(0,0,0,0)'

    d1 = cumulative[0]['data']
    d2 = all[0]['data']

    for t in d1:
        t['legendgroup'] = cfg.metric.name + ", cum. max"
        if 'name' in t and t['name'] == 'Generator change':
            t['name'] = 'End Random Iterations'
        else:
            t['name'] = cfg.metric.name + ", cum. max"

    for t in d2:
        t['legendgroup'] = cfg.metric.name
        if 'name' in t and t['name'] == 'Generator change':
            t['name'] = 'End Random Iterations'
        else:
            t['name'] = cfg.metric.name

    fig = {
        "data": d1 + d2,  # data,
        "layout": layout_learn,
    }
    import plotly.graph_objects as go
    return go.Figure(fig)


def get_rewards(states, actions, fncs=[]):
    rews = [[] for _ in range(len(fncs))]
    for (s, a) in zip(states, actions):
        for i, f in enumerate(fncs):
            rews[i].append(f(s, a))

    return rews


######################################################################
@hydra.main(config_path='conf/bopid.yaml')
def pid(cfg):
    env_name = cfg.env.params.name
    env = gym.make(env_name)
    env.reset()
    full_rewards = []
    exp_cfg = cfg.experiment

    pid_s = PID_scalar(cfg)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # Evalutation Function  # # # # # # # # # # # # # # # # # # # #
    def bo_rollout_wrapper(params, weights=None):  # env, controller, exp_cfg):
        # pid_1 = pid_s.transform(np.array(params)[0, :3])
        # pid_2 = pid_s.transform(np.array(params)[0, 3:])
        # pid_1 = [params["pitch-p"], params["pitch-i"], params["pitch-d"]]
        pid_1 = [params["roll-p"], params["roll-i"], params["roll-d"]] #[params["pitch-p"], params["pitch-i"], params["pitch-d"]]
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
            for i, vec in enumerate(rewards_full):
                mult_rewards[i].append(vec)

            if sim_error:
                print("Repeating strange simulation")
                continue
            # if len(rews) < 400:
            #     cum_cost.append(-(cfg.experiment.r_len - len(rews)) / cfg.experiment.r_len)
            # else:
            rollout_cost = np.sum(rews) / cfg.experiment.r_len  # / len(rews)
            # if rollout_cost > max_cost:
            #      max_cost = rollout_cost
            # rollout_cost += get_reward_euler(states[-1], actions[-1])
            cum_cost.append(rollout_cost)
            r += 1

        std = np.std(cum_cost)
        cum_cost = np.mean(cum_cost)
        # print(f"Cum. Cost {cum_cost / max_cost}")
        # print(f"- Mean Cum. Cost / Rew: {cum_cost}, std dev: {std}")
        eval = {"Square": (-np.mean(rewards_full[0]), np.std(rewards_full[0])),
                "Living": (np.mean(rewards_full[1]), np.std(rewards_full[1])),
                "Rotation": (np.mean(rewards_full[2]), np.std(rewards_full[2]))}

        for n, (key, value) in enumerate(eval.items()):
            if n == 0:
                print(f"- Square {np.round(value, 4)}")
            elif n == 1:
                print(f"- Living {np.round(value, 4)}")
            else:
                print(f"- Rotn {np.round(value, 4)}")
        return eval
        # return cum_cost.reshape(1, 1), std

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
                name=f"roll-p", parameter_type=ParameterType.FLOAT, lower=1.0, upper=5000.0, log_scale=True,
            ),
            # FixedParameter(name="roll-i", value=0.0, parameter_type=ParameterType.FLOAT),
            RangeParameter(
                name=f"roll-i", parameter_type=ParameterType.FLOAT, lower=0, upper=100.0, log_scale=False,
            ),
            RangeParameter(
                name=f"roll-d", parameter_type=ParameterType.FLOAT, lower=1, upper=5000.0, log_scale=True,
            ),

            # RangeParameter(
            #     name=f"pitch-p", parameter_type=ParameterType.FLOAT, lower=1.0, upper=5000.0, log_scale=True,
            # ),
            # RangeParameter(
            #     name=f"pitch-d", parameter_type=ParameterType.FLOAT, lower=0.0, upper=100.0
            # ),
            # RangeParameter(
            #     name=f"pitch-i", parameter_type=ParameterType.FLOAT, lower=0.0, upper=100.0
            # ),
            # FixedParameter(name="pitch-i", value=0.0, parameter_type=ParameterType.FLOAT),

        ]),
        evaluation_function=bo_rollout_wrapper,
        objective_name=cfg.metric.name,
        minimize=cfg.metric.minimize,
        outcome_constraints=[],
    )

    from ax.storage.metric_registry import register_metric
    from ax.storage.runner_registry import register_runner

    class GenericMetric(Metric):
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

    class MyRunner(Runner):
        def run(self, trial):
            return {"name": str(trial.index)}

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=GenericMetric(name=cfg.metric.name),
            minimize=cfg.metric.minimize,
        ),
    )
    register_metric(GenericMetric)
    register_runner(MyRunner)

    exp.runner = MyRunner()
    exp.optimization_config = optimization_config

    print(f"Running Sobol initialization trials...")
    sobol = Models.SOBOL(exp.search_space)
    num_search = cfg.bo.random
    for i in range(num_search):
        exp.new_trial(generator_run=sobol.gen(1))
        exp.trials[len(exp.trials) - 1].run()

    import plotly.graph_objects as go

    gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

    objectives = ["Living", "Square", "Rotation"]

    def plot_all(model, objectives, name="", rend=False):
        for o in objectives:
            plot = plot_contour(model=model,
                                param_x="roll-p",
                                param_y="roll-d",
                                metric_name=o, )
            plot[0]['layout']['title'] = o
            data = plot[0]['data']
            lay = plot[0]['layout']

            for i, d in enumerate(data):
                if i > 1:
                    d['cliponaxis'] = False

            fig = {
                "data": data,
                "layout": lay,
            }
            go.Figure(fig).write_image(name + o + ".png")
            if rend: render(plot)

    plot_all(gpei, objectives, name="Random fit-")

    num_opt = cfg.bo.optimized
    for i in range(num_opt):
        print(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
        # Reinitialize GP+EI model at each step with updated data.
        batch = exp.new_trial(generator_run=gpei.gen(1))
        gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

        if ((i + 1) % 10) == 0:
            plot_all(gpei, objectives, name=f"optimizing {str(i + 1)}-", rend=False)

    from ax.plot.exp_utils import exp_to_df

    best_arm, _ = batch.best_arm_predictions
    best_parameters = best_arm.parameters

    print(exp_to_df(exp=exp))
    experiment_log = {
        "Exp": exp_to_df(exp=exp),
        "Cfg": cfg,
        "Best_param": best_parameters,
    }

    log.info("Printing Parameters")
    print(exp_to_df(exp=exp))
    save_log(cfg, exp, experiment_log)

    plot_learning(exp, cfg).show()
    plot_all(gpei, objectives, name=f"FINAL -", rend=True)


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



if __name__ == '__main__':
    sys.exit(pid())
