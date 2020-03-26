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


from learn.simulate_sac import *
from learn.simulate_mpc import *
from learn.simulate_bopid import *


######################################################################
@hydra.main(config_path='conf/bopid.yaml')
def pid(cfg):
    env_name = cfg.env.params.name
    env = gym.make(env_name)
    env.reset()
    full_rewards = []
    exp_cfg = cfg.experiment

    def plot_results():
        import glob
        SAC_example = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-25/18-46-58/'
        MPC_example = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-25/19-48-27/'

        sweep_dirs = [SAC_example, MPC_example]
        alg_names = ['SAC', 'MPC']
        labels = []
        rewards = []
        samples = []
        algs = []
        # outer loop is control type
        for dir, alg in zip(sweep_dirs, alg_names):
            sub_dirs = os.listdir(dir)
            # This sweeps across the reward functions eg
            for sub in sub_dirs:
                # this
                path = os.path.join(dir, sub)
                seeds = glob.glob(path + "/*")
                # This sweeps across the random seeds
                seed_r = []
                seed_samples = []
                for s in seeds:
                    trials = glob.glob(s + '/trial_*')
                    trials.sort(key=lambda x: os.path.getmtime(x))
                    data = torch.load(trials[-1])
                    seed_r.append(data['rewards'])
                    seed_samples.append(data['steps'])
                rewards.append(seed_r)
                samples.append(seed_samples)
                algs.append(alg)
                labels.append(sub[sub.find("name=") + len("name="):sub.rfind(',')])

        import plotly
        import plotly.graph_objects as go

        unique_labels = np.unique(labels)
        figs = []
        for u in unique_labels:
            fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                                # subplot_titles=("Pitch Responses", "Roll Responses"),
                                                # vertical_spacing=.15,
                                                shared_xaxes=False, )  # go.Figure()
            idx = np.argwhere(np.array(labels) == u)
            for i in idx.squeeze():
                to_plot_rew = rewards[i]
                to_plot_samples = samples[i]
                alg = algs[i]
                lab = labels[i]

                min_len = min([len(r) for r in to_plot_rew])
                to_plot_rew = np.stack([t[:min_len] for t in to_plot_rew]).squeeze()
                to_plot_samples = np.stack([t[:min_len] for t in to_plot_samples]).squeeze()

                rew_mean = np.mean(np.stack(to_plot_rew).squeeze(), axis=0)
                rew_std = np.std(np.stack(to_plot_rew).squeeze(), axis=0)
                samp_mean = np.mean(np.stack(to_plot_samples).squeeze(), axis=0)
                samp_std = np.std(np.stack(to_plot_samples).squeeze(), axis=0)

                fig.add_trace(go.Scatter(y=rew_mean, x=samp_mean, name=alg + lab, legendgroup=alg,
                                         error_y=dict(
                                             type='data',  # value of error bar given in data coordinates
                                             array=rew_std,
                                             visible=True),
                                         error_x=dict(
                                             type='data',  # value of error bar given in data coordinates
                                             array=samp_std,
                                             visible=True)
                                         # line=dict(color=colors[i], width=2),  # mode='lines+markers',
                                         # marker=dict(color=colors[i], symbol=markers[i], size=16)
                                         ), row=1, col=1)

            fig.update_layout(title=f"Sample Efficiency vs Reward, {u}",
                              font=dict(
                                  family="Times New Roman, Times, serif",
                                  size=24,
                                  color="black"
                              ),
                              # legend_orientation="h",
                              # legend=dict(x=.05, y=0.5,
                              #             bgcolor='rgba(205, 223, 212, .4)',
                              #             bordercolor="Black",
                              #             ),
                              # xaxis_title='Timestep',
                              # yaxis_title='Angle (Degrees)',
                              plot_bgcolor='white',
                              width=1000,
                              height=1000,
                              xaxis=dict(
                                  showline=True,
                                  showgrid=False,
                                  showticklabels=True, ),
                              yaxis=dict(
                                  showline=True,
                                  showgrid=False,
                                  showticklabels=True, ),
                              )
            fig.show()
        return

    plot_results()
    quit()

    def compare_control(env, cfg):
        import torch
        from learn.control.pid import PidPolicy

        controllers = []
        labels = []

        # from learn.simulate_sac import *
        # Rotation policy
        sac_policy1 = torch.load(
            '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/outputs/2020-03-24/18-32-26/trial_70000.dat')
        controllers.append(sac_policy1['policy'])
        labels.append("SAC - Rotation")

        # Living reward policy
        sac_policy2 = torch.load(
            '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/outputs/2020-03-24/18-31-45/trial_35000.dat')
        controllers.append(sac_policy2['policy'])
        labels.append("SAC - Living")

        # Optimized PID parameters
        pid_params = [[2531.917, 61.358, 33.762], [2531.917, 61.358, 33.762]]
        pid = PidPolicy(cfg)
        pid.set_params(pid_params)
        controllers.append(pid)
        labels.append("PID - temp")

        from learn.control.mpc import MPController
        cfg.policy.mode = 'mpc'
        dynam_model = torch.load(
            '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/outputs/2020-03-25/10-45-17/trial_1.dat')
        mpc = MPController(env, dynam_model['model'], cfg)

        controllers.append(mpc)
        labels.append("MPC - temp")

        import plotly.graph_objects as go
        import plotly

        colors = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf'  # blue-teal
        ]

        markers = [
            "cross-open-dot",
            "circle-open-dot",
            "x-open-dot",
            "triangle-up-open-dot",
            "y-down-open",
            "diamond-open-dot",
            "hourglass-open",
            "hash-open-dot",
            "star-open-dot",
            "square-open-dot",
        ]

        if cfg.metric.name == 'Living':
            metric = living_reward
        elif cfg.metric.name == 'Rotation':
            metric = rotation_mat
        elif cfg.metric.name == 'Square':
            metric = squ_cost
        else:
            raise ValueError("Improper metric name passed")

        fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                            # subplot_titles=("Pitch Responses", "Roll Responses"),
                                            # vertical_spacing=.15,
                                            shared_xaxes=True, )  # go.Figure()

        pry = [1, 0, 2]
        # state0 = 2*env.reset()
        state0 = env.reset()
        for i, (con, lab) in enumerate(zip(controllers, labels)):
            print(f"Evaluating controller type {lab}")
            _ = env.reset()
            env.set_state(np.concatenate((np.zeros(6), state0)))
            state = state0
            states = []
            actions = []
            rews = []
            done = False
            for t in range(cfg.experiment.r_len + 1):
                if done:
                    break
                if "SAC" in lab:
                    with torch.no_grad():
                        with eval_mode(con):
                            action = con.select_action(state)
                            action = np.array([65535, 65535, 65535, 65535]) * (action + 1) / 2

                else:
                    action = con.get_action(state, metric=metric)
                states.append(state)
                actions.append(action)

                state, rew, done, _ = env.step(action)
                done = done

            states = np.stack(states)
            actions = np.stack(actions)

            pitch = np.degrees(states[:, pry[0]])
            roll = np.degrees(states[:, pry[1]])

            fig.add_trace(go.Scatter(y=roll, name=lab, legendgroup=lab,
                                     line=dict(color=colors[i], width=2),  # mode='lines+markers',
                                     marker=dict(color=colors[i], symbol=markers[i], size=16)
                                     ), row=1, col=1)

            fig.add_trace(go.Scatter(y=pitch, name=lab + "pitch", legendgroup=lab, showlegend=False,
                                     line=dict(color=colors[i], width=2),
                                     marker=dict(color=colors[i], symbol=markers[i], size=16)
                                     ), row=2, col=1)

        fig.update_xaxes(title_text="Timestep", row=2, col=1)
        # fig.update_xaxes(title_text="xaxis 1 title", row=1, col=1)
        fig.update_yaxes(title_text="Roll (Degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Pitch (Degrees)", row=2, col=1)

        fig.update_layout(title='Euler Angles from MPC Rollout',
                          font=dict(
                              family="Times New Roman, Times, serif",
                              size=24,
                              color="black"
                          ),
                          legend_orientation="h",
                          legend=dict(x=.05, y=0.5,
                                      bgcolor='rgba(205, 223, 212, .4)',
                                      bordercolor="Black",
                                      ),
                          # xaxis_title='Timestep',
                          # yaxis_title='Angle (Degrees)',
                          plot_bgcolor='white',
                          width=1600,
                          height=1000,
                          xaxis=dict(
                              showline=True,
                              showgrid=False,
                              showticklabels=True, ),
                          yaxis=dict(
                              showline=True,
                              showgrid=False,
                              showticklabels=True, ),
                          )
        print(f"Plotting {len(labels)} control responses")
        save = False
        if save:
            fig.write_image(os.getcwd() + "compare.png")
        else:
            fig.show()

        return fig

    compare_control(env, cfg)
    quit()
    pid_s = PID_scalar(cfg)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # Evalutation Function  # # # # # # # # # # # # # # # # # # # #
    def bo_rollout_wrapper(params, weights=None):  # env, controller, exp_cfg):
        # pid_1 = pid_s.transform(np.array(params)[0, :3])
        # pid_2 = pid_s.transform(np.array(params)[0, 3:])
        # pid_1 = [params["pitch-p"], params["pitch-i"], params["pitch-d"]]
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
