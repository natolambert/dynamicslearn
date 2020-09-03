import plotly.io as pio

import hydra
from omegaconf import OmegaConf

import logging
import os
import sys
import glob
# from natsort import natsorted
import numpy as np
import torch
from collections import defaultdict

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
log = logging.getLogger(__name__)

from learn.utils.plotly import generate_errorbar_traces, plot_rewards_over_trials, hv_characterization, plot_sweep_1, plot_rollout_dat


######################################################################
@hydra.main(config_path='conf/plotting.yaml')
def plot(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")
    hv_characterization()
    quit()
    # Yaw control
    yaw_dir = "/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-05-13/08-01-09/metric.name=Yaw,robot=iono_sim/"
    ex = "0/trial_33.dat"
    yaw_ex = yaw_dir+ex
    # plot_sweep_1(yaw_dir)
    plot_rollout_dat(yaw_ex)
    quit()
    # dir=2020-02-10/15-39-36
    files = glob.glob(hydra.utils.get_original_cwd() + '/outputs/' + cfg.dir + '/*/**.dat')
    ms = []
    cl = []
    for g in files:
        mse, clust = torch.load(g)
        if clust < 500:
            continue
        ms.append(mse)
        cl.append(clust)

    # ms = np.array(ms)
    # cl = np.array(cl)

    # Non clustered data
    full_size = [4000]
    base = [0.6844194601919266,
            0.6426670856359498,
            0.6760970001662061,
            0.7867345088097977,
            0.6402819700817463,
            0.6432612884414582,
            0.614643476721318,
            0.673518857099874,
            0.5565854257191823,
            0.9437187183401807]
    #
    # for b in base:
    #     ms.append(b)
    #     cl.append(full_size[0])

    cl, ms = zip(*sorted(zip(cl, ms)))
    ids = np.unique(cl)

    cl_arr = np.stack(cl).reshape((len(ids), -1))
    ms_arr = np.stack(ms).reshape((len(ids), -1))

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    colors = plt.get_cmap('tab10').colors
    traces = []
    i = 1
    cs_str = 'rgb' + str(colors[i])

    err_traces, xs, ys = generate_errorbar_traces(ms_arr.T, xs=cl_arr.T.tolist(), color=cs_str,
                                                  name=f"Clustered Training")
    for t in err_traces:
        traces.append(t)

    layout = dict( #title=f"Test Set Prediction Error",  # (Env: {env_name})",
                  xaxis={'title': 'Cluster Size (Log Scale)',
                         'autorange': 'reversed',
                         'range':[3.7, 2.6]
                         },
                  yaxis={'title': 'Prediction Mean Squared Error',
                         'range':[.3,1]},
                  font=dict(family='Times New Roman', size=33, color='#7f7f7f'),
                  xaxis_type="log",
                  # yaxis_type="log",
                  height=600,
                  width=1300,
                  margin=dict(l=0, r=0, b=0, t=0),
                  plot_bgcolor='white',
                  legend={'x': .6, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'})


    fig = go.Figure(
        data=traces,
        layout=layout,
    )

    fig.add_trace(
        # Line Horizontal
        go.Scatter(
            mode="lines",
            x=[max(ids), min(ids)],
            y=[np.mean(base), np.mean(base)],
            line=dict(
                color="gray",
                width=4,
                dash="dashdot",
            ),
            name='Default Training (4000 Datapoints)'
        ))

    import plotly.io as pio
    pio.show(fig)
    fig.write_image('clustering_thin.pdf')
    quit()
    hv_characterization()

    ######################################################################
    logs = defaultdict(list)
    configs = defaultdict(list)
    logs_dirs = ['/Users/nol/Documents/code-bases/dynamicslearn/multirun/2019-12-16/20-01-04/', ]

    def load_log(directory, trial_file=None):
        if '.hydra' in os.listdir(directory):
            full_conf = OmegaConf.load(f"{directory}/.hydra/config.yaml")
        else:
            full_conf = OmegaConf.load(f"{directory}/config.yaml")
        trial_files = glob.glob(f"{directory}/trial_*.dat")
        if len(trial_files) > 1:
            if trial_file is not None:
                last_trial_log = f"{directory}/{trial_file}"
            else:
                last_trial_log = max(trial_files, key=os.path.getctime)
            vis_log = torch.load(last_trial_log)
            logs[log_dir].append(vis_log)
            configs[log_dir].append(full_conf)

    for log_dir in logs_dirs:
        if os.path.exists(os.path.join(log_dir, 'config.yaml')):
            log.info(f"Loading latest trial from {log_dir}")
            d = os.path.join(log_dir)
            load_log(d)
        else:
            # Assuming directory with multiple identical experiments (dir/0, dir/1 ..)
            latest = defaultdict(list)
            for ld in os.listdir(log_dir):
                directory = os.path.join(log_dir, ld)
                if os.path.isdir(directory):
                    trial_files = glob.glob(f"{directory}/trial_*.dat")
                    if len(trial_files) == 0:
                        continue
                    last_trial_log = max(trial_files, key=os.path.getctime)
                    last_trial_log = last_trial_log[len(directory) + 1:]
                    latest[log_dir].append(last_trial_log)

            for ld in os.listdir(log_dir):
                if ld == '.slurm': continue
                log_subdir = os.path.join(log_dir, ld)
                if os.path.isdir(log_subdir):
                    # Load data for the smallest trial number from all sub directories
                    if len(latest[log_dir]) == 0:
                        log.warn(f"No trial files found under {log_dir}")
                        break
                    trial_file = natsorted(latest[log_dir])[0]
                    load_log(log_subdir, trial_file)

    # To display the figure defined by this dict, use the low-level plotly.io.show function
    plot_rewards_over_trials(logs)


if __name__ == '__main__':
    sys.exit(plot())
