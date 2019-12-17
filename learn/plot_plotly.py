import plotly.io as pio

import hydra
from omegaconf import OmegaConf

import logging
import os
import sys
import glob
from natsort import natsorted

import torch
from collections import defaultdict


# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
log = logging.getLogger(__name__)

from learn.utils.plotly import plot_rewards_over_trials
######################################################################
@hydra.main(config_path='conf/plotting.yaml')
def plot(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    ######################################################################
    logs = defaultdict(list)
    configs = defaultdict(list)
    logs_dirs = ['/Users/nol/Documents/code-bases/dynamicslearn/multirun/2019-12-16/20-01-04/',]

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
